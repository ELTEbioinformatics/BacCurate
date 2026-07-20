"""
Map host annotations from sample metadata to NCBI Taxonomy IDs and
scientific names (binomial nomenclature).

See docs/host.md for the documentation.
"""

import logging
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from baccurate.paths import DEFAULT_TAXIDS_NCBI
from baccurate.utils.config import load_config
from baccurate.utils.text import split_pipe_separated

logger = logging.getLogger(__name__)

# --- Match-quality scores ---

# Direct numeric taxid, scientific name or synonym
SCORE_TAXID = 1.0
SCORE_SCINAME = 1.0
SCORE_SYNONYM = 1.0

# Locally curated host term
SCORE_CURATED_TERM = 0.95

# NCBI genbank_common_name match
SCORE_CURATED_COMMON = 0.9

# NCBI common_name match - multiple per taxon and can apply to more taxa
SCORE_BROAD_COMMON = 0.7

# Subset matching
SCORE_SUBSET_MULTIWORD = 0.7
SCORE_SUBSET_SINGLEWORD = 0.5

# --- Attribute-name precedence ---

# Tiebreaker between candidates of equal score and equal taxonomic
# specificity. Lower wins.
ATTR_PRIORITY: dict[str, int] = {
    "host_taxid": 1,
    "host": 2,
}
ATTR_PRIORITY_DEFAULT = 3


def _attr_priority(attribute: str) -> int:
    return ATTR_PRIORITY.get(attribute.lower(), ATTR_PRIORITY_DEFAULT)


# --- Text normalization ---

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().replace("_", " ").replace("-", " ")
    text = text.translate(_PUNCT_TABLE)
    return _WHITESPACE_RE.sub(" ", text).strip()


@dataclass(frozen=True, slots=True)
class CuratedTaxonPolicy:
    """Validated manual matching policy for one NCBI taxon."""

    taxid: int
    scientific_name: str
    exact_terms: tuple[str, ...]
    subset_terms: tuple[str, ...]
    force_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HostPolicy:
    """Validated host-standardization policy compiled from host.yaml."""

    ignored_substrings: tuple[str, ...]
    isolation_source_keywords: tuple[str, ...]
    curated_taxa: tuple[CuratedTaxonPolicy, ...]
    value_rejections: tuple[str, ...]


def _require_mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    for item in value:
        if not _normalize_text(item):
            raise ValueError(f"{label} contains an empty normalized value")
    return value


def _reject_unknown_keys(config: Mapping, allowed: set[str], label: str) -> None:
    unknown = set(config) - allowed
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"unknown host config key in {label}: {names}")


def _parse_host_policy(config: Mapping, config_path: Path | str) -> HostPolicy:
    if config.get("schema_version") != 2:
        raise ValueError(f"{config_path}: schema_version must be 2")
    _reject_unknown_keys(
        config,
        {"schema_version", "normalization", "routing", "curated_taxa", "value_rejections"},
        "top level",
    )

    normalization = _require_mapping(config.get("normalization"), "normalization")
    _reject_unknown_keys(normalization, {"ignored_substrings"}, "normalization")
    ignored_substrings = _require_string_list(
        normalization.get("ignored_substrings"), "normalization.ignored_substrings"
    )

    routing = _require_mapping(config.get("routing"), "routing")
    _reject_unknown_keys(routing, {"isolation_source_keywords"}, "routing")
    isolation_source_keywords = _require_string_list(
        routing.get("isolation_source_keywords"), "routing.isolation_source_keywords"
    )

    curated_taxa = _require_mapping(config.get("curated_taxa"), "curated_taxa")
    policies: list[CuratedTaxonPolicy] = []
    curated_term_taxids: dict[str, int] = {}
    for taxid, raw_taxon in curated_taxa.items():
        if not isinstance(taxid, str) or not re.fullmatch(r"[1-9]\d*", taxid):
            raise ValueError(f"curated_taxa key {taxid!r} must be a canonical quoted taxid")
        taxon = _require_mapping(raw_taxon, f"curated_taxa.{taxid}")
        _reject_unknown_keys(taxon, {"scientific_name", "match_terms"}, f"curated_taxa.{taxid}")
        scientific_name = taxon.get("scientific_name")
        if not isinstance(scientific_name, str) or not scientific_name.strip():
            raise ValueError(f"curated_taxa.{taxid}.scientific_name must be a non-empty string")
        match_terms = _require_mapping(
            taxon.get("match_terms"), f"curated_taxa.{taxid}.match_terms"
        )
        _reject_unknown_keys(
            match_terms,
            {"exact", "subset", "force"},
            f"curated_taxa.{taxid}.match_terms",
        )
        normalized_modes: dict[str, str] = {}
        terms_by_mode: dict[str, tuple[str, ...]] = {}
        for mode in ("exact", "subset", "force"):
            terms = _require_string_list(
                match_terms.get(mode, []), f"curated_taxa.{taxid}.match_terms.{mode}"
            )
            terms_by_mode[mode] = tuple(terms)
            for term in terms:
                normalized = _normalize_text(term)
                previous_mode = normalized_modes.setdefault(normalized, mode)
                if previous_mode != mode:
                    raise ValueError(
                        f"curated term {term!r} appears in both {previous_mode} and {mode} "
                        f"for taxid {taxid}"
                    )
                previous_taxid = curated_term_taxids.setdefault(normalized, int(taxid))
                if previous_taxid != int(taxid):
                    raise ValueError(
                        f"Curated term {normalized!r} maps to taxids {previous_taxid} and {taxid}."
                    )
        policies.append(
            CuratedTaxonPolicy(
                taxid=int(taxid),
                scientific_name=scientific_name,
                exact_terms=terms_by_mode["exact"],
                subset_terms=terms_by_mode["subset"],
                force_terms=terms_by_mode["force"],
            )
        )

    value_rejections = _require_mapping(config.get("value_rejections"), "value_rejections")
    _reject_unknown_keys(value_rejections, {"exact"}, "value_rejections")
    rejection_terms = _require_string_list(value_rejections.get("exact"), "value_rejections.exact")
    overlap = set(curated_term_taxids) & {_normalize_text(term) for term in rejection_terms}
    if overlap:
        raise ValueError(f"{sorted(overlap)[0]!r} is both a curated term and a value rejection")
    return HostPolicy(
        ignored_substrings=tuple(ignored_substrings),
        isolation_source_keywords=tuple(isolation_source_keywords),
        curated_taxa=tuple(policies),
        value_rejections=tuple(rejection_terms),
    )


# --- Data structures ---


@dataclass(frozen=True, slots=True)
class TaxonInfo:
    """One row of the taxid lookup table."""

    taxid: int
    scientific_name: str
    rank: str
    # Row index in the source TSV. The table is sorted from most-specific
    # (subspecies) to least-specific (genus and above), so lower numbers
    # mean a more specific taxon.
    table_priority: int


@dataclass(frozen=True, slots=True)
class ValueMatch:
    """Result of matching one normalized value against the lookup tables."""

    info: TaxonInfo
    score: float
    # "" for exact matches; "multi-word" or "single-word" for subset matches.
    match_tier: str = ""
    # Populated when subset matching found multiple distinct taxa. Empty
    # for unambiguous matches.
    tier_candidates: tuple[str, ...] = ()


class HostDiagnostic(StrEnum):
    """The fixed set of host-classification results used in build reports."""

    ISO_KEYWORD_PREEMPTION = "iso_keyword_preemption"
    OVERRIDE_REJECTION = "override_rejection"
    FORCED_OVERRIDE = "forced_override"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    SUBSET_MATCH = "subset_match"
    AMBIGUOUS_SUBSET = "ambiguous_subset"
    ATTRIBUTE_DISAGREEMENT = "attribute_disagreement"


@dataclass(frozen=True, slots=True)
class HostMatch:
    """Winning match for a record, with the (attribute, value) it came from."""

    info: TaxonInfo
    score: float
    source_index: int
    attribute: str
    value: str
    match_tier: str
    tier_candidates: tuple[str, ...]
    # True when worth reviewing:
    # any subset match, ambiguous subset, or cross-attribute disagreement.
    low_confidence: bool = False
    diagnostics: tuple[HostDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class StandardizedHost:
    """Taxonomic identity selected for one record."""

    taxid: int
    scientific_name: str


@dataclass(frozen=True, slots=True)
class HostOrigin:
    """One source attribute/value pair supporting a standardized host."""

    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class HostOverflowContext:
    """Unresolved host information reused as extra context in the isolation standardizer."""

    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class HostOutcome:
    """Record-level host result, including distinct absence and overflow states."""

    standardized: StandardizedHost | None
    score: float | None
    low_confidence: bool
    origins: tuple[HostOrigin, ...]
    overflow: HostOverflowContext | None
    diagnostics: tuple[HostDiagnostic, ...]
    retry_eligible: bool = False

    def __post_init__(self) -> None:
        if (self.standardized is None) != (self.score is None):
            raise ValueError("A standardized host and score must be present together")
        if self.standardized is not None and not self.origins:
            raise ValueError("A standardized host requires a source origin")
        if self.standardized is None and self.origins:
            raise ValueError("An absent standardized host cannot have source origins")
        if not self.diagnostics:
            raise ValueError("A host outcome requires at least one diagnostic")


# --- Main class ---


class HostStandardizer:
    def __init__(
        self,
        config_path: Path | str,
        ncbi_table_path: Path | str = DEFAULT_TAXIDS_NCBI,
        result_logger: logging.Logger | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.config = load_config(config_path)
        self.policy = _parse_host_policy(self.config, config_path)
        self._build_lookups(Path(ncbi_table_path))
        self._compile_filters()

    def _build_lookups(self, ncbi_table_path: Path) -> None:
        ncbi_df = pd.read_csv(
            ncbi_table_path,
            sep="\t",
            dtype={
                "comments": str,
                "genbank_common_name": str,
                "common_name": str,
                "synonym": str,
            },
        )

        # Lookups by source/score tier. The split exists because each
        # gets a different reliability score in the matching cascade.
        self.taxid_to_info: dict[str, TaxonInfo] = {}
        self.sciname_to_info: dict[str, TaxonInfo] = {}
        self.synonym_to_info: dict[str, TaxonInfo] = {}
        self.curated_term_to_info: dict[str, TaxonInfo] = {}
        self.curated_common_to_info: dict[str, TaxonInfo] = {}
        self.broad_common_to_info: dict[str, TaxonInfo] = {}

        # Subset matching, separated by term arity. The multi-word index
        # supports word-set lookups via per-word inverted index; single-word
        # terms are looked up directly.
        self.multiword_term_to_info: dict[str, TaxonInfo] = {}
        self.singleword_term_to_info: dict[str, TaxonInfo] = {}
        self.multiword_inverted_index: dict[str, set[str]] = {}

        # Higher-precision sources fill each lookup first via setdefault,
        # so collisions resolve in favor of the more authoritative source.
        # Order: scinames -> synonyms -> curated terms -> NCBI commons -> broad.
        synonym_entries: list[tuple[str, TaxonInfo]] = []
        curated_term_entries: list[tuple[str, TaxonInfo, bool]] = []
        ncbi_curated_entries: list[tuple[str, TaxonInfo]] = []
        broad_entries: list[tuple[str, TaxonInfo]] = []

        for idx, row in ncbi_df.iterrows():
            info = TaxonInfo(
                taxid=int(row["taxid"]),
                scientific_name=str(row["scientific_name"]),
                rank=str(row.get("rank", "")).strip().lower(),
                table_priority=idx,
            )
            self.taxid_to_info[str(info.taxid)] = info

            sciname = info.scientific_name.strip()
            if sciname and sciname.lower() != "nan":
                norm_sciname = _normalize_text(sciname)
                self.sciname_to_info.setdefault(norm_sciname, info)
                self._index_for_subset(norm_sciname, info)

            for term in self._split_cell(row.get("synonym")):
                synonym_entries.append((_normalize_text(term), info))
            for term in self._split_cell(row.get("genbank_common_name")):
                ncbi_curated_entries.append((_normalize_text(term), info))
            for term in self._split_cell(row.get("common_name")):
                broad_entries.append((_normalize_text(term), info))

        ncbi_exact_terms = dict(self.sciname_to_info)
        for normalized_term, exact_info in synonym_entries:
            ncbi_exact_terms.setdefault(normalized_term, exact_info)

        for taxon_policy in self.policy.curated_taxa:
            info = self.taxid_to_info.get(str(taxon_policy.taxid))
            if info is None:
                raise ValueError(
                    f"Curated taxid {taxon_policy.taxid} is not present in the NCBI "
                    "taxonomy reference."
                )
            if taxon_policy.scientific_name != info.scientific_name:
                raise ValueError(
                    f"Curated taxid {taxon_policy.taxid} scientific_name must match NCBI "
                    f"{info.scientific_name!r}; got {taxon_policy.scientific_name!r}."
                )
            terms_by_mode = (
                ("exact", taxon_policy.exact_terms),
                ("subset", taxon_policy.subset_terms),
                ("force", taxon_policy.force_terms),
            )
            for mode, terms in terms_by_mode:
                for term in terms:
                    normalized_term = _normalize_text(str(term))
                    ncbi_exact = ncbi_exact_terms.get(normalized_term)
                    if (
                        mode != "force"
                        and ncbi_exact is not None
                        and ncbi_exact.taxid != info.taxid
                    ):
                        raise ValueError(
                            f"Curated term {term!r} conflicts with an NCBI exact term for "
                            f"taxid {ncbi_exact.taxid}; place it under force to map it to "
                            f"taxid {taxon_policy.taxid}."
                        )
                    if mode != "force":
                        curated_term_entries.append((normalized_term, info, mode == "subset"))

        for norm, info in synonym_entries:
            if not norm:
                continue
            self.synonym_to_info.setdefault(norm, info)
            self._index_for_subset(norm, info)

        for norm, info, allow_subset in curated_term_entries:
            if not norm:
                continue
            self.curated_term_to_info.setdefault(norm, info)
            if allow_subset:
                self._index_for_subset(norm, info, include_subspecies=True)

        for norm, info in ncbi_curated_entries:
            if not norm:
                continue
            self.curated_common_to_info.setdefault(norm, info)
            self._index_for_subset(norm, info)

        for norm, info in broad_entries:
            if not norm:
                continue
            self.broad_common_to_info.setdefault(norm, info)
            self._index_for_subset(norm, info)

        self.logger.info(
            "Loaded lookup tables: %d taxids, %d unique scinames, "
            "%d unique synonyms, %d unique curated terms, "
            "%d unique NCBI curated common names, "
            "%d unique NCBI broad common names, "
            "%d multi-word subset terms, %d single-word subset terms",
            len(self.taxid_to_info),
            len(self.sciname_to_info),
            len(self.synonym_to_info),
            len(self.curated_term_to_info),
            len(self.curated_common_to_info),
            len(self.broad_common_to_info),
            len(self.multiword_term_to_info),
            len(self.singleword_term_to_info),
        )

    @staticmethod
    def _split_cell(cell) -> list[str]:
        """Split a semicolon-separated TSV cell into clean terms; tolerates NaN/missing/empty."""
        if cell is None or pd.isna(cell):
            return []
        parts = str(cell).split(";")
        return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]

    def _index_for_subset(
        self,
        norm_term: str,
        info: TaxonInfo,
        *,
        include_subspecies: bool = False,
    ) -> None:
        """
        Add a normalized term to the subset-matching index.

        Subspecies are excluded - their trinomial names produce false
        positives when bag-of-words matching ignores order (e.g. 'Gallus
        gallus gallus' matching the input 'Gallus gallus'). They remain
        in the exact-match lookups, so a value typed as the full
        trinomial still resolves correctly.
        """
        if not norm_term or (info.rank == "subspecies" and not include_subspecies):
            return
        words = norm_term.split()
        if len(words) >= 2:
            self.multiword_term_to_info.setdefault(norm_term, info)
            for w in words:
                self.multiword_inverted_index.setdefault(w, set()).add(norm_term)
        else:
            self.singleword_term_to_info.setdefault(norm_term, info)

    def _compile_filters(self) -> None:
        self.ignored_patterns: list[re.Pattern] = [
            re.compile(re.escape(value), re.IGNORECASE) for value in self.policy.ignored_substrings
        ]
        # Each iso_keyword is stored alongside its normalized word set.
        # Matching is whole-word: all the keyword's words must appear
        # as whole words in the value's normalized word set.
        self.iso_keywords: list[tuple[str, frozenset[str]]] = []
        for kw in self.policy.isolation_source_keywords:
            words = frozenset(_normalize_text(str(kw)).split())
            if words:
                self.iso_keywords.append((str(kw), words))

        # Preemptive decisions keyed by normalized value: int = force taxid,
        # None = reject as host and preserve as overflow for iso_source.
        self.preemptive_decisions: dict[str, int | None] = {}
        for taxon_policy in self.policy.curated_taxa:
            for raw_value in taxon_policy.force_terms:
                self.preemptive_decisions[_normalize_text(raw_value)] = taxon_policy.taxid
        for raw_value in self.policy.value_rejections:
            self.preemptive_decisions[_normalize_text(raw_value)] = None
        if self.preemptive_decisions:
            self.logger.info(
                "Loaded %d preemptive host decision(s)", len(self.preemptive_decisions)
            )

    # --- Per-value matching ---

    def _strip_ignored_substrings(self, value: str) -> str:
        for pattern in self.ignored_patterns:
            value = pattern.sub("", value)
        return value

    def _match_numeric_value(self, normalized: str) -> ValueMatch | None:
        info = self.taxid_to_info.get(normalized)
        if info is None:
            return None
        return ValueMatch(info, SCORE_TAXID)

    def _match_text_value(self, normalized: str) -> ValueMatch | None:
        # Tiers tried in priority order. Sciname and synonym both score
        # 1.0, but sciname is checked first to keep the logged tier
        # label honest.
        for lookup, score in (
            (self.sciname_to_info, SCORE_SCINAME),
            (self.synonym_to_info, SCORE_SYNONYM),
            (self.curated_term_to_info, SCORE_CURATED_TERM),
            (self.curated_common_to_info, SCORE_CURATED_COMMON),
            (self.broad_common_to_info, SCORE_BROAD_COMMON),
        ):
            info = lookup.get(normalized)
            if info is not None:
                return ValueMatch(info, score)
        return self._match_subset_value(normalized)

    def _match_subset_value(self, normalized: str) -> ValueMatch | None:
        """Whole-word containment matching: multi-word terms first, single-word as fallback."""
        input_words = set(normalized.split())
        # Numeric-stripped variants so e.g. "patient1" matches as "patient".
        search_words = set(input_words)
        for w in input_words:
            stripped = w.strip(string.digits)
            if stripped:
                search_words.add(stripped)

        # --- Multi-word terms ---
        # Use raw input word count, not the deduped set, so a 2-word
        # input like "Gallus gallus" cannot match a 3-word term like
        # "Gallus gallus gallus" even though their distinct words match.
        input_word_count = len(normalized.split())
        candidate_terms: set[str] = set()
        for w in search_words:
            candidate_terms.update(self.multiword_inverted_index.get(w, set()))

        multiword_matches: list[TaxonInfo] = []
        for term in candidate_terms:
            term_words = term.split()
            if len(term_words) > input_word_count:
                continue
            if not set(term_words).issubset(search_words):
                continue
            info = self.multiword_term_to_info.get(term)
            if info is not None:
                multiword_matches.append(info)

        if multiword_matches:
            return self._build_subset_match(multiword_matches, SCORE_SUBSET_MULTIWORD, "multi-word")

        # --- Single-word terms (fallback) ---
        singleword_matches: list[TaxonInfo] = []
        for w in search_words:
            info = self.singleword_term_to_info.get(w)
            if info is not None:
                singleword_matches.append(info)

        if singleword_matches:
            return self._build_subset_match(
                singleword_matches, SCORE_SUBSET_SINGLEWORD, "single-word"
            )

        return None

    @staticmethod
    def _build_subset_match(matches: list[TaxonInfo], score: float, tier_label: str) -> ValueMatch:
        # Dedupe by taxid (a taxon can be reached via several names)
        # then pick the most specific, recording other distinct taxa for
        # the caller to warn about.
        infos_by_taxid = {i.taxid: i for i in matches}
        best = min(infos_by_taxid.values(), key=lambda i: i.table_priority)
        if len(infos_by_taxid) > 1:
            all_names = tuple(sorted(i.scientific_name for i in infos_by_taxid.values()))
            return ValueMatch(best, score, match_tier=tier_label, tier_candidates=all_names)
        return ValueMatch(best, score, match_tier=tier_label)

    def _match_value(self, value: str, attribute: str) -> ValueMatch | None:
        """Dispatch a single (attribute, value) pair to the right matcher."""
        normalized = _normalize_text(self._strip_ignored_substrings(value.strip()))
        if not normalized:
            return None
        if normalized.isdigit():
            if attribute.lower() != "host_taxid":
                return None
            return self._match_numeric_value(normalized)
        return self._match_text_value(normalized)

    def _find_iso_keyword(self, val_str: str) -> str | None:
        """
        Return the first iso_keyword whose words all appear in any value, else None.

        Whole-word match: keyword 'food' matches 'duck food' but not
        'seafood', because normalization splits on whitespace and
        'seafood' is a single word.
        """
        if not self.iso_keywords:
            return None
        for value in split_pipe_separated(val_str):
            value_words = set(_normalize_text(value).split())
            if not value_words:
                continue
            for original, kw_words in self.iso_keywords:
                if kw_words.issubset(value_words):
                    return original
        return None

    def _check_preemptive_decisions(self, val_str: str) -> tuple[str, str | int | None]:
        """
        Check whether any value in the row hits a forced match or value rejection.

        Returns one of:
          ("none",   None)    no preemptive decision; proceed with normal matching
          ("reject", raw_val) any value mapped to null; row goes to iso_source
          ("force",  taxid)   any value mapped to a taxid; force that match

        """
        if not self.preemptive_decisions:
            return "none", None
        forced: tuple[str, int] | None = None
        for value in split_pipe_separated(val_str):
            norm = _normalize_text(value)
            if norm not in self.preemptive_decisions:
                continue
            target = self.preemptive_decisions[norm]
            if target is None:
                return "reject", value.strip()
            if forced is None:
                forced = (value.strip(), target)
        if forced is not None:
            return "force", forced[1]
        return "none", None

    def _build_forced_match(self, val_str: str, attributes_str: str, taxid: int) -> HostMatch:
        """Build a HostMatch from a forced taxid, picking the first value that triggered it."""
        attributes = split_pipe_separated(attributes_str)
        values = split_pipe_separated(val_str)
        info = self.taxid_to_info[str(taxid)]
        for idx, (raw_attr, raw_val) in enumerate(zip(attributes, values, strict=False)):
            if self.preemptive_decisions.get(_normalize_text(raw_val)) == taxid:
                return HostMatch(
                    info=info,
                    score=1.0,
                    source_index=idx,
                    attribute=raw_attr.strip(),
                    value=raw_val.strip(),
                    match_tier="",
                    tier_candidates=(),
                    low_confidence=False,
                )
        raise AssertionError(f"_build_forced_match: no value matched taxid {taxid}")

    # --- Per-record dispatch ---

    def classify_row(
        self,
        accession: str,
        attr_str: str,
        val_str: str,
        skip_iso_keywords: bool = False,
    ) -> HostMatch | None:
        """Run the full per-row cascade: iso keyword -> preemptive decision -> match.

        Returns the winning HostMatch when a host is identified (including
        forced terms). Returns None when the row should be forwarded to
        the iso pipeline (iso keyword hit, value rejection, or no match).

        `skip_iso_keywords` bypasses the iso_keyword guard. Used when
        the value has already been classified by the iso pipeline, to find
        the source organism named in it (e.g.'chicken meat' -> Gallus gallus).
        """
        match, _diagnostic = self._classify_row_with_diagnostic(
            accession,
            attr_str,
            val_str,
            skip_iso_keywords=skip_iso_keywords,
        )
        return match

    def _classify_row_with_diagnostic(
        self,
        accession: str,
        attr_str: str,
        val_str: str,
        *,
        skip_iso_keywords: bool = False,
    ) -> tuple[HostMatch | None, HostDiagnostic]:
        attribute_count = len(split_pipe_separated(attr_str))
        value_count = len(split_pipe_separated(val_str))
        if attribute_count != value_count:
            raise ValueError(
                f"Malformed host candidates for {accession}: "
                f"host_attr_orig={attribute_count}, host_val_orig={value_count}; "
                "counts must match"
            )
        if not skip_iso_keywords:
            iso_keyword = self._find_iso_keyword(val_str)
            if iso_keyword is not None:
                return None, HostDiagnostic.ISO_KEYWORD_PREEMPTION

        outcome, payload = self._check_preemptive_decisions(val_str)
        if outcome == "reject":
            return None, HostDiagnostic.OVERRIDE_REJECTION
        if outcome == "force":
            match = self._build_forced_match(val_str, attr_str, payload)
            return match, HostDiagnostic.FORCED_OVERRIDE

        match = self.find_best_match(accession, attr_str, val_str)
        diagnostic = HostDiagnostic.MATCHED if match is not None else HostDiagnostic.UNMATCHED
        return match, diagnostic

    def standardize(self, record: Mapping[str, str]) -> HostOutcome:
        accession = record.get("accession", "")
        attributes = record.get("host_attr_orig", "") or ""
        values = record.get("host_val_orig", "") or ""
        match, diagnostic = self._classify_row_with_diagnostic(
            accession,
            attributes,
            values,
        )
        if match is not None:
            return self._matched_outcome(match, diagnostic)
        overflow = None
        if values.strip():
            overflow = HostOverflowContext(
                attribute=attributes,
                value=values,
            )
        return HostOutcome(
            standardized=None,
            score=None,
            low_confidence=False,
            origins=(),
            overflow=overflow,
            diagnostics=(diagnostic,),
        )

    def retry(self, accession: str, attributes: str, values: str) -> HostOutcome:
        """Retry host classification after eligible isolation interpretation."""
        match, diagnostic = self._classify_row_with_diagnostic(
            accession,
            attributes,
            values,
            skip_iso_keywords=True,
        )
        if match is None:
            return HostOutcome(
                standardized=None,
                score=None,
                low_confidence=False,
                origins=(),
                overflow=None,
                diagnostics=(diagnostic,),
                retry_eligible=True,
            )
        return self._matched_outcome(match, diagnostic, retry_eligible=True)

    @staticmethod
    def _matched_outcome(
        match: HostMatch,
        diagnostic: HostDiagnostic,
        *,
        retry_eligible: bool = False,
    ) -> HostOutcome:
        return HostOutcome(
            standardized=StandardizedHost(
                taxid=match.info.taxid,
                scientific_name=match.info.scientific_name,
            ),
            score=match.score,
            low_confidence=match.low_confidence,
            origins=(HostOrigin(match.attribute, match.value),),
            overflow=None,
            diagnostics=(diagnostic, *match.diagnostics),
            retry_eligible=retry_eligible,
        )

    def find_best_match(
        self,
        accession: str,
        attributes_str: str,
        values_str: str,
    ) -> HostMatch | None:
        attributes = split_pipe_separated(attributes_str)
        values = split_pipe_separated(values_str)

        candidates: list[HostMatch] = []
        for idx, (raw_attr, raw_val) in enumerate(zip(attributes, values, strict=False)):
            attr = raw_attr.strip()
            val = raw_val.strip()
            match = self._match_value(val, attr)
            if match is None:
                continue
            candidates.append(
                HostMatch(
                    info=match.info,
                    score=match.score,
                    source_index=idx,
                    attribute=attr,
                    value=val,
                    match_tier=match.match_tier,
                    tier_candidates=match.tier_candidates,
                )
            )

        if not candidates:
            return None

        # Sort key (smaller wins):
        #   1. score              higher score wins
        #   2. table_priority     more specific taxon wins
        #   3. attr priority      host_taxid > host > other
        #   4. source_index       earlier position as last-resort tiebreaker
        candidates.sort(
            key=lambda c: (
                -c.score,
                c.info.table_priority,
                _attr_priority(c.attribute),
                c.source_index,
            )
        )
        best = candidates[0]

        distinct_taxa = {c.info.taxid for c in candidates}
        has_multiple_taxa = len(distinct_taxa) > 1
        has_ambiguous_subset = bool(best.tier_candidates)
        is_subset_match = best.match_tier != ""
        low_confidence = is_subset_match or has_ambiguous_subset or has_multiple_taxa
        diagnostics = tuple(
            diagnostic
            for applies, diagnostic in (
                (is_subset_match, HostDiagnostic.SUBSET_MATCH),
                (has_ambiguous_subset, HostDiagnostic.AMBIGUOUS_SUBSET),
                (has_multiple_taxa, HostDiagnostic.ATTRIBUTE_DISAGREEMENT),
            )
            if applies
        )

        return HostMatch(
            info=best.info,
            score=best.score,
            source_index=best.source_index,
            attribute=best.attribute,
            value=best.value,
            match_tier=best.match_tier,
            tier_candidates=best.tier_candidates,
            low_confidence=low_confidence,
            diagnostics=diagnostics,
        )
