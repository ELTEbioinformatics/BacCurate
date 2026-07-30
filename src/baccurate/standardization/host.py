"""
Map host annotations from sample metadata to NCBI Taxonomy IDs and
scientific names (binomial nomenclature).

See docs/host.md for the documentation.
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from baccurate.adapters.policy_yaml import PolicyConfigurationError, load_policy_mapping
from baccurate.pathogen_registry.registry import PathogenRegistry
from baccurate.paths import DEFAULT_TAXIDS_NCBI
from baccurate.standardization._attribute_value_text import split_pipe_separated
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair

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

# Tiebreaker between host matches of equal score and equal taxonomic
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
_PREFIXED_TAXID_PATTERN = r"NCBITaxon:(\d+)"
_PREFIXED_TAXID_RE = re.compile(_PREFIXED_TAXID_PATTERN, re.IGNORECASE)
_LABELED_TAXID_PATTERNS = (
    re.compile(rf"(.+?)\s*\[\s*{_PREFIXED_TAXID_PATTERN}\s*\]", re.IGNORECASE),
    re.compile(rf"(.+?)\s*\(\s*{_PREFIXED_TAXID_PATTERN}\s*\)", re.IGNORECASE),
    re.compile(rf"(.+?)\s+{_PREFIXED_TAXID_PATTERN}", re.IGNORECASE),
)
_LEADING_TAXID_RE = re.compile(rf"{_PREFIXED_TAXID_PATTERN}\s+(.+)", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().replace("_", " ").replace("-", " ")
    text = text.translate(_PUNCT_TABLE)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _parse_prefixed_taxid(value: str) -> tuple[str, str] | None:
    """Return a whole-value label and taxid from an accepted prefixed form."""
    stripped = value.strip()
    identifier_only = _PREFIXED_TAXID_RE.fullmatch(stripped)
    if identifier_only is not None:
        return "", identifier_only.group(1)

    for pattern in _LABELED_TAXID_PATTERNS:
        labeled_identifier = pattern.fullmatch(stripped)
        if labeled_identifier is not None:
            label, taxid = labeled_identifier.groups()
            if _PREFIXED_TAXID_RE.search(label) is None:
                return label.strip(), taxid

    leading_identifier = _LEADING_TAXID_RE.fullmatch(stripped)
    if leading_identifier is not None:
        taxid, label = leading_identifier.groups()
        if _PREFIXED_TAXID_RE.search(label) is None:
            return label.strip(), taxid
    return None


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

    schema_version: int
    ignored_substrings: tuple[str, ...]
    isolation_source_keywords: tuple[str, ...]
    curated_taxa: tuple[CuratedTaxonPolicy, ...]
    value_rejection_entries: tuple[str | TargetPathogenHostRejection, ...]
    value_rejections: tuple[str, ...]

    @classmethod
    def load(cls, path: Path | str, pathogen_registry: PathogenRegistry) -> HostPolicy:
        """Strictly load host policy and resolve target-pathogen references."""
        source = Path(path)
        try:
            return _parse_host_policy(
                load_policy_mapping(source),
                source,
                pathogen_registry,
            )
        except PolicyConfigurationError:
            raise
        except ValueError as error:
            raise PolicyConfigurationError(f"{source}: {error}") from error

    def serialize(self) -> str:
        """Return deterministic canonical JSON while preserving rejection order."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "normalization": {
                    "ignored_substrings": list(self.ignored_substrings),
                },
                "routing": {
                    "isolation_source_keywords": list(self.isolation_source_keywords),
                },
                "curated_taxa": [
                    {
                        "taxid": taxon.taxid,
                        "scientific_name": taxon.scientific_name,
                        "match_terms": {
                            "exact": list(taxon.exact_terms),
                            "subset": list(taxon.subset_terms),
                            "force": list(taxon.force_terms),
                        },
                    }
                    for taxon in self.curated_taxa
                ],
                "value_rejections": {
                    "exact": [
                        (
                            {"pathogen_key": entry.pathogen_key}
                            if isinstance(entry, TargetPathogenHostRejection)
                            else entry
                        )
                        for entry in self.value_rejection_entries
                    ],
                },
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def legacy_configuration(self) -> dict[str, object]:
        """Reconstruct the pre-v3 effective mapping used by existing fingerprints."""
        return {
            "schema_version": 2,
            "normalization": {"ignored_substrings": list(self.ignored_substrings)},
            "routing": {
                "isolation_source_keywords": list(self.isolation_source_keywords),
            },
            "curated_taxa": {
                str(taxon.taxid): {
                    "scientific_name": taxon.scientific_name,
                    "match_terms": {
                        mode: list(terms)
                        for mode, terms in (
                            ("exact", taxon.exact_terms),
                            ("subset", taxon.subset_terms),
                            ("force", taxon.force_terms),
                        )
                        if terms
                    },
                }
                for taxon in self.curated_taxa
            },
            "value_rejections": {"exact": list(self.value_rejections)},
        }


@dataclass(frozen=True, slots=True)
class TargetPathogenHostRejection:
    """A host rejection whose scientific name is owned by the pathogen registry."""

    pathogen_key: str


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


def _reject_unknown_keys(
    config: Mapping,
    allowed: set[str],
    config_path: Path,
    label: str,
) -> None:
    unknown = set(config) - allowed
    if unknown:
        name = sorted(str(key) for key in unknown)[0]
        raise PolicyConfigurationError(f"{config_path}: {label}.{name}: unknown policy key")


def _parse_host_policy(
    config: Mapping,
    config_path: Path,
    pathogen_registry: PathogenRegistry,
) -> HostPolicy:
    if config.get("schema_version") != 3:
        raise PolicyConfigurationError(f"{config_path}: schema_version: must be 3")
    _reject_unknown_keys(
        config,
        {"schema_version", "normalization", "routing", "curated_taxa", "value_rejections"},
        config_path,
        "top level",
    )

    normalization = _require_mapping(config.get("normalization"), "normalization")
    _reject_unknown_keys(
        normalization,
        {"ignored_substrings"},
        config_path,
        "normalization",
    )
    ignored_substrings = _require_string_list(
        normalization.get("ignored_substrings"), "normalization.ignored_substrings"
    )

    routing = _require_mapping(config.get("routing"), "routing")
    _reject_unknown_keys(
        routing,
        {"isolation_source_keywords"},
        config_path,
        "routing",
    )
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
        _reject_unknown_keys(
            taxon,
            {"scientific_name", "match_terms"},
            config_path,
            f"curated_taxa.{taxid}",
        )
        scientific_name = taxon.get("scientific_name")
        if not isinstance(scientific_name, str) or not scientific_name.strip():
            raise ValueError(f"curated_taxa.{taxid}.scientific_name must be a non-empty string")
        match_terms = _require_mapping(
            taxon.get("match_terms"), f"curated_taxa.{taxid}.match_terms"
        )
        _reject_unknown_keys(
            match_terms,
            {"exact", "subset", "force"},
            config_path,
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
    _reject_unknown_keys(
        value_rejections,
        {"exact"},
        config_path,
        "value_rejections",
    )
    raw_rejections = value_rejections.get("exact")
    if not isinstance(raw_rejections, list):
        raise ValueError("value_rejections.exact must be a list")
    rejection_entries: list[str | TargetPathogenHostRejection] = []
    rejection_terms: list[str] = []
    for index, raw_rejection in enumerate(raw_rejections):
        policy_key = f"value_rejections.exact.{index}"
        if isinstance(raw_rejection, str):
            if not _normalize_text(raw_rejection):
                raise ValueError(f"{policy_key} contains an empty normalized value")
            rejection_entries.append(raw_rejection)
            rejection_terms.append(raw_rejection)
            continue
        rejection = _require_mapping(raw_rejection, policy_key)
        _reject_unknown_keys(rejection, {"pathogen_key"}, config_path, policy_key)
        pathogen_key = rejection.get("pathogen_key")
        if not isinstance(pathogen_key, str) or not pathogen_key.strip():
            raise ValueError(f"{policy_key}.pathogen_key must be a non-empty string")
        scientific_name = pathogen_registry.scientific_name(pathogen_key)
        if not scientific_name:
            raise PolicyConfigurationError(
                f"{config_path}: {policy_key}.pathogen_key: unknown pathogen key {pathogen_key!r}"
            )
        rejection_entries.append(TargetPathogenHostRejection(pathogen_key))
        rejection_terms.append(scientific_name)
    overlap = set(curated_term_taxids) & {_normalize_text(term) for term in rejection_terms}
    if overlap:
        raise PolicyConfigurationError(
            f"{config_path}: value_rejections.exact: "
            f"{sorted(overlap)[0]!r} is both a curated term and a value rejection"
        )
    return HostPolicy(
        schema_version=3,
        ignored_substrings=tuple(ignored_substrings),
        isolation_source_keywords=tuple(isolation_source_keywords),
        curated_taxa=tuple(policies),
        value_rejection_entries=tuple(rejection_entries),
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
    """Taxon selected for one submitted host value before record-level ranking."""

    info: TaxonInfo
    match_quality_score: float
    # "" for exact matches; "multi-word" or "single-word" for subset matches.
    match_tier: str = ""
    # Populated when subset matching found multiple distinct taxa. Empty
    # for unambiguous matches.
    tier_taxon_names: tuple[str, ...] = ()
    # True when a paired label resolves to a different taxon than its identifier.
    identifier_disagreement: bool = False


class HostDiagnostic(StrEnum):
    """The fixed set of host-classification results used in build statistics."""

    ISOLATION_SOURCE_KEYWORD_PREEMPTION = "isolation_source_keyword_preemption"
    OVERRIDE_REJECTION = "override_rejection"
    FORCED_OVERRIDE = "forced_override"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    SUBSET_MATCH = "subset_match"
    AMBIGUOUS_SUBSET = "ambiguous_subset"
    ATTRIBUTE_DISAGREEMENT = "attribute_disagreement"


@dataclass(frozen=True, slots=True)
class HostMatch:
    """Winning match with its supporting attribute-value pair."""

    info: TaxonInfo
    match_quality_score: float
    pair_index: int
    attribute: str
    value: str
    match_tier: str
    tier_taxon_names: tuple[str, ...]
    # True when worth reviewing:
    # any subset match, ambiguous subset, identifier disagreement, or
    # cross-attribute disagreement.
    needs_review: bool = False
    diagnostics: tuple[HostDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class StandardizedHost:
    """Taxonomic identity selected for one extracted metadata record."""

    taxid: int
    scientific_name: str


@dataclass(frozen=True, slots=True)
class HostOverflowContext:
    """Answer which unresolved pair is routed onward, unlike SupportingAttributeValuePair."""

    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class HostOutcome:
    """Record-level host result, including distinct absence and overflow states."""

    standardized: StandardizedHost | None
    match_quality_score: float | None
    needs_review: bool
    supporting_pairs: tuple[SupportingAttributeValuePair, ...]
    overflow: HostOverflowContext | None
    diagnostics: tuple[HostDiagnostic, ...]
    from_recovery_pass: bool = False

    def __post_init__(self) -> None:
        if (self.standardized is None) != (self.match_quality_score is None):
            raise ValueError("A standardized host and score must be present together")
        if self.standardized is not None and not self.supporting_pairs:
            raise ValueError("A standardized host requires a supporting attribute-value pair")
        if self.standardized is None and self.supporting_pairs:
            raise ValueError(
                "An absent standardized host cannot have supporting attribute-value pairs"
            )
        if not self.diagnostics:
            raise ValueError("A host outcome requires at least one diagnostic")


# --- Main class ---


class HostStandardizer:
    def __init__(
        self,
        policy: HostPolicy,
        ncbi_table_path: Path | str = DEFAULT_TAXIDS_NCBI,
        result_logger: logging.Logger | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.policy = policy
        self.config = policy.legacy_configuration()
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
        # gets a different match-quality score in the matching cascade.
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

        optional_columns = {
            "rank": "",
            "synonym": None,
            "genbank_common_name": None,
            "common_name": None,
        }
        for column, default in optional_columns.items():
            if column not in ncbi_df.columns:
                ncbi_df[column] = default
        lookup_columns = (
            "taxid",
            "scientific_name",
            "rank",
            "synonym",
            "genbank_common_name",
            "common_name",
        )
        reference_rows = ncbi_df.loc[:, lookup_columns].itertuples(index=False, name=None)
        for idx, (
            taxid,
            scientific_name,
            rank,
            synonym,
            genbank_common_name,
            common_name,
        ) in enumerate(reference_rows):
            info = TaxonInfo(
                taxid=int(taxid),
                scientific_name=str(scientific_name),
                rank=str(rank).strip().lower(),
                table_priority=idx,
            )
            self.taxid_to_info[str(info.taxid)] = info

            sciname = info.scientific_name.strip()
            if sciname and sciname.lower() != "nan":
                norm_sciname = _normalize_text(sciname)
                self.sciname_to_info.setdefault(norm_sciname, info)
                self._index_for_subset(norm_sciname, info)

            for term in self._split_cell(synonym):
                synonym_entries.append((_normalize_text(term), info))
            for term in self._split_cell(genbank_common_name):
                ncbi_curated_entries.append((_normalize_text(term), info))
            for term in self._split_cell(common_name):
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
        # Each isolation_source_keyword is stored alongside its normalized word set.
        # Matching is whole-word: all the keyword's words must appear
        # as whole words in the value's normalized word set.
        self.isolation_source_keywords: list[tuple[str, frozenset[str]]] = []
        for kw in self.policy.isolation_source_keywords:
            words = frozenset(_normalize_text(str(kw)).split())
            if words:
                self.isolation_source_keywords.append((str(kw), words))

        # Preemptive decisions keyed by normalized value: int = force taxid,
        # None = reject as host and preserve as overflow for isolation-source standardization.
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
        for lookup, match_quality_score in (
            (self.sciname_to_info, SCORE_SCINAME),
            (self.synonym_to_info, SCORE_SYNONYM),
            (self.curated_term_to_info, SCORE_CURATED_TERM),
            (self.curated_common_to_info, SCORE_CURATED_COMMON),
            (self.broad_common_to_info, SCORE_BROAD_COMMON),
        ):
            info = lookup.get(normalized)
            if info is not None:
                return ValueMatch(info, match_quality_score)
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
        indexed_terms: set[str] = set()
        for w in search_words:
            indexed_terms.update(self.multiword_inverted_index.get(w, set()))

        multiword_matches: list[TaxonInfo] = []
        for term in indexed_terms:
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
    def _build_subset_match(
        matches: list[TaxonInfo], match_quality_score: float, tier_label: str
    ) -> ValueMatch:
        # Dedupe by taxid (a taxon can be reached via several names)
        # then pick the most specific, recording other distinct taxa for
        # the caller to warn about.
        infos_by_taxid = {i.taxid: i for i in matches}
        best = min(infos_by_taxid.values(), key=lambda i: i.table_priority)
        if len(infos_by_taxid) > 1:
            all_names = tuple(sorted(i.scientific_name for i in infos_by_taxid.values()))
            return ValueMatch(
                best,
                match_quality_score,
                match_tier=tier_label,
                tier_taxon_names=all_names,
            )
        return ValueMatch(best, match_quality_score, match_tier=tier_label)

    def _match_value(self, value: str, attribute: str) -> ValueMatch | None:
        """Dispatch a single (attribute, value) pair to the right matcher."""
        prefixed_taxid = _parse_prefixed_taxid(value)
        if prefixed_taxid is not None:
            label, taxid = prefixed_taxid
            identifier_match = self._match_numeric_value(taxid)
            if identifier_match is None or not label:
                return identifier_match
            label_match = self._match_text_value(_normalize_text(label))
            return ValueMatch(
                identifier_match.info,
                identifier_match.match_quality_score,
                identifier_disagreement=(
                    label_match is not None
                    and label_match.info.taxid != identifier_match.info.taxid
                ),
            )

        normalized = _normalize_text(self._strip_ignored_substrings(value.strip()))
        if not normalized:
            return None
        if normalized.isdigit():
            if attribute.lower() != "host_taxid":
                return self._match_text_value(normalized)
            return self._match_numeric_value(normalized)
        return self._match_text_value(normalized)

    def _find_isolation_source_keyword(self, val_str: str) -> str | None:
        """
        Return the first isolation-source keyword whose words all appear in any value, else None.

        Whole-word match: keyword 'food' matches 'duck food' but not
        'seafood', because normalization splits on whitespace and
        'seafood' is a single word.
        """
        if not self.isolation_source_keywords:
            return None
        for value in split_pipe_separated(val_str):
            value_words = set(_normalize_text(value).split())
            if not value_words:
                continue
            for original, kw_words in self.isolation_source_keywords:
                if kw_words.issubset(value_words):
                    return original
        return None

    def _check_preemptive_decisions(self, val_str: str) -> tuple[str, str | int | None]:
        """
        Check whether any value in the row hits a forced match or value rejection.

        Returns one of:
          ("none",   None)    no preemptive decision; proceed with normal matching
          ("reject", raw_val) any value mapped to null; row goes to isolation-source standardization
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
                    match_quality_score=1.0,
                    pair_index=idx,
                    attribute=raw_attr.strip(),
                    value=raw_val.strip(),
                    match_tier="",
                    tier_taxon_names=(),
                    needs_review=False,
                )
        raise AssertionError(f"_build_forced_match: no value matched taxid {taxid}")

    # --- Per-record dispatch ---

    def classify_extracted_record(
        self,
        accession: str,
        attr_str: str,
        val_str: str,
        skip_isolation_source_keywords: bool = False,
    ) -> HostMatch | None:
        """Run the full cascade for one extracted metadata record.

        Return the winning HostMatch when a host is identified, including a forced
        match. Return None when an isolation-source keyword or value rejection sends
        the extracted metadata record to isolation-source standardization, or when no
        host match is found.

        `skip_isolation_source_keywords` bypasses the isolation-source keyword guard.
        Set it after the value has been classified for the isolation-source target so
        host standardization can find the source organism named in text such as
        'chicken meat' -> Gallus gallus.
        """
        match, _diagnostic = self._classify_extracted_record_with_diagnostic(
            accession,
            attr_str,
            val_str,
            skip_isolation_source_keywords=skip_isolation_source_keywords,
        )
        return match

    def _classify_extracted_record_with_diagnostic(
        self,
        accession: str,
        attr_str: str,
        val_str: str,
        *,
        skip_isolation_source_keywords: bool = False,
    ) -> tuple[HostMatch | None, HostDiagnostic]:
        attribute_count = len(split_pipe_separated(attr_str))
        value_count = len(split_pipe_separated(val_str))
        if attribute_count != value_count:
            raise ValueError(
                f"Malformed host attribute-value pairs for {accession}: "
                f"host_attr_orig={attribute_count}, host_val_orig={value_count}; "
                "counts must match"
            )
        if not skip_isolation_source_keywords:
            isolation_source_keyword = self._find_isolation_source_keyword(val_str)
            if isolation_source_keyword is not None:
                return None, HostDiagnostic.ISOLATION_SOURCE_KEYWORD_PREEMPTION

        outcome, payload = self._check_preemptive_decisions(val_str)
        if outcome == "reject":
            return None, HostDiagnostic.OVERRIDE_REJECTION
        if outcome == "force":
            match = self._build_forced_match(val_str, attr_str, payload)
            return match, HostDiagnostic.FORCED_OVERRIDE

        match = self.find_best_match(accession, attr_str, val_str)
        diagnostic = HostDiagnostic.MATCHED if match is not None else HostDiagnostic.UNMATCHED
        return match, diagnostic

    def standardize(self, extracted_record: Mapping[str, str]) -> HostOutcome:
        """Standardize the host in one extracted metadata record."""
        accession = extracted_record.get("accession", "")
        attributes = extracted_record.get("host_attr_orig", "") or ""
        values = extracted_record.get("host_val_orig", "") or ""
        match, diagnostic = self._classify_extracted_record_with_diagnostic(
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
            match_quality_score=None,
            needs_review=False,
            supporting_pairs=(),
            overflow=overflow,
            diagnostics=(diagnostic,),
        )

    def recovery_pass(self, accession: str, attributes: str, values: str) -> HostOutcome:
        """Match a host in isolation-source values the ontology judged host-implying.

        The isolation-source keyword guard is skipped: it exists to keep
        isolation-source wording out of host matching, and here that wording is the input.
        """
        match, diagnostic = self._classify_extracted_record_with_diagnostic(
            accession,
            attributes,
            values,
            skip_isolation_source_keywords=True,
        )
        if match is None:
            return HostOutcome(
                standardized=None,
                match_quality_score=None,
                needs_review=False,
                supporting_pairs=(),
                overflow=None,
                diagnostics=(diagnostic,),
                from_recovery_pass=True,
            )
        return self._matched_outcome(match, diagnostic, from_recovery_pass=True)

    @staticmethod
    def _matched_outcome(
        match: HostMatch,
        diagnostic: HostDiagnostic,
        *,
        from_recovery_pass: bool = False,
    ) -> HostOutcome:
        return HostOutcome(
            standardized=StandardizedHost(
                taxid=match.info.taxid,
                scientific_name=match.info.scientific_name,
            ),
            match_quality_score=match.match_quality_score,
            needs_review=match.needs_review,
            supporting_pairs=(SupportingAttributeValuePair(match.attribute, match.value),),
            overflow=None,
            diagnostics=(diagnostic, *match.diagnostics),
            from_recovery_pass=from_recovery_pass,
        )

    def find_best_match(
        self,
        accession: str,
        attributes_str: str,
        values_str: str,
    ) -> HostMatch | None:
        attributes = split_pipe_separated(attributes_str)
        values = split_pipe_separated(values_str)

        host_matches: list[HostMatch] = []
        for idx, (raw_attr, raw_val) in enumerate(zip(attributes, values, strict=False)):
            attr = raw_attr.strip()
            val = raw_val.strip()
            match = self._match_value(val, attr)
            if match is None:
                continue
            host_matches.append(
                HostMatch(
                    info=match.info,
                    match_quality_score=match.match_quality_score,
                    pair_index=idx,
                    attribute=attr,
                    value=val,
                    match_tier=match.match_tier,
                    tier_taxon_names=match.tier_taxon_names,
                    needs_review=match.identifier_disagreement,
                )
            )

        if not host_matches:
            return None

        # Sort key (smaller wins):
        #   1. score              higher score wins
        #   2. table_priority     more specific taxon wins
        #   3. attr priority      host_taxid > host > other
        #   4. pair_index         earlier position as last-resort tiebreaker
        host_matches.sort(
            key=lambda c: (
                -c.match_quality_score,
                c.info.table_priority,
                _attr_priority(c.attribute),
                c.pair_index,
            )
        )
        best = host_matches[0]

        distinct_taxa = {host_match.info.taxid for host_match in host_matches}
        has_multiple_taxa = len(distinct_taxa) > 1
        has_ambiguous_subset = bool(best.tier_taxon_names)
        is_subset_match = best.match_tier != ""
        has_identifier_disagreement = best.needs_review
        needs_review = (
            is_subset_match
            or has_ambiguous_subset
            or has_identifier_disagreement
            or has_multiple_taxa
        )
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
            match_quality_score=best.match_quality_score,
            pair_index=best.pair_index,
            attribute=best.attribute,
            value=best.value,
            match_tier=best.match_tier,
            tier_taxon_names=best.tier_taxon_names,
            needs_review=needs_review,
            diagnostics=diagnostics,
        )
