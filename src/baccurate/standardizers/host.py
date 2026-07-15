"""
Map host annotations from sample metadata to NCBI Taxonomy IDs and
scientific names (binomial nomenclature).

See docs/host.md for the documentation.
"""

import csv
import logging
import re
import string
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from baccurate.paths import DEFAULT_TAXIDS_CURATED, DEFAULT_TAXIDS_NCBI, HOST_OUTPUT, HOST_OVERFLOW
from baccurate.utils.args import create_arg_parser
from baccurate.utils.config import load_config
from baccurate.utils.logging import setup_standardizer_logging
from baccurate.utils.progress import count_tsv_rows, make_inner_bar
from baccurate.utils.text import split_pipe_separated

logger = logging.getLogger(__name__)

# --- Match-quality scores ---

# Direct numeric taxid, scientific name or synonym
SCORE_TAXID = 1.0
SCORE_SCINAME = 1.0
SCORE_SYNONYM = 1.0

# Locally-curated keyword match
SCORE_KEYWORD = 0.95

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


# --- Main class ---


class HostStandardizer:
    def __init__(
        self,
        config_path: Path | str,
        ncbi_table_path: Path | str = DEFAULT_TAXIDS_NCBI,
        curated_table_path: Path | str = DEFAULT_TAXIDS_CURATED,
    ) -> None:
        self.config = load_config(config_path)
        self._build_lookups(Path(ncbi_table_path), Path(curated_table_path))
        self._compile_filters()

    def _build_lookups(self, ncbi_table_path: Path, curated_table_path: Path) -> None:
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

        curated_df = pd.read_csv(curated_table_path, sep="\t", dtype={"keywords": str})

        # Lookups by source/score tier. The split exists because each
        # gets a different reliability score in the matching cascade.
        self.taxid_to_info: dict[str, TaxonInfo] = {}
        self.sciname_to_info: dict[str, TaxonInfo] = {}
        self.synonym_to_info: dict[str, TaxonInfo] = {}
        self.keyword_to_info: dict[str, TaxonInfo] = {}
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
        # Order: scinames -> synonyms -> keywords -> NCBI commons -> broad.
        synonym_entries: list[tuple[str, TaxonInfo]] = []
        keyword_entries: list[tuple[str, TaxonInfo]] = []
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

        # Curated keywords reuse TaxonInfo from the NCBI rows so they share
        # table_priority for tie-breaking.
        if curated_df is not None:
            for _, row in curated_df.iterrows():
                taxid_str = str(row["taxid"])
                info = self.taxid_to_info.get(taxid_str)
                if info is None:
                    logger.warning(
                        "Curated row for taxid %s has no matching NCBI row - skipped",
                        taxid_str,
                    )
                    continue
                for term in self._split_cell(row.get("keywords")):
                    keyword_entries.append((_normalize_text(term), info))

        for norm, info in synonym_entries:
            if not norm:
                continue
            self.synonym_to_info.setdefault(norm, info)
            self._index_for_subset(norm, info)

        for norm, info in keyword_entries:
            if not norm:
                continue
            self.keyword_to_info.setdefault(norm, info)
            self._index_for_subset(norm, info)

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

        logger.info(
            "Loaded lookup tables: %d taxids, %d unique scinames, "
            "%d unique synonyms, %d unique keywords, "
            "%d unique NCBI curated common names, "
            "%d unique NCBI broad common names, "
            "%d multi-word subset terms, %d single-word subset terms",
            len(self.taxid_to_info),
            len(self.sciname_to_info),
            len(self.synonym_to_info),
            len(self.keyword_to_info),
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

    def _index_for_subset(self, norm_term: str, info: TaxonInfo) -> None:
        """
        Add a normalized term to the subset-matching index.

        Subspecies are excluded - their trinomial names produce false
        positives when bag-of-words matching ignores order (e.g. 'Gallus
        gallus gallus' matching the input 'Gallus gallus'). They remain
        in the exact-match lookups, so a value typed as the full
        trinomial still resolves correctly.
        """
        if not norm_term or info.rank == "subspecies":
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
            re.compile(re.escape(str(s)), re.IGNORECASE)
            for s in self.config.get("ignored_substrings", [])
        ]
        # Each iso_keyword is stored alongside its normalized word set.
        # Matching is whole-word: all the keyword's words must appear
        # as whole words in the value's normalized word set.
        self.iso_keywords: list[tuple[str, frozenset[str]]] = []
        for kw in self.config.get("iso_keywords", []):
            words = frozenset(_normalize_text(str(kw)).split())
            if words:
                self.iso_keywords.append((str(kw), words))

        # Manual overrides keyed by normalized value: int = force taxid,
        # None = reject and forward to iso_source. Validated at startup
        # so a typo'd taxid fails fast.
        self.taxid_overrides: dict[str, int | None] = {}
        raw_overrides = (self.config.get("overrides") or {}).get("taxid_overrides") or {}
        for raw_key, raw_taxid in raw_overrides.items():
            norm_key = _normalize_text(str(raw_key))
            if not norm_key:
                continue
            if raw_taxid is None:
                self.taxid_overrides[norm_key] = None
                continue
            taxid_str = str(raw_taxid)
            if taxid_str not in self.taxid_to_info:
                raise ValueError(
                    f"Override for {raw_key!r} points to taxid {raw_taxid} "
                    f"which is not present in the taxid table."
                )
            self.taxid_overrides[norm_key] = int(raw_taxid)
        if self.taxid_overrides:
            logger.info("Loaded %d manual taxid override(s)", len(self.taxid_overrides))

    # --- Per-value matching ---

    def _strip_ignored_substrings(self, value: str) -> str:
        for pattern in self.ignored_patterns:
            value = pattern.sub("", value)
        return value

    def _match_numeric_value(self, normalized: str) -> ValueMatch | None:
        info = self.taxid_to_info.get(normalized)
        if info is None:
            logger.debug("No taxid match for %r", normalized)
            return None
        logger.debug(
            "Taxid match: %r -> %s (taxid %d)", normalized, info.scientific_name, info.taxid
        )
        return ValueMatch(info, SCORE_TAXID)

    def _match_text_value(self, normalized: str) -> ValueMatch | None:
        # Tiers tried in priority order. Sciname and synonym both score
        # 1.0, but sciname is checked first to keep the logged tier
        # label honest.
        for label, lookup, score in (
            ("sciname", self.sciname_to_info, SCORE_SCINAME),
            ("synonym", self.synonym_to_info, SCORE_SYNONYM),
            ("keyword", self.keyword_to_info, SCORE_KEYWORD),
            ("curated common name", self.curated_common_to_info, SCORE_CURATED_COMMON),
            ("broad common name", self.broad_common_to_info, SCORE_BROAD_COMMON),
        ):
            info = lookup.get(normalized)
            if info is not None:
                logger.debug(
                    "%s exact match: %r -> %s (taxid %d, score %.2f)",
                    label,
                    normalized,
                    info.scientific_name,
                    info.taxid,
                    score,
                )
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

        logger.debug("Subset match for %r: search_words=%s", normalized, sorted(search_words))

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
            logger.debug(
                "Multi-word matches for %r: %s",
                normalized,
                sorted({(i.scientific_name, i.taxid) for i in multiword_matches}),
            )
            return self._build_subset_match(multiword_matches, SCORE_SUBSET_MULTIWORD, "multi-word")
        logger.debug("No multi-word match for %r", normalized)

        # --- Single-word terms (fallback) ---
        singleword_matches: list[TaxonInfo] = []
        for w in search_words:
            info = self.singleword_term_to_info.get(w)
            if info is not None:
                singleword_matches.append(info)

        if singleword_matches:
            logger.debug(
                "Single-word matches for %r: %s",
                normalized,
                sorted({(i.scientific_name, i.taxid) for i in singleword_matches}),
            )
            return self._build_subset_match(
                singleword_matches, SCORE_SUBSET_SINGLEWORD, "single-word"
            )
        logger.debug("No single-word match for %r", normalized)

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
            logger.debug("Empty after normalization: %r (attribute=%r)", value, attribute)
            return None
        if normalized.isdigit():
            if attribute.lower() != "host_taxid":
                logger.debug(
                    "Numeric value %r in non-host_taxid attribute %r - skipped",
                    normalized,
                    attribute,
                )
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
                    logger.debug("iso_keyword match: %r in value %r", original, value)
                    return original
        return None

    def _check_overrides(self, val_str: str) -> tuple[str, str | int | None]:
        """
        Check whether any value in the row hits a manual override.

        Returns one of:
          ("none",   None)    no override applies; proceed with normal matching
          ("reject", raw_val) any value mapped to null; row goes to iso_source
          ("force",  taxid)   any value mapped to a taxid; force that match

        """
        if not self.taxid_overrides:
            return "none", None
        forced: tuple[str, int] | None = None
        for value in split_pipe_separated(val_str):
            norm = _normalize_text(value)
            if norm not in self.taxid_overrides:
                continue
            target = self.taxid_overrides[norm]
            if target is None:
                return "reject", value.strip()
            if forced is None:
                forced = (value.strip(), target)
        if forced is not None:
            return "force", forced[1]
        return "none", None

    def _build_override_match(self, val_str: str, attributes_str: str, taxid: int) -> HostMatch:
        """Build a HostMatch from a forced taxid, picking the first value that triggered it."""
        attributes = split_pipe_separated(attributes_str)
        values = split_pipe_separated(val_str)
        info = self.taxid_to_info[str(taxid)]
        for idx, (raw_attr, raw_val) in enumerate(
            zip(attributes, values, strict=False)
        ):
            if self.taxid_overrides.get(_normalize_text(raw_val)) == taxid:
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
        raise AssertionError(f"_build_override_match called for taxid {taxid} but no value matched")

    # --- Per-record dispatch ---

    def classify_row(
        self,
        accession: str,
        attr_str: str,
        val_str: str,
        skip_iso_keywords: bool = False,
    ) -> HostMatch | None:
        """Run the full per-row cascade: iso_keyword -> override -> match.

        Returns the winning HostMatch when a host is identified (including
        forced overrides). Returns None when the row should be forwarded to
        the iso pipeline (iso_keyword hit, reject override, or no match).

        `skip_iso_keywords` bypasses the iso_keyword guard. Used when
        the value has already been classified by the iso pipeline, to find
        the source organism named in it (e.g.'chicken meat' -> Gallus gallus).
        """
        if not skip_iso_keywords:
            iso_keyword = self._find_iso_keyword(val_str)
            if iso_keyword is not None:
                logger.warning(
                    "%s: value %r matches iso_keyword %r - forwarding to iso_source",
                    accession,
                    val_str,
                    iso_keyword,
                )
                return None

        outcome, payload = self._check_overrides(val_str)
        if outcome == "reject":
            logger.warning(
                "%s: value %r overridden as reject - forwarding to iso_source",
                accession,
                payload,
            )
            return None
        if outcome == "force":
            match = self._build_override_match(val_str, attr_str, payload)
            logger.warning(
                "%s: value %r overridden to taxid %d (%s)",
                accession,
                match.value,
                match.info.taxid,
                match.info.scientific_name,
            )
            return match

        return self.find_best_match(accession, attr_str, val_str)

    def classify_values(
        self,
        rows: Iterable[tuple[str, str, str]],
        skip_iso_keywords: bool = False,
    ) -> Iterator[tuple[str, "HostMatch"]]:
        """Yield (accession, HostMatch) for each row that classifies as host.

        Non-matches and rows forwarded to iso are silently dropped - this is the
        building block for the pass-3 retry, which only cares about confirmed hits.
        """
        for accession, attr_str, val_str in rows:
            match = self.classify_row(
                accession,
                attr_str,
                val_str,
                skip_iso_keywords=skip_iso_keywords,
            )
            if match is not None:
                yield accession, match

    def find_best_match(
        self,
        accession: str,
        attributes_str: str,
        values_str: str,
    ) -> HostMatch | None:
        attributes = split_pipe_separated(attributes_str)
        values = split_pipe_separated(values_str)

        logger.debug(
            "%s: matching values=%r against attrs=%r", accession, values_str, attributes_str
        )

        candidates: list[HostMatch] = []
        for idx, (raw_attr, raw_val) in enumerate(
            zip(attributes, values, strict=False)
        ):
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
            logger.debug("%s: no candidates", accession)
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

        logger.debug(
            "%s: ranked candidates: %s; chose %s (taxid %d, score %.1f, attribute %r)",
            accession,
            [(c.info.scientific_name, c.info.taxid, c.attribute, c.score) for c in candidates],
            best.info.scientific_name,
            best.info.taxid,
            best.score,
            best.attribute,
        )

        distinct_taxa = {c.info.taxid for c in candidates}
        has_multiple_taxa = len(distinct_taxa) > 1
        has_ambiguous_subset = bool(best.tier_candidates)
        is_subset_match = best.match_tier != ""
        low_confidence = is_subset_match or has_ambiguous_subset or has_multiple_taxa

        if has_multiple_taxa:
            logger.warning(
                "%s: %d distinct host taxa across attributes - keeping %s (taxid %d). "
                "Candidates: %s",
                accession,
                len(distinct_taxa),
                best.info.scientific_name,
                best.info.taxid,
                " | ".join(
                    f"{c.attribute}={c.value!r}->{c.info.scientific_name}" for c in candidates
                ),
            )

        if has_ambiguous_subset:
            logger.warning(
                "%s: %s subset match for %r had multiple candidate taxa [%s] "
                "- using table-priority pick %s",
                accession,
                best.match_tier,
                best.value,
                ", ".join(best.tier_candidates),
                best.info.scientific_name,
            )

        if low_confidence:
            logger.debug(
                "%s: flagged low_confidence (subset=%s ambiguous=%s multi_taxa=%s)",
                accession,
                is_subset_match,
                has_ambiguous_subset,
                has_multiple_taxa,
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
        )

    # --- File processing ---

    def process_file(
        self,
        input_path: Path,
        host_path: Path,
        overflow_path: Path,
        pathogen: str | None = None,
        disable_progress: bool = False,
    ) -> None:
        """Classify host rows from input_path; matches go to host_path, the rest
        (forwarded by iso_keyword/override, or unmatched) go to overflow_path
        for downstream iso processing.
        """
        host_header = [
            "accession",
            "host_taxid",
            "host_sci_name",
            "host_score",
            "host_low_conf",
            "host_attr_orig",
            "host_val_orig",
        ]
        overflow_header = ["accession", "attribute", "value", "package"]

        total = count_tsv_rows(input_path)
        bar_desc = f"host [{pathogen}]" if pathogen else "host"

        with (
            input_path.open("r", encoding="utf-8", newline="") as infile,
            host_path.open("w", encoding="utf-8", newline="") as out_host,
            overflow_path.open("w", encoding="utf-8", newline="") as out_overflow,
            make_inner_bar(total, bar_desc, disable=disable_progress) as bar,
        ):
            reader = csv.DictReader(infile, delimiter="\t")
            writer_host = csv.writer(
                out_host, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            writer_overflow = csv.writer(
                out_overflow, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            writer_host.writerow(host_header)
            writer_overflow.writerow(overflow_header)

            for row in reader:
                if pathogen and row.get("pathogen") != pathogen:
                    bar.update(1)
                    continue

                accession = row.get("accession", "")
                attr_str = row.get("host_attr_orig", "") or ""
                val_str = row.get("host_val_orig", "") or ""
                package = row.get("package", "") or ""

                match = self.classify_row(accession, attr_str, val_str)
                if match is not None:
                    writer_host.writerow(
                        [
                            accession,
                            match.info.taxid,
                            match.info.scientific_name,
                            match.score,
                            match.low_confidence,
                            match.attribute,
                            match.value,
                        ]
                    )
                elif val_str.strip():
                    writer_overflow.writerow([accession, attr_str, val_str, package])
                bar.update(1)


def main(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    log_level: str = "INFO",
    pathogen: str | None = None,
    overflow_path: Path | None = None,
    disable_progress: bool = False,
) -> None:
    setup_standardizer_logging(logger, output_path, "host_standardized", log_level)

    standardizer = HostStandardizer(config_path)
    if overflow_path is None:
        overflow_path = output_path.parent / HOST_OVERFLOW
    standardizer.process_file(
        input_path, output_path, overflow_path, pathogen=pathogen, disable_progress=disable_progress
    )


if __name__ == "__main__":
    parser = create_arg_parser(
        description="Standardize free-text host values from a TSV file into NCBI taxids.",
        default_config_path="config/host.yaml",
    )
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_dir) / HOST_OUTPUT
    cfg_path = Path(args.config)

    main(in_path, out_path, cfg_path, args.log_level)
