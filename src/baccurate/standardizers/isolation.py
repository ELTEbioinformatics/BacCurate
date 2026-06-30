"""
Maps isolation source annotations from sample metadata to curated ontology
terms via a single-call LLM classifier.

See docs/isolation_source.md for the full pipeline description.
"""

import csv
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import instructor
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from baccurate.paths import DEFAULT_ISO_CACHE_DB, DEFAULT_ONTOLOGY_TSV, ISO_OUTPUT
from baccurate.standardizers.iso_renderer import (
    render_ontology,
    valid_display_terms,
)
from baccurate.utils.args import create_arg_parser
from baccurate.utils.cache import SQLiteKVCache
from baccurate.utils.config import load_config
from baccurate.utils.llm import load_llm_client
from baccurate.utils.logging import setup_standardizer_logging
from baccurate.utils.progress import count_tsv_rows, make_inner_bar
from baccurate.utils.text import normalize_keyword, split_pipe_separated

logger = logging.getLogger(__name__)

# --- Constants ---

ONTOLOGY_ID_PATTERN = re.compile(r"\b([A-Z]+:\d+)\b", re.IGNORECASE)

# Patterns for masking variable data before cache hashing.
MASKING_PATTERNS: list[tuple[re.Pattern, str]] = [
            (re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'), '<DATE>'), # e.g.,2022-05-21
            (re.compile(r'\b\d{2}[-/]\d{2}[-/]\d{4}\b'), '<DATE>'), # e.g., 21/05/2022
            (re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b', re.IGNORECASE), # e.g., Dec 2024
             '<DATE>'),
            (re.compile(r'\b\d+\.\d+\s*[NSEW]\b', re.IGNORECASE), '<COORD>'), # 32.522 N, 113.2154 W
            (re.compile(r'\b\d+(?:\.\d+)?\s*(?:[muµ]?l|[mkµ]?g|[cmµ]?m)\b', re.IGNORECASE), '<MEASURE>'), # common lab units
            (re.compile(r'\b\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?\s*(?:[cmµ]?m)?\b', re.IGNORECASE), '<DIMENSION>'), # e.g., 13x100mm, 2x4
            (re.compile(r'\b[A-Za-z]+[-_](?<!:)\d+\b'), '<ID>'),  # identifiers with separators, e.g. Patient-1
            (re.compile(r'\b[A-Za-z]{1,3}\d+\b'), '<ID>'),  # short identifiers without separators, e.g. A1 or b2
            (re.compile(r'(?<![:\d])\d+'), '<NUM>')     # e.g. 32 or patient1. ontology-IDs (ENVO:00...) kept intact
        ]

# --- Data structures ---


@dataclass(frozen=True, slots=True)
class StandardizedSource:
    """Standardization result for one record."""

    categories: str
    display_terms: str
    ontology_links: str
    term_paths: str
    reasoning: list[dict]


# --- Cache ---


class SQLiteCache(SQLiteKVCache):
    """SQLite-backed cache keyed on (attr, masked_value, host, model) hashes."""

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS cache (
            hash_id TEXT PRIMARY KEY,
            category TEXT,
            display_term TEXT,
            ontology_link TEXT,
            term_path TEXT,
            reasoning TEXT
        )
    """

    def __init__(self, db_path: Path | str = DEFAULT_ISO_CACHE_DB) -> None:
        super().__init__(db_path)

    @staticmethod
    def _mask_value(text: str) -> str:
        masked_text = str(text).lower()
        for pattern, replacement in MASKING_PATTERNS:
            masked_text = pattern.sub(replacement, masked_text)
        return masked_text.strip()

    def _generate_hash(self, attr_name: str, value: str, host: str, model: str = "") -> str:
        masked_value = self._mask_value(value)
        original_lower = str(value).strip().lower()
        if original_lower != masked_value:
            logger.info("[MASKING] %r --> %r", original_lower, masked_value)

        norm_attr = normalize_keyword(attr_name)
        norm_host = normalize_keyword(host)
        norm_model = (model or "").strip().lower()

        raw_string = f"{norm_attr}|{masked_value}|{norm_host}|{norm_model}"
        return self._sha256(raw_string)

    def get(
        self, attr_name: str, value: str, host: str, model: str = ""
    ) -> StandardizedSource | None:
        hash_id = self._generate_hash(attr_name, value, host, model)
        self.cursor.execute(
            "SELECT category, display_term, ontology_link, term_path, reasoning "
            "FROM cache WHERE hash_id=?",
            (hash_id,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None

        # Restore the original reasoning in case of cache hit
        reasoning = (
            json.loads(row[4])
            if row[4]
            else [{"node": "cache", "reasoning": "Cache hit", "selections": []}]
        )

        return StandardizedSource(
            categories=row[0],
            display_terms=row[1],
            ontology_links=row[2],
            term_paths=row[3] or "",
            reasoning=reasoning,
        )

    def set(
        self,
        attr_name: str,
        value: str,
        host: str,
        record: StandardizedSource,
        model: str = "",
    ) -> None:
        hash_id = self._generate_hash(attr_name, value, host, model)
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO cache
                (hash_id, category, display_term, ontology_link, term_path, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                hash_id,
                record.categories,
                record.display_terms,
                record.ontology_links,
                record.term_paths,
                json.dumps(record.reasoning),
            ),
        )
        self.conn.commit()


# --- Ontology graph ---


class OntologyManager:
    """Parses the ontology TSV into a tree with crosslinks and lookup indexes."""

    def __init__(self, ontology_tsv_path: Path | str):

        # parent_path -> list of full child_paths
        self.children_map: dict[str, list[str]] = defaultdict(list)

        # full_path -> {display_term, ontology_link}
        self.node_metadata: dict[str, dict[str, str]] = {}

        # term -> full_path
        self.exact_match_index: dict[str, str] = {}

        self.id_match_index: dict[str, str] = {}

        # display_term -> full_path. Distinct from
        # exact_match_index because that one also indexes synonyms, while this
        # one is the strict 1:1 mapping the LLM classifier uses to turn LLM
        # output back into term paths.
        self.display_term_to_path: dict[str, str] = {}

        # term_path -> [crosslinked term_paths]
        self.crosslink_map: dict[str, list[str]] = defaultdict(list)

        df = pd.read_csv(ontology_tsv_path, sep="\t")
        self._build_tree(df)
        self._resolve_crosslinks(df)

    def _build_tree(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            term_path = str(row.get("term", "")).strip()

            raw_display = row.get("display_term")
            if pd.notna(raw_display) and str(raw_display).strip():
                display_term = str(raw_display).strip()
            else:
                display_term = term_path.split(":")[-1] if term_path else ""
            ont_link = (
                str(row.get("ontology_link", "")).strip()
                if pd.notna(row.get("ontology_link"))
                else ""
            )
            comment = str(row.get("comment", "")).strip() if pd.notna(row.get("comment")) else ""
            synonyms_raw = (
                str(row.get("synonyms", "")).strip() if pd.notna(row.get("synonyms")) else ""
            )
            synonyms = (
                [s.strip() for s in synonyms_raw.split(";") if s.strip()] if synonyms_raw else []
            )

            self.node_metadata[term_path] = {
                "display_term": display_term,
                "ontology_link": ont_link,
                "comment": comment,
                "synonyms": synonyms,
            }

            norm_term = normalize_keyword(display_term)
            if norm_term not in self.exact_match_index:
                self.exact_match_index[norm_term] = term_path

            if norm_term:
                existing = self.display_term_to_path.get(norm_term)
                if existing is not None and existing != term_path:
                    logger.warning(
                        "Duplicate display_term %r maps to multiple paths: %r and %r. "
                        "LLM output for this display_term will be ambiguous.",
                        display_term,
                        existing,
                        term_path,
                    )
                self.display_term_to_path[norm_term] = term_path

            # Index synonyms for direct-match
            for syn in synonyms:
                norm_syn = normalize_keyword(syn)
                if norm_syn and norm_syn not in self.exact_match_index:
                    self.exact_match_index[norm_syn] = term_path

            if ont_link:
                for link in ont_link.split(";"):
                    link_upper = link.strip().upper()
                    if link_upper not in self.id_match_index:
                        self.id_match_index[link_upper] = term_path

            # Build the tree hierarchy. Each ':' in the term_path defines
            # a parent-child edge. The empty string acts as the root.
            parts = term_path.split(":")
            for i in range(len(parts)):
                current_path = ":".join(parts[: i + 1])
                parent_path = ":".join(parts[:i]) if i > 0 else ""
                if current_path not in self.children_map[parent_path]:
                    self.children_map[parent_path].append(current_path)

        logger.info(
            "Ontology tree: %d root nodes, %d total nodes.",
            len(self.children_map[""]),
            len(self.node_metadata),
        )

    def _resolve_crosslinks(self, df: pd.DataFrame) -> None:
        if "crosslink_term" not in df.columns:
            return

        for _, row in df.iterrows():
            term_path = str(row.get("term", "")).strip()
            raw = row.get("crosslink_term", "")
            if pd.isna(raw) or not str(raw).strip():
                continue

            for target in str(raw).split(";"):
                target = target.strip()
                if not target:
                    continue

                target_path = self.exact_match_index.get(normalize_keyword(target))
                if not target_path:
                    target_path = self.id_match_index.get(target.upper())

                if target_path is None:
                    logger.warning(
                        "Crosslink target %r for term %r not found in ontology indices.",
                        target,
                        term_path,
                    )
                    continue

                if target_path not in self.crosslink_map[term_path]:
                    self.crosslink_map[term_path].append(target_path)


# --- LLM classifier ---


def _build_schema(valid_display_terms_normalized: set[str]) -> type[BaseModel]:
    """Build a Pydantic model whose `terms` is validated against the ontology."""

    valid = valid_display_terms_normalized

    class IsolationClassification(BaseModel):
        reasoning: str = Field(..., description="Brief reason for the chosen terms.")
        terms: list[str] = Field(
            ...,
            description=(
                "Display names of nodes from the ontology above. "
                "Each must be COPIED VERBATIM from a bullet in the tree."
            ),
        )

        @field_validator("terms")
        @classmethod
        def _check_terms(cls, v: list[str]) -> list[str]:
            invalid = [t for t in v if normalize_keyword(t) not in valid]
            if invalid:
                raise ValueError(
                    "Unknown terms: "
                    + ", ".join(repr(t) for t in invalid)
                    + ". Each term must be the exact display name of a node "
                    "in the ontology above (e.g. 'rectum', 'blood', "
                    "'hospital'). Do not include the colon-separated path."
                )
            return v

    return IsolationClassification


class LLMClassifier:
    def __init__(
        self,
        config: dict,
        ontology_manager: OntologyManager,
        cache_manager: SQLiteCache,
    ) -> None:
        self.config = config
        self.ont = ontology_manager
        self.cache = cache_manager
        self.null_values = {normalize_keyword(v) for v in self.config.get("null_values", [])}

        # load expected attribute name -> term_path from config
        self.expected_attr_to_term: dict[str, str] = {}
        for attr, term in (self.config.get("expected") or {}).items():
            path = self.ont.display_term_to_path.get(normalize_keyword(term))
            self.expected_attr_to_term[normalize_keyword(attr)] = path

        self.stats = {"cache_hits": 0, "exact_matches": 0, "llm_calls": 0}

        raw_client, env_model = load_llm_client()
        self.default_model = self.config.get("default_model") or env_model or ""
        self.client = instructor.from_openai(raw_client) if raw_client else None

        valid_set = {normalize_keyword(t) for t in valid_display_terms(self.ont)}
        self._schema = _build_schema(valid_set)
        self._ontology_block = render_ontology(self.ont)

        system_template = self.config.get("system_prompt") or ""
        self.system_prompt = system_template.format(ontology_tree=self._ontology_block)
        self.user_template = self.config.get("user_prompt")

    def _direct_match(self, value: str) -> str | None:
        """Try ontology-ID and exact-display-name match. Returns a term path or None."""
        found_ids = ONTOLOGY_ID_PATTERN.findall(value)
        for ext_id in found_ids:
            ext_id_upper = ext_id.upper()
            if ext_id_upper in self.ont.id_match_index:
                return self.ont.id_match_index[ext_id_upper]
        norm_v = normalize_keyword(value)
        return self.ont.exact_match_index.get(norm_v)

    @staticmethod
    def _format_metadata(attrs: list[str], vals: list[str], host: str) -> str:
        lines = [f"{a} = {v}" for a, v in zip(attrs, vals, strict=True)]
        if host.strip():
            lines.append(f"host = {host}")
        return "Metadata:\n" + "\n".join(lines)

    def standardize_record(
        self,
        accession: str,
        attr_name: str,
        value: str,
        host: str,
        package: str,
        *,
        model: str | None = None,
        mode: str = "full",
    ) -> StandardizedSource:
        """Process one record.

        `mode` possible values:
        - "full: default behavior. This is the only mode that reads/writes the
          SQLite cache.
        - "direct_only" - deterministic matching only against the ontology nodes
        - "llm_only" - always call the LLM.
        """

        effective_model = (model or self.default_model or "").strip()
        logger.info(
            "%s - Value: %r | Host: %r | Package: %r | Model: %r",
            accession,
            value,
            host,
            package,
            effective_model,
        )

        # Split and filter null values
        attrs = split_pipe_separated(str(attr_name))
        vals = split_pipe_separated(str(value))
        valid_attrs, valid_vals = [], []
        for a, v in zip(attrs, vals):
            if normalize_keyword(v) in self.null_values or v.strip() == "":
                continue
            valid_attrs.append(a.strip())
            valid_vals.append(v.strip())

        if not valid_vals:
            return StandardizedSource(
                categories="unspecified",
                display_terms="unspecified",
                ontology_links="NA",
                term_paths="",
                reasoning=[
                    {
                        "node": "classifier",
                        "reasoning": "All values filtered as NA.",
                        "selections": [],
                    }
                ],
            )

        filtered_attr_str = "||".join(valid_attrs)
        filtered_val_str = "||".join(valid_vals)

        if mode == "full":
            cached_result = self.cache.get(
                filtered_attr_str, filtered_val_str, host, effective_model
            )
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result

        # Direct-match pass over each (attr, val) pair before calling the LLM.
        direct_paths: set[str] = set()
        if mode != "llm_only":
            for v in valid_vals:
                path = self._direct_match(v)
                if path is not None:
                    direct_paths.add(path)
                    self.stats["exact_matches"] += 1

        final_nodes: set[str] = set()
        reasoning_history: list[dict] = []

        direct_covers_all = bool(direct_paths) and len(direct_paths) >= len(valid_vals)

        if mode == "direct_only":
            final_nodes |= direct_paths
            reasoning_history.append(
                {
                    "node": "direct_match",
                    "reasoning": "direct_only mode: ontology-ID / exact display-term matches only.",
                    "selections": sorted(direct_paths),
                }
            )
        elif mode == "full" and direct_covers_all:
            final_nodes |= direct_paths
            reasoning_history.append(
                {
                    "node": "direct_match",
                    "reasoning": "All values resolved manually.",
                    "selections": sorted(direct_paths),
                }
            )
        else:
            metadata_block = self._format_metadata(valid_attrs, valid_vals, host)
            user_prompt = self.user_template.format(metadata=metadata_block)

            logger.debug("Classifier user prompt:\n%s", user_prompt)

            try:
                self.stats["llm_calls"] += 1
                resp = self.client.chat.completions.create(
                    model=effective_model,
                    response_model=self._schema,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_retries=3,
                )
                llm_paths: set[str] = set()
                for term in resp.terms:
                    path = self.ont.display_term_to_path.get(normalize_keyword(term))
                    if path is not None:
                        llm_paths.add(path)
                    else:
                        logger.warning(
                            "Display term %r passed validator but has no path mapping.",
                            term,
                        )
                final_nodes |= direct_paths
                final_nodes |= llm_paths

                reasoning_history.append(
                    {
                        "node": "classifier",
                        "reasoning": resp.reasoning,
                        "selections": sorted(llm_paths),
                        "selected_terms": list(resp.terms),
                    }
                )
                logger.info(
                    "Classifier: %r selected %r -> %r — %s",
                    accession,
                    list(resp.terms),
                    sorted(llm_paths),
                    resp.reasoning,
                )
            except Exception as e:
                logger.error("LLM call failed for %r: %s", accession, e)
                reasoning_history.append({"node": "classifier", "error": str(e), "selections": []})
                final_nodes |= direct_paths

        extra_crosslink_nodes: set[str] = set()
        for node in list(final_nodes):
            for linked in self.ont.crosslink_map.get(node, []):
                if linked not in final_nodes:
                    extra_crosslink_nodes.add(linked)
        if extra_crosslink_nodes:
            final_nodes |= extra_crosslink_nodes
            reasoning_history.append(
                {
                    "node": "crosslink",
                    "reasoning": "Crosslinks applied from selected terms.",
                    "selections": sorted(extra_crosslink_nodes),
                }
            )

        # warn when a configuree attribute is present but its expected node
        # (or a descendant) was not assigned anywhere in the record.
        for a in valid_attrs:
            expected = self.expected_attr_to_term.get(normalize_keyword(a))
            if expected and not any(
                p == expected or p.startswith(expected + ":") for p in final_nodes
            ):
                logger.warning(
                    "%s - attribute_shortcut mismatch: %r present but no %r (or child) in %r.",
                    accession, a, expected, sorted(final_nodes),
                )

        if not final_nodes:
            final_record = StandardizedSource(
                categories="unspecified",
                display_terms="unspecified",
                ontology_links="NA",
                term_paths="",
                reasoning=reasoning_history,
            )
        else:
            cat_list, term_list, link_list = [], [], []
            for node in sorted(final_nodes):
                cat_list.append(node.split(":")[0])
                meta = self.ont.node_metadata.get(node, {})
                term_list.append(meta.get("display_term", node.split(":")[-1]))
                link = meta.get("ontology_link", "")
                link_list.append(link if link else "NA")
            final_record = StandardizedSource(
                categories="||".join(sorted(set(cat_list))),
                display_terms="||".join(term_list),
                ontology_links="||".join(link_list),
                term_paths="||".join(sorted(final_nodes)),
                reasoning=reasoning_history,
            )

        if mode == "full":
            self.cache.set(
                filtered_attr_str, filtered_val_str, host, final_record, effective_model
            )
        return final_record


# ---- Orchestration ----


def _load_standardized_hosts(path: Path | str | None) -> dict[str, str]:
    """Map accession -> standardized host scientific name from host_standardized.tsv."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    mapping: dict[str, str] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = (row.get("accession") or "").strip()
            sci = (row.get("host_sci_name") or "").strip()
            if acc and sci:
                mapping[acc] = sci
    return mapping


class IsoStandardizer:
    def __init__(self, config_path: Path | str) -> None:
        self.config = load_config(config_path)

        db_path = self.config.get("cache_db_path", DEFAULT_ISO_CACHE_DB)
        ontology_path = self.config.get("ontology_tsv_path", DEFAULT_ONTOLOGY_TSV)

        self.cache = SQLiteCache(db_path)
        self.ontology = OntologyManager(ontology_path)
        self.pipeline = LLMClassifier(self.config, self.ontology, self.cache)
        logger.info("IsoStandardizer initialised (LLMClassifier).")

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        pathogen: str | None = None,
        additional_input: Path | None = None,
        host_standardized_path: Path | None = None,
        disable_progress: bool = False,
    ) -> None:
        """Classify iso rows from input_path and (optionally) additional_input.

        input_path is expected to follow the extracted_metadata schema (iso_attr_orig,
        iso_val_orig, host_val_orig, package columns; pathogen-filterable). additional_input
        follows the host-overflow schema (attribute, value, package; already
        filtered to one pathogen, no host context).
        """
        host_map = _load_standardized_hosts(host_standardized_path)

        if host_map:
            logger.info("Loaded %d standardized host values.", len(host_map))

        # Pre-load overflow rows
        overflow_by_acc: dict[str, list[tuple[str, str, str]]] = {}
        if additional_input is not None and additional_input.exists():
            with additional_input.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    value = str(row.get("value", "") or "").strip()
                    if not value:
                        continue
                    accession = row.get("accession", "")
                    attr = row.get("attribute", "") or ""
                    package = str(row.get("package", "") or "").strip()
                    overflow_by_acc.setdefault(accession, []).append((attr, value, package))

        output_header = [
            "accession",
            "iso_attr_orig",
            "iso_val_orig",
            "iso_terms",
            "iso_display_term",
            "iso_ontology_id",
            "iso_host",
        ]
        jsonl_path = output_path.with_suffix(".jsonl")
        records_processed = 0

        total = count_tsv_rows(input_path)
        if additional_input is not None and additional_input.exists():
            total += count_tsv_rows(additional_input)
        primary_desc = f"iso [{pathogen}]" if pathogen else "iso"
        overflow_desc = f"{primary_desc} (overflow)"

        try:
            with (
                output_path.open("w", encoding="utf-8", newline="") as outfile,
                jsonl_path.open("w", encoding="utf-8") as jsonl_out,
                make_inner_bar(total, primary_desc, disable=disable_progress) as bar,
            ):
                writer = csv.writer(outfile, delimiter="\t")
                writer.writerow(output_header)

                # Stage A: primary input (extracted_metadata schema).
                with input_path.open("r", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile, delimiter="\t")
                    for row in reader:
                        if pathogen and row.get("pathogen") != pathogen:
                            bar.update(1)
                            continue

                        accession = row.get("accession", "")
                        attr = row.get("iso_attr_orig", "") or ""
                        value = str(row.get("iso_val_orig", "") or "").strip()
                        raw_host = str(row.get("host_val_orig", "") or "").strip()
                        host = host_map.get(accession, raw_host)
                        package = str(row.get("package", "") or "").strip()

                        # Fold any forwarded host values for this accession into the same record
                        overflow_entries = overflow_by_acc.pop(accession, [])
                        attrs = [attr] if value else []
                        vals = [value] if value else []
                        for o_attr, o_val, _ in overflow_entries:
                            attrs.append(o_attr)
                            vals.append(o_val)

                        if not vals:
                            bar.update(1)
                            continue

                        records_processed += self._process_record(
                            writer,
                            jsonl_out,
                            accession,
                            "||".join(attrs),
                            "||".join(vals),
                            host,
                            package,
                        )
                        bar.update(1 + len(overflow_entries))


                if overflow_by_acc:
                    bar.set_description(overflow_desc)
                    for accession, entries in overflow_by_acc.items():
                        attrs = [e[0] for e in entries]
                        vals = [e[1] for e in entries]
                        package = next((e[2] for e in entries if e[2]), "")

                        # Overflow rows have no host context by definition (host failed).
                        records_processed += self._process_record(
                            writer,
                            jsonl_out,
                            accession,
                            "||".join(attrs),
                            "||".join(vals),
                            "",
                            package,
                        )
                        bar.update(len(entries))

            stats = self.pipeline.stats
            logger.info(
                "Processed %d records: %d cache hits, %d exact matches, %d LLM calls.",
                records_processed,
                stats["cache_hits"],
                stats["exact_matches"],
                stats["llm_calls"],
            )
        finally:
            self.cache.close()

    def _process_record(
        self,
        writer: "csv.writer",
        jsonl_out,
        accession: str,
        attr: str,
        value: str,
        host: str,
        package: str,
    ) -> int:
        res = self.pipeline.standardize_record(accession, attr, value, host, package)
        writer.writerow(
            [
                accession,
                attr,
                value,
                res.term_paths,
                res.display_terms,
                res.ontology_links,
                host,
            ]
        )
        jsonl_out.write(json.dumps({"accession": accession, "history": res.reasoning}) + "\n")
        return 1


def main(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    log_level: str = "INFO",
    pathogen: str | None = None,
    additional_input: Path | None = None,
    host_standardized_path: Path | None = None,
    disable_progress: bool = False,
) -> None:
    setup_standardizer_logging(logger, output_path, "iso_standardized", log_level)

    standardizer = IsoStandardizer(config_path)
    standardizer.process_file(
        input_path,
        output_path,
        pathogen=pathogen,
        additional_input=additional_input,
        host_standardized_path=host_standardized_path,
        disable_progress=disable_progress,
    )


if __name__ == "__main__":
    parser = create_arg_parser(default_config_path="config/isolation_source.yaml")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_dir) / ISO_OUTPUT
    cfg_path = Path(args.config)

    main(in_path, out_path, cfg_path, args.log_level)
