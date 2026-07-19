"""
Maps isolation source annotations from sample metadata to curated ontology
terms via a single-call LLM classifier.

See docs/isolation_source.md for the full pipeline description.
"""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import instructor
import openai
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from baccurate.llm.client import LLMSettings, load_llm_client, load_llm_settings
from baccurate.llm.diagnostics import LLMFailureCategory, observe_llm_call
from baccurate.paths import DEFAULT_ISO_CACHE_DB, DEFAULT_ONTOLOGY_TSV
from baccurate.standardizers.host import HostOverflowContext
from baccurate.standardizers.iso_renderer import (
    render_ontology,
    valid_display_terms,
)
from baccurate.utils.cache import SQLiteKVCache
from baccurate.utils.config import load_config
from baccurate.utils.text import normalize_keyword, split_pipe_separated

logger = logging.getLogger(__name__)
_LOAD_CONFIGURED_CLIENT = object()

# --- Constants ---

ONTOLOGY_ID_PATTERN = re.compile(r"\b([A-Z]+:\d+)\b", re.IGNORECASE)
HOST_RETRY_TRIGGERS: tuple[str, ...] = (
    "host-associated",
    "environmental:anthropogenic environment:food:animal product",
    "environmental:anthropogenic environment:food:plant food product",
)

# --- Data structures ---


@dataclass(frozen=True, slots=True)
class StandardizedSource:
    """Standardization result for one record."""

    categories: str
    display_terms: str
    ontology_links: str
    term_paths: str
    reasoning: list[dict]


class IsolationDiagnostic(StrEnum):
    """The fixed set of isolation-source results used in build reports."""

    NO_CANDIDATES = "no_candidates"
    EXACT_MATCH = "exact_match"
    CACHE_HIT = "cache_hit"
    LLM_CALL = "llm_call"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True, slots=True)
class IsolationOrigin:
    """One source attribute/value pair used for isolation classification."""

    attribute: str
    value: str


@dataclass(frozen=True, slots=True)
class IsolationOutcome:
    """Typed isolation-source classification for one extracted record."""

    categories: str
    display_terms: str
    ontology_links: str
    term_paths: str
    host_context: str
    origins: tuple[IsolationOrigin, ...]
    reasoning: tuple[dict, ...]
    diagnostics: tuple[IsolationDiagnostic, ...]
    exact_matches: int
    cache_hits: int
    llm_calls: int

    @property
    def standardized_term_paths(self) -> tuple[str, ...]:
        """Selected ontology paths as structured values."""
        return tuple(split_pipe_separated(self.term_paths)) if self.term_paths else ()

    @property
    def host_retry_eligible(self) -> bool:
        """Whether this classification can support another host attempt."""
        return any(
            path == trigger or path.startswith(f"{trigger}:")
            for path in self.standardized_term_paths
            for trigger in HOST_RETRY_TRIGGERS
        )


@dataclass(frozen=True, slots=True)
class IsolationRejection:
    """A record with no isolation-source candidates to classify."""

    diagnostics: tuple[IsolationDiagnostic, ...]


# --- Cache ---


class SQLiteCache(SQLiteKVCache):
    """SQLite-backed cache keyed on normalized evidence and model hashes."""

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

    def _generate_hash(self, attr_name: str, value: str, host: str, model: str = "") -> str:
        norm_attr = normalize_keyword(str(attr_name).strip())
        norm_value = normalize_keyword(str(value).strip())
        norm_host = normalize_keyword(str(host).strip())
        norm_model = (model or "").strip().lower()

        raw_string = f"{norm_attr}|{norm_value}|{norm_host}|{norm_model}"
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
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.config = config
        self.ont = ontology_manager
        self.cache = cache_manager
        self.stats = {"cache_hits": 0, "exact_matches": 0, "llm_calls": 0}

        if client is _LOAD_CONFIGURED_CLIENT:
            raw_client, env_model = load_llm_client(llm_settings)
        else:
            raw_client = client
            settings = llm_settings or load_llm_settings()
            env_model = settings.model
        self._raw_client = raw_client
        try:
            self.model = env_model or ""
            self.client = instructor.from_openai(raw_client) if raw_client else None

            valid_set = {normalize_keyword(t) for t in valid_display_terms(self.ont)}
            self._schema = _build_schema(valid_set)
            self._ontology_block = render_ontology(self.ont)

            system_template = self.config.get("system_prompt") or ""
            self.system_prompt = system_template.replace("{ontology_tree}", self._ontology_block)
            self.user_template = self.config.get("user_prompt")
        except BaseException:
            if raw_client is not None:
                raw_client.close()
            raise

    def close(self) -> None:
        if self._raw_client is not None:
            self._raw_client.close()

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
    ) -> StandardizedSource:
        """Classify one record through deterministic matching, cache, and model fallback."""

        attrs = split_pipe_separated(str(attr_name))
        vals = split_pipe_separated(str(value))
        valid_attrs, valid_vals = [], []
        for a, v in zip(attrs, vals, strict=False):
            if v.strip() == "":
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
                        "reasoning": "No non-empty candidate values were provided.",
                        "selections": [],
                    }
                ],
            )

        evidence_attributes = "||".join(valid_attrs)
        evidence_values = "||".join(valid_vals)

        # Direct-match pass over each (attr, val) pair before calling the LLM.
        direct_paths: set[str] = set()
        direct_match_count = 0
        for v in valid_vals:
            path = self._direct_match(v)
            if path is not None:
                direct_paths.add(path)
                direct_match_count += 1
                self.stats["exact_matches"] += 1

        final_nodes: set[str] = set()
        reasoning_history: list[dict] = []

        direct_covers_all = direct_match_count == len(valid_vals)

        if direct_covers_all:
            final_nodes |= direct_paths
            reasoning_history.append(
                {
                    "node": "direct_match",
                    "reasoning": "All values resolved manually.",
                    "selections": sorted(direct_paths),
                }
            )
        else:
            cached_result = self.cache.get(evidence_attributes, evidence_values, host, self.model)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result

            if self.client is None:
                final_nodes |= direct_paths
                reasoning_history.append(
                    {
                        "node": "classifier",
                        "reasoning": "LLM classification is disabled.",
                        "selections": [],
                    }
                )
            else:
                metadata_block = self._format_metadata(valid_attrs, valid_vals, host)
                user_prompt = self.user_template.format(
                    metadata=metadata_block,
                    bioproject_context="",
                )

                try:
                    self.stats["llm_calls"] += 1
                    with observe_llm_call(
                        accession=accession,
                        target="isolation",
                        model=self.model,
                    ) as call:
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            response_model=self._schema,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0,
                            max_retries=3,
                        )
                    call.accepted()
                except Exception as e:
                    if isinstance(e, openai.APIError):
                        call.validation_retries_exhausted()
                    else:
                        call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
                    raise RuntimeError(
                        f"Isolation-source LLM failed for accession {accession}"
                    ) from e

                llm_paths: set[str] = set()
                for term in resp.terms:
                    path = self.ont.display_term_to_path.get(normalize_keyword(term))
                    if path is not None:
                        llm_paths.add(path)
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

        if not direct_covers_all and self.client is not None:
            self.cache.set(evidence_attributes, evidence_values, host, final_record, self.model)
        return final_record


class IsoStandardizer:
    def __init__(
        self,
        config_path: Path | str,
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.config = load_config(config_path)

        db_path = self.config.get("cache_db_path", DEFAULT_ISO_CACHE_DB)
        ontology_path = self.config.get("ontology_tsv_path", DEFAULT_ONTOLOGY_TSV)

        self.cache = SQLiteCache(db_path)
        try:
            self.ontology = OntologyManager(ontology_path)
            self.pipeline = LLMClassifier(
                self.config,
                self.ontology,
                self.cache,
                result_logger=self.logger,
                client=client,
                llm_settings=llm_settings,
            )
        except BaseException:
            self.cache.close()
            raise
        self.logger.info("IsoStandardizer initialised (LLMClassifier).")

    def standardize(
        self,
        record: Mapping[str, str],
        *,
        host_context: str,
        overflow: HostOverflowContext | None = None,
    ) -> IsolationOutcome | IsolationRejection:
        """Classify one record with optional in-memory host overflow context."""
        attributes = str(record.get("iso_attr_orig", "") or "")
        values = str(record.get("iso_val_orig", "") or "")
        if overflow is not None and overflow.value.strip():
            attributes = "||".join(part for part in (attributes, overflow.attribute) if part)
            values = "||".join(part for part in (values, overflow.value) if part)

        attribute_parts = split_pipe_separated(attributes)
        value_parts = split_pipe_separated(values)
        if len(attribute_parts) != len(value_parts):
            accession = str(record.get("accession", "") or "")
            raise ValueError(
                f"Malformed isolation-source candidates for accession {accession}: "
                f"{len(attribute_parts)} attributes for {len(value_parts)} values"
            )
        origins = tuple(
            IsolationOrigin(attribute.strip(), value.strip())
            for attribute, value in zip(
                attribute_parts,
                value_parts,
                strict=True,
            )
            if value.strip()
        )
        if not origins:
            return IsolationRejection((IsolationDiagnostic.NO_CANDIDATES,))

        before = dict(self.pipeline.stats)
        standardized = self.pipeline.standardize_record(
            str(record.get("accession", "") or ""),
            "||".join(origin.attribute for origin in origins),
            "||".join(origin.value for origin in origins),
            host_context,
        )
        exact_matches = self.pipeline.stats["exact_matches"] - before["exact_matches"]
        cache_hits = self.pipeline.stats["cache_hits"] - before["cache_hits"]
        llm_calls = self.pipeline.stats["llm_calls"] - before["llm_calls"]
        diagnostics = []
        if exact_matches:
            diagnostics.append(IsolationDiagnostic.EXACT_MATCH)
        if cache_hits:
            diagnostics.append(IsolationDiagnostic.CACHE_HIT)
        if llm_calls:
            diagnostics.append(IsolationDiagnostic.LLM_CALL)
        if not standardized.term_paths:
            diagnostics.append(IsolationDiagnostic.UNSPECIFIED)
        return IsolationOutcome(
            categories=standardized.categories,
            display_terms=standardized.display_terms,
            ontology_links=standardized.ontology_links,
            term_paths=standardized.term_paths,
            host_context=host_context,
            origins=origins,
            reasoning=tuple(standardized.reasoning),
            diagnostics=tuple(diagnostics),
            exact_matches=exact_matches,
            cache_hits=cache_hits,
            llm_calls=llm_calls,
        )

    def close(self) -> None:
        try:
            self.pipeline.close()
        finally:
            self.cache.close()
