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
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal

import instructor
import openai
import pandas as pd
from pydantic import BaseModel, Field, create_model, field_validator

from baccurate.llm.client import LLMSettings, load_llm_client, load_llm_settings
from baccurate.llm.diagnostics import LLMFailureCategory, observe_llm_call
from baccurate.llm.request import CanonicalLLMRequest, canonical_json_sha256
from baccurate.paths import DEFAULT_ISO_CACHE_DB, DEFAULT_ONTOLOGY_TSV
from baccurate.source_snapshot import (
    SourceSnapshotError,
    bioproject_catalog_path_for,
)
from baccurate.standardizers.host import HostOverflowContext
from baccurate.standardizers.iso_renderer import (
    render_ontology,
    valid_display_terms,
)
from baccurate.utils.cache import SQLiteKVCache
from baccurate.utils.config import load_config
from baccurate.utils.text import normalize_keyword, split_pipe_separated

logger = logging.getLogger(__name__)

# None disables the LLM path here.
_LOAD_CONFIGURED_CLIENT = object()

# --- Constants ---

ONTOLOGY_ID_PATTERN = re.compile(r"\b([A-Z]+:\d+)\b", re.IGNORECASE)

# Branches whose terms name an organism rather than a place or a process, so a
# host may still be recoverable from the isolation-source text. Food products
# qualify because the material is the organism ('chicken meat', 'cow milk').
HOST_RETRY_TRIGGERS: tuple[str, ...] = (
    "host-associated",
    "environmental:anthropogenic environment:food:animal product",
    "environmental:anthropogenic environment:food:plant food product",
)

ISOLATION_LLM_PARAMETERS: dict[str, object] = {"temperature": 0, "seed": 100}
ISOLATION_RESPONSE_SCHEMA_ID = "baccurate.isolation.classification.v2"

# Relevance flags that count as origin evidence (spec excludes Medical/Evolution).
_ORIGIN_RELEVANCE = frozenset({"Agricultural", "Environmental", "Veterinary"})

# The ontology's explicit "looked and found nothing" term, as opposed to the
# empty string, which means the classification never produced a result.
_UNSPECIFIED_TERM = "unspecified"

# --- Prompts and fingerprints ---


@dataclass(frozen=True, slots=True)
class IsolationPrompts:
    """The isolation source prompt text used in canonical LLM requests."""

    system: str
    user_template: str
    bioproject_system: str
    bioproject_user: str


def effective_isolation_prompts(config: dict, ontology: "OntologyManager") -> IsolationPrompts:
    """Return prompt text with the run's ontology rendered into the system prompt."""
    system_template = config.get("system_prompt") or ""
    return IsolationPrompts(
        system=system_template.replace("{ontology_tree}", render_ontology(ontology)),
        user_template=config.get("user_prompt") or "",
        bioproject_system=config.get("bioproject_system_prompt") or "",
        bioproject_user=config.get("bioproject_user_prompt") or "",
    )


def ontology_semantics_fingerprint(
    node_metadata: Mapping[str, Mapping[str, object]],
    children_map: Mapping[str, list[str]],
    crosslink_map: Mapping[str, list[str]],
) -> str:
    """Fingerprint parsed ontology meaning independently of TSV formatting."""
    nodes = {
        term_path: {
            **metadata,
            "synonyms": sorted(metadata.get("synonyms", [])),
        }
        for term_path, metadata in sorted(node_metadata.items())
    }
    hierarchy = {parent: sorted(children) for parent, children in sorted(children_map.items())}
    crosslinks = {source: sorted(targets) for source, targets in sorted(crosslink_map.items())}
    return canonical_json_sha256({"nodes": nodes, "hierarchy": hierarchy, "crosslinks": crosslinks})


# --- Data structures ---


class IsolationEvidenceLevel(StrEnum):
    """The BioSample or BioProject level evidence supporting an isolation-source result."""

    SAMPLE = "sample"
    PROJECT = "project"
    SAMPLE_AND_PROJECT = "sample_and_project"
    NONE = "none"


class _IsolationRequestMode(StrEnum):
    """Which context the prompt actually carried."""

    SAMPLE_ONLY = "sample_only"
    PROJECT_ONLY = "project_only"
    COMBINED = "combined"


# The model cannot cite evidence it was never shown, so the request mode narrows
# the Literal of evidence levels its response schema will accept.
_EVIDENCE_LEVELS_BY_REQUEST_MODE = {
    _IsolationRequestMode.SAMPLE_ONLY: (
        IsolationEvidenceLevel.SAMPLE,
        IsolationEvidenceLevel.NONE,
    ),
    _IsolationRequestMode.PROJECT_ONLY: (
        IsolationEvidenceLevel.PROJECT,
        IsolationEvidenceLevel.NONE,
    ),
    _IsolationRequestMode.COMBINED: tuple(IsolationEvidenceLevel),
}


@dataclass(frozen=True, slots=True)
class StandardizedSource:
    """Standardization result for one record."""

    categories: str
    display_terms: str
    ontology_links: str
    term_paths: str
    reasoning: list[dict]
    evidence_level: IsolationEvidenceLevel = IsolationEvidenceLevel.NONE
    request_fingerprint: str | None = None


def _canonicalize_unspecified(record: StandardizedSource) -> StandardizedSource:
    """Represent an accepted empty classification as the explicit ontology term."""
    if record.term_paths:
        return record
    return replace(
        record,
        categories=_UNSPECIFIED_TERM,
        display_terms=_UNSPECIFIED_TERM,
        ontology_links="NA",
        term_paths=_UNSPECIFIED_TERM,
        evidence_level=IsolationEvidenceLevel.NONE,
    )


class IsolationDiagnostic(StrEnum):
    """The fixed set of isolation-source results used in build reports."""

    NO_CANDIDATES = "no_candidates"
    EXACT_MATCH = "exact_match"
    CACHE_HIT = "cache_hit"
    LLM_CALL = "llm_call"
    UNSPECIFIED = "unspecified"
    UNRESOLVED_BIOPROJECT_LINK = "unresolved_bioproject_link"


@dataclass(frozen=True, slots=True)
class IsolationOrigin:
    """One source attribute/value pair used for isolation classification."""

    attribute: str
    value: str


def _isolation_origins(
    accession: str,
    attributes: str,
    values: str,
) -> tuple[IsolationOrigin, ...]:
    """Parse aligned isolation candidates while preserving their raw pairing."""
    attribute_parts = split_pipe_separated(attributes)
    value_parts = split_pipe_separated(values)
    if len(attribute_parts) != len(value_parts):
        raise ValueError(
            f"Malformed isolation-source candidates for accession {accession}: "
            f"{len(attribute_parts)} attributes for {len(value_parts)} values"
        )
    return tuple(
        IsolationOrigin(attribute.strip(), value.strip())
        for attribute, value in zip(attribute_parts, value_parts, strict=True)
        if value.strip()
    )


@dataclass(frozen=True, slots=True)
class IsolationReasoningStep:
    """One step in the classifier's reasoning trace.

    `selections` are the term paths the step contributed; `selected_terms` are the
    display names the model returned verbatim, and only the classifier step has them.
    """

    node: str
    reasoning: str
    selections: tuple[str, ...]
    selected_terms: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "IsolationReasoningStep":
        """Type one raw step from the classifier or the cache."""
        return cls(
            node=str(value.get("node", "")),
            reasoning=str(value.get("reasoning", "")),
            selections=tuple(str(item) for item in value.get("selections", ())),
            selected_terms=tuple(str(item) for item in value.get("selected_terms", ())),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON representation used by run artifacts."""
        result: dict[str, object] = {
            "node": self.node,
            "reasoning": self.reasoning,
            "selections": list(self.selections),
        }
        if self.selected_terms:
            result["selected_terms"] = list(self.selected_terms)
        return result


@dataclass(frozen=True, slots=True)
class IsolationOutcome:
    """Typed isolation-source classification for one extracted record."""

    categories: str
    display_terms: str
    ontology_links: str
    term_paths: str
    evidence_level: IsolationEvidenceLevel
    host_context: str
    origins: tuple[IsolationOrigin, ...]
    sample_origins: tuple[IsolationOrigin, ...]
    reasoning: tuple[IsolationReasoningStep, ...]
    diagnostics: tuple[IsolationDiagnostic, ...]
    exact_matches: int
    cache_hits: int
    llm_calls: int
    request_fingerprint: str | None = None
    resolved_bioproject_accessions: tuple[str, ...] = ()

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


@dataclass(frozen=True, slots=True)
class ResolvedBioProjectContext:
    """Validated study-level context for one canonical BioProject accession."""

    id: str
    accession: str
    title: str
    description: str
    relevance: tuple[str, ...]

    def prompt_object(self) -> dict[str, object]:
        """Return the stable JSON shape supplied to the isolation prompt."""
        return {
            "id": self.id,
            "accession": self.accession,
            "title": self.title,
            "description": self.description,
            "relevance": list(self.relevance),
        }


def _load_bioproject_catalog(
    extracted_metadata_path: Path | str,
) -> dict[str, ResolvedBioProjectContext]:
    """Discover and validate the BioProject catalog paired with an extracted TSV."""
    catalog_path = bioproject_catalog_path_for(extracted_metadata_path)
    projects: dict[str, ResolvedBioProjectContext] = {}
    project_ids: set[str] = set()
    try:
        # A well-formed bundle always pairs a (possibly empty) catalog with the
        # TSV; a missing file means a broken bundle, not an unlinked sample set.
        stream = catalog_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise SourceSnapshotError(
            f"Invalid BioProject context catalog {catalog_path}: {exc}"
        ) from exc

    with stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceSnapshotError(
                    f"Invalid BioProject context catalog {catalog_path} line {line_number}: {exc}"
                ) from exc
            project = _validate_bioproject_context(raw, catalog_path, line_number)
            if project.id in project_ids:
                raise SourceSnapshotError(
                    f"Invalid BioProject context catalog {catalog_path} line "
                    f"{line_number}: duplicate project ID {project.id!r}"
                )
            if project.accession in projects:
                raise SourceSnapshotError(
                    f"Invalid BioProject context catalog {catalog_path} line "
                    f"{line_number}: duplicate project accession {project.accession!r}"
                )
            project_ids.add(project.id)
            projects[project.accession] = project
    return projects


# Public name for benchmark clients. Retain the established private name used by
# integrated regression tests until those callers migrate independently.
load_bioproject_catalog = _load_bioproject_catalog


def _validate_bioproject_context(
    raw: object,
    catalog_path: Path,
    line_number: int,
) -> ResolvedBioProjectContext:
    """Validate one catalog line into a ResolvedBioProjectContext, or raise."""
    expected_fields = {"id", "accession", "title", "description", "relevance"}
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise SourceSnapshotError(
            f"Invalid BioProject context catalog {catalog_path} line {line_number}: "
            f"expected fields {', '.join(sorted(expected_fields))}"
        )
    project_id = raw["id"]
    accession = raw["accession"]
    title = raw["title"]
    description = raw["description"]
    relevance = raw["relevance"]
    if not isinstance(project_id, str) or not project_id.isascii() or not project_id.isdecimal():
        raise SourceSnapshotError(
            f"Invalid BioProject context catalog {catalog_path} line {line_number}: "
            "project ID must be a numeric string"
        )
    for field_name, value, allow_empty in (
        ("accession", accession, False),
        ("title", title, False),
        ("description", description, True),
    ):
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            raise SourceSnapshotError(
                f"Invalid BioProject context catalog {catalog_path} line {line_number}: "
                f"{field_name} must be {'a string' if allow_empty else 'a non-empty string'}"
            )
    if (
        not isinstance(relevance, list)
        or any(not isinstance(flag, str) or flag not in _ORIGIN_RELEVANCE for flag in relevance)
        or len(relevance) != len(set(relevance))
    ):
        raise SourceSnapshotError(
            f"Invalid BioProject context catalog {catalog_path} line {line_number}: "
            "relevance must contain distinct Agricultural, Environmental, or Veterinary flags"
        )
    return ResolvedBioProjectContext(
        id=project_id,
        accession=accession,
        title=title,
        description=description,
        relevance=tuple(relevance),
    )


# --- Cache ---


class SQLiteCache(SQLiteKVCache):
    """SQLite-backed store keyed by canonical LLM request fingerprints."""

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS cache (
            hash_id TEXT PRIMARY KEY,
            category TEXT,
            display_term TEXT,
            ontology_link TEXT,
            term_path TEXT,
            reasoning TEXT,
            evidence_level TEXT NOT NULL DEFAULT 'none'
        )
    """

    def __init__(self, db_path: Path | str = DEFAULT_ISO_CACHE_DB) -> None:
        super().__init__(db_path)
        # In-place migration for caches written before evidence levels existed.
        # Their rows default to 'none' rather than being discarded.
        columns = {row[1] for row in self.cursor.execute("PRAGMA table_info(cache)").fetchall()}
        if "evidence_level" not in columns:
            self.cursor.execute(
                "ALTER TABLE cache ADD COLUMN evidence_level TEXT NOT NULL DEFAULT 'none'"
            )
            self.conn.commit()

    def get(self, request_fingerprint: str) -> StandardizedSource | None:
        self.cursor.execute(
            "SELECT category, display_term, ontology_link, term_path, reasoning, "
            "evidence_level "
            "FROM cache WHERE hash_id=?",
            (request_fingerprint,),
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
            evidence_level=IsolationEvidenceLevel(row[5]),
        )

    def set(
        self,
        request_fingerprint: str,
        record: StandardizedSource,
    ) -> None:
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO cache
                (hash_id, category, display_term, ontology_link, term_path, reasoning,
                 evidence_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_fingerprint,
                record.categories,
                record.display_terms,
                record.ontology_links,
                record.term_paths,
                json.dumps(record.reasoning),
                record.evidence_level.value,
            ),
        )
        self.conn.commit()


# --- Ontology graph ---


class OntologyManager:
    """Parse the ontology TSV into a tree with crosslinks and lookup indexes."""

    def __init__(self, ontology_tsv_path: Path | str):

        # parent_path -> list of full child_paths
        self.children_map: dict[str, list[str]] = defaultdict(list)

        # full_path -> {display_term, ontology_link}
        self.node_metadata: dict[str, dict[str, str]] = {}

        # term -> full_path
        self.exact_match_index: dict[str, str] = {}

        # upper-cased external ontology ID (UBERON:0001234) -> full_path
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
        """Populate the node metadata, the lookup indexes and the parent-child edges."""
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
        """Link terms across branches, by display name or external ID.

        A second pass, because a crosslink target may appear anywhere in the TSV
        and can only be resolved once every term is indexed. Unresolvable targets
        are logged and skipped rather than raising: the ontology stays usable.
        """
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


def _build_schema(
    valid_display_terms_normalized: set[str],
    request_mode: _IsolationRequestMode,
) -> type[BaseModel]:
    """Build a Pydantic model whose `terms` is validated against the ontology."""

    valid = valid_display_terms_normalized

    class IsolationClassificationBase(BaseModel):
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
            normalized_terms = [normalize_keyword(term) for term in v]
            invalid = [
                term
                for term, normalized in zip(v, normalized_terms, strict=True)
                if normalized not in valid
            ]
            if invalid:
                raise ValueError(
                    "Unknown terms: "
                    + ", ".join(repr(t) for t in invalid)
                    + ". Each term must be the exact display name of a node "
                    "in the ontology above (e.g. 'rectum', 'blood', "
                    "'hospital'). Do not include the colon-separated path."
                )
            return v

    permitted_evidence_levels = _EVIDENCE_LEVELS_BY_REQUEST_MODE[request_mode]
    evidence_description = (
        "Evidence provenance for the selected terms: 'sample' when only BioSample "
        "metadata supports them, 'project' when only BioProject context supports "
        "them, 'sample_and_project' when both support them, and 'none' only for an "
        "unspecified or unsupported result. Permitted values for this request: "
        + ", ".join(level.value for level in permitted_evidence_levels)
        + "."
    )
    evidence_literal = Literal.__getitem__(
        tuple(level.value for level in permitted_evidence_levels)
    )
    return create_model(
        "IsolationClassification",
        __base__=IsolationClassificationBase,
        evidence_level=(evidence_literal, Field(..., description=evidence_description)),
    )


def _resolve_evidence_level(
    *,
    direct_paths: set[str],
    llm_paths: set[str],
    claimed_level: IsolationEvidenceLevel,
) -> IsolationEvidenceLevel:
    """Reconcile model provenance with the specific terms in the final result."""
    specific_direct_paths = direct_paths - {_UNSPECIFIED_TERM}
    specific_llm_paths = llm_paths - {_UNSPECIFIED_TERM}
    if not specific_direct_paths and not specific_llm_paths:
        return IsolationEvidenceLevel.NONE
    if specific_direct_paths and not specific_llm_paths:
        return IsolationEvidenceLevel.SAMPLE
    if specific_direct_paths and claimed_level in {
        IsolationEvidenceLevel.PROJECT,
        IsolationEvidenceLevel.SAMPLE_AND_PROJECT,
    }:
        return IsolationEvidenceLevel.SAMPLE_AND_PROJECT
    if claimed_level is IsolationEvidenceLevel.NONE:
        raise ValueError("A specific isolation source must name supporting evidence")
    return claimed_level


class LLMClassifier:
    """Resolve one record's values to ontology terms.

    Calls the model only when deterministic matching falls short.
    """

    def __init__(
        self,
        config: dict,
        ontology_manager: OntologyManager,
        cache_manager: SQLiteCache,
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
        read_cache: bool = True,
    ) -> None:
        self.logger = result_logger or logger
        self.config = config
        self.ont = ontology_manager
        self.cache = cache_manager
        self.read_cache = read_cache
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
            self._response_schemas = {
                request_mode: _build_schema(valid_set, request_mode)
                for request_mode in _IsolationRequestMode
            }
            prompts = effective_isolation_prompts(self.config, self.ont)
            self.system_prompt = prompts.system
            self.user_template = prompts.user_template
            self.bioproject_system_prompt = prompts.bioproject_system
            self.bioproject_user_prompt = prompts.bioproject_user
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
        bioproject_contexts: tuple[ResolvedBioProjectContext, ...] = (),
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

        # A host alone is sample context: 'host = cattle' places the sample even
        # with no isolation-source attribute of its own.
        has_sample_context = bool(valid_vals) or bool(host.strip())
        if not has_sample_context and not bioproject_contexts:
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

        # LLM processing is skipped only when every value resolved on its own. Partial
        # coverage still goes to the model, which sees the unresolved values in
        # context rather than having them dropped.
        direct_covers_all = bool(valid_vals) and direct_match_count == len(valid_vals)
        evidence_level = IsolationEvidenceLevel.NONE

        if direct_covers_all:
            final_nodes |= direct_paths
            evidence_level = _resolve_evidence_level(
                direct_paths=direct_paths,
                llm_paths=set(),
                claimed_level=IsolationEvidenceLevel.NONE,
            )
            reasoning_history.append(
                {
                    "node": "direct_match",
                    "reasoning": "All values resolved manually.",
                    "selections": sorted(direct_paths),
                }
            )
        else:
            request_mode = (
                _IsolationRequestMode.COMBINED
                if has_sample_context and bioproject_contexts
                else _IsolationRequestMode.SAMPLE_ONLY
                if has_sample_context
                else _IsolationRequestMode.PROJECT_ONLY
            )
            response_schema = self._response_schemas[request_mode]
            permitted_evidence_levels = _EVIDENCE_LEVELS_BY_REQUEST_MODE[request_mode]
            metadata_block = self._format_metadata(valid_attrs, valid_vals, host)
            system_prompt = self.system_prompt
            bioproject_context = ""
            if bioproject_contexts:
                rendered_context = json.dumps(
                    [project.prompt_object() for project in bioproject_contexts],
                    ensure_ascii=False,
                    indent=2,
                )
                system_prompt += "\n" + self.bioproject_system_prompt
                bioproject_context = self.bioproject_user_prompt.format(
                    bioproject_context=rendered_context,
                )
            user_prompt = self.user_template.format(
                metadata=metadata_block,
                bioproject_context=bioproject_context,
            )
            request = CanonicalLLMRequest(
                model=self.model,
                messages=(
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ),
                parameters=ISOLATION_LLM_PARAMETERS,
                response_schema_id=(
                    f"{ISOLATION_RESPONSE_SCHEMA_ID}:"
                    f"{canonical_json_sha256(response_schema.model_json_schema())}"
                ),
            )
            if self.read_cache:
                cached_result = self.cache.get(request.fingerprint)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return replace(
                        _canonicalize_unspecified(cached_result),
                        request_fingerprint=request.fingerprint,
                    )

            if self.client is None:
                final_nodes |= direct_paths
                evidence_level = _resolve_evidence_level(
                    direct_paths=direct_paths,
                    llm_paths=set(),
                    claimed_level=IsolationEvidenceLevel.NONE,
                )
                reasoning_history.append(
                    {
                        "node": "classifier",
                        "reasoning": "LLM classification is disabled.",
                        "selections": [],
                    }
                )
            else:
                llm_paths: set[str] = set()
                try:
                    self.stats["llm_calls"] += 1
                    with observe_llm_call(
                        accession=accession,
                        target="isolation",
                        model=self.model,
                    ) as call:
                        resp = self.client.chat.completions.create(
                            model=request.model,
                            response_model=response_schema,
                            messages=list(request.messages),
                            **request.parameters,
                        )
                    evidence_level = IsolationEvidenceLevel(resp.evidence_level)
                    if evidence_level not in permitted_evidence_levels:
                        raise ValueError(
                            f"Invalid isolation evidence level {evidence_level.value!r} "
                            "for the available record context"
                        )
                    for term in resp.terms:
                        path = self.ont.display_term_to_path.get(normalize_keyword(term))
                        if path is not None:
                            llm_paths.add(path)
                    evidence_level = _resolve_evidence_level(
                        direct_paths=direct_paths,
                        llm_paths=llm_paths,
                        claimed_level=evidence_level,
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

        # "unspecified" beside a real term contradicts it, and the model does
        # occasionally select both. The specific terms win.
        if _UNSPECIFIED_TERM in final_nodes and len(final_nodes) > 1:
            final_nodes.remove(_UNSPECIFIED_TERM)

        if not final_nodes:
            final_record = StandardizedSource(
                categories="unspecified",
                display_terms="unspecified",
                ontology_links="NA",
                term_paths="",
                reasoning=reasoning_history,
                evidence_level=IsolationEvidenceLevel.NONE,
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
                evidence_level=evidence_level,
            )

        final_record = _canonicalize_unspecified(final_record)
        # `request` exists only on the branch that built one, which is exactly the
        # branch where direct matching fell short -- hence both guards.
        if not direct_covers_all and self.client is not None:
            self.cache.set(request.fingerprint, final_record)
        if not direct_covers_all:
            final_record = replace(final_record, request_fingerprint=request.fingerprint)
        return final_record


# --- Main class ---


class IsoStandardizer:
    """Standardize isolation source for one record, with BioProject study context."""

    def __init__(
        self,
        config_path: Path | str,
        extracted_metadata_path: Path | str,
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
        read_llm_cache: bool = True,
    ) -> None:
        self.logger = result_logger or logger
        self.config = load_config(config_path)
        self._projects_by_accession = load_bioproject_catalog(extracted_metadata_path)

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
                read_cache=read_llm_cache,
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
        """Classify one record, including any overflow values from the host pass.

        `overflow` carries values the host standardizer rejected as hosts but kept
        as isolation-source candidates. They join the classification input, while
        `sample_origins` stays the record's own pairs, so a host retry never sees
        back the values host already declined.
        """
        accession = str(record.get("accession", "") or "")
        sample_attributes = str(record.get("iso_attr_orig", "") or "")
        sample_values = str(record.get("iso_val_orig", "") or "")
        sample_origins = _isolation_origins(accession, sample_attributes, sample_values)
        attributes = sample_attributes
        values = sample_values
        if overflow is not None and overflow.value.strip():
            attributes = "||".join(part for part in (attributes, overflow.attribute) if part)
            values = "||".join(part for part in (values, overflow.value) if part)

        origins = _isolation_origins(accession, attributes, values)
        linked_project_ids = split_pipe_separated(str(record.get("bioproject_id", "") or ""))
        linked_project_accessions = split_pipe_separated(
            str(record.get("bioproject_accession", "") or "")
        )
        # Extraction emits an accession per ID it could resolve, so a shortfall in
        # the accession list means a linked project never made it into the catalog.
        has_unresolved_project_link = len(linked_project_ids) > len(linked_project_accessions)
        # Resolve linked accessions to catalog projects, dropping unresolved
        # ones, deduping, and ordering canonically by accession.
        resolved_projects = {
            linked_accession: project
            for linked_accession in linked_project_accessions
            if (project := self._projects_by_accession.get(linked_accession)) is not None
        }
        project_contexts = tuple(
            resolved_projects[accession] for accession in sorted(resolved_projects)
        )
        if not origins and not host_context.strip() and not project_contexts:
            diagnostics = [IsolationDiagnostic.NO_CANDIDATES]
            if has_unresolved_project_link:
                diagnostics.append(IsolationDiagnostic.UNRESOLVED_BIOPROJECT_LINK)
            return IsolationRejection(tuple(diagnostics))

        before = dict(self.pipeline.stats)
        standardized = self.pipeline.standardize_record(
            accession,
            "||".join(origin.attribute for origin in origins),
            "||".join(origin.value for origin in origins),
            host_context,
            project_contexts,
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
        if standardized.term_paths == _UNSPECIFIED_TERM:
            diagnostics.append(IsolationDiagnostic.UNSPECIFIED)
        if has_unresolved_project_link:
            diagnostics.append(IsolationDiagnostic.UNRESOLVED_BIOPROJECT_LINK)
        return IsolationOutcome(
            categories=standardized.categories,
            display_terms=standardized.display_terms,
            ontology_links=standardized.ontology_links,
            term_paths=standardized.term_paths,
            evidence_level=standardized.evidence_level,
            host_context=host_context,
            origins=origins,
            sample_origins=sample_origins,
            reasoning=tuple(
                IsolationReasoningStep.from_mapping(step) for step in standardized.reasoning
            ),
            diagnostics=tuple(diagnostics),
            exact_matches=exact_matches,
            cache_hits=cache_hits,
            llm_calls=llm_calls,
            request_fingerprint=standardized.request_fingerprint,
            resolved_bioproject_accessions=tuple(project.accession for project in project_contexts),
        )

    def close(self) -> None:
        try:
            self.pipeline.close()
        finally:
            self.cache.close()
