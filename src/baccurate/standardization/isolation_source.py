"""
Maps isolation source annotations from sample metadata to curated ontology
terms via a single-call LLM classifier.

See docs/isolation_source.md for the full pipeline description.
"""

import json
import logging
import os
import re
import string
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal

import instructor
import openai
from instructor.core import InstructorRetryException
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from baccurate.adapters.llm.client import LLMSettings, load_llm_client, load_llm_settings
from baccurate.adapters.llm.diagnostics import LLMFailureCategory, observe_llm_call
from baccurate.adapters.llm.request import CanonicalLLMRequest, canonical_json_sha256
from baccurate.adapters.policy_yaml import PolicyConfigurationError, load_policy_mapping
from baccurate.paths import (
    DEFAULT_ISOLATION_SOURCE_CACHE_DB,
    DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY,
)
from baccurate.standardization._attribute_value_text import normalize_keyword, split_pipe_separated
from baccurate.standardization._cache import SQLiteKVCache
from baccurate.standardization._isolation_source_ontology_renderer import (
    ordered_facets,
    render_ontology,
)
from baccurate.standardization.host import HostOverflowContext
from baccurate.standardization.isolation_source_ontology import (
    FacetCardinality,
    IsolationSourceOntology,
    IsolationSourceOntologyError,
    IsolationSourceTerm,
)
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair
from baccurate.standardization_target.specifications import TARGET_SPECS, StandardizationTarget

logger = logging.getLogger(__name__)

# None disables the LLM path here.
_LOAD_CONFIGURED_CLIENT = object()

# --- Constants ---

_IDENTIFIER_TOKEN = r"(?P<identifier>(?P<prefix>[A-Z][A-Z0-9._-]*):(?:(?P=prefix):)?\d+)"
ONTOLOGY_ID_PATTERN = re.compile(_IDENTIFIER_TOKEN, re.IGNORECASE)
_BRACKETED_IDENTIFIER_PATTERN = re.compile(
    rf"(?P<label>.+?)\s*\[\s*{_IDENTIFIER_TOKEN}\s*\]",
    re.IGNORECASE,
)
_PARENTHESIZED_IDENTIFIER_PATTERN = re.compile(
    rf"(?P<label>.+?)\s*\(\s*{_IDENTIFIER_TOKEN}\s*\)",
    re.IGNORECASE,
)
_SUFFIX_IDENTIFIER_PATTERN = re.compile(
    rf"(?P<label>.+?)\s+{_IDENTIFIER_TOKEN}",
    re.IGNORECASE,
)
_LEADING_IDENTIFIER_PATTERN = re.compile(
    rf"{_IDENTIFIER_TOKEN}\s+(?P<label>.+)",
    re.IGNORECASE,
)
_BRACKETED_BARE_IDENTIFIER_PATTERN = re.compile(
    rf"\[\s*{_IDENTIFIER_TOKEN}\s*\]",
    re.IGNORECASE,
)

ISOLATION_SOURCE_LLM_PARAMETERS: dict[str, object] = {"temperature": 0, "seed": 100}
ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE = instructor.Mode.JSON_SCHEMA
# The value is part of existing request fingerprints and cache keys.
ISOLATION_SOURCE_RESPONSE_SCHEMA_ID = "baccurate.isolation.classification.v4"

# These submitted values explicitly state that no isolation source is available. They
# are consumed without selecting an ontology term or invoking the classifier.
_NON_SOURCE_VALUES = frozenset({"no_host"})

_SOURCE_TYPE_FACET = "source_type"
_HOST_ASSOCIATED_TERM_ID = "BACC:0000001"
_ANIMAL_HOST_TERM_ID = "BACC:0000002"
_PLANT_HOST_TERM_ID = "BACC:0000003"
_ENVIRONMENTAL_TERM_ID = "BACC:0000004"
_FOOD_OR_FEED_TERM_ID = "BACC:0000007"
HOST_SOURCE_TYPE_BY_LINEAGE_ROOT = (
    (33208, _ANIMAL_HOST_TERM_ID),
    (33090, _PLANT_HOST_TERM_ID),
)
_SOURCE_TYPE_BY_IMPLYING_FACET = {
    "body_product": _ANIMAL_HOST_TERM_ID,
    "body_site": _ANIMAL_HOST_TERM_ID,
    "lesion": _ANIMAL_HOST_TERM_ID,
    "environmental_material": _ENVIRONMENTAL_TERM_ID,
    "facility": _ENVIRONMENTAL_TERM_ID,
    "sampled_object": _ENVIRONMENTAL_TERM_ID,
    "food_type": _FOOD_OR_FEED_TERM_ID,
}


def _validate_enrichment_vocabulary(ontology: IsolationSourceOntology) -> None:
    """Validate the fixed term identities required by deterministic enrichment."""
    required_ids = {
        _HOST_ASSOCIATED_TERM_ID,
        _ANIMAL_HOST_TERM_ID,
        _PLANT_HOST_TERM_ID,
        _ENVIRONMENTAL_TERM_ID,
        _FOOD_OR_FEED_TERM_ID,
    }
    for term_id in sorted(required_ids):
        term = ontology.terms.get(term_id)
        if term is None:
            raise ValueError(f"required enrichment term {term_id!r} is missing")
        if term.facet != _SOURCE_TYPE_FACET:
            raise ValueError(
                f"required enrichment term {term_id!r} must belong to facet {_SOURCE_TYPE_FACET!r}"
            )

    for root_id in (
        _HOST_ASSOCIATED_TERM_ID,
        _ENVIRONMENTAL_TERM_ID,
        _FOOD_OR_FEED_TERM_ID,
    ):
        if ontology.terms[root_id].parent_id is not None:
            raise ValueError(f"required broad source-kind term {root_id!r} must be a root term")

    for host_term_id in (_ANIMAL_HOST_TERM_ID, _PLANT_HOST_TERM_ID):
        ancestor_id = ontology.terms[host_term_id].parent_id
        while ancestor_id is not None and ancestor_id != _HOST_ASSOCIATED_TERM_ID:
            ancestor_id = ontology.terms[ancestor_id].parent_id
        if ancestor_id is None:
            raise ValueError(
                f"required enrichment term {host_term_id!r} must descend from "
                f"{_HOST_ASSOCIATED_TERM_ID!r}"
            )


# --- Prompts and fingerprints ---


@dataclass(frozen=True, slots=True)
class IsolationSourcePromptTemplates:
    """Validated isolation-source prompt templates before ontology rendering."""

    sample_system_template: str
    sample_user_template: str


@dataclass(frozen=True, slots=True)
class IsolationSourcePrompts:
    """Effective isolation-source prompt text used in canonical LLM requests."""

    system: str
    user_template: str


@dataclass(frozen=True, slots=True)
class IsolationSourceProvenance:
    """Versions and content identities for isolation-source results."""

    vocabulary_version: str
    vocabulary_fingerprint: str
    mapping_set_version: str
    mapping_set_fingerprint: str
    prompt_version: str
    prompt_configuration_fingerprint: str


@dataclass(frozen=True, slots=True)
class IsolationSourcePromptPolicy:
    """Validated isolation-source prompt, ontology, and cache policy."""

    schema_version: int
    prompt_version: str
    prompts: IsolationSourcePromptTemplates
    effective_prompts: IsolationSourcePrompts
    ontology_directory: Path
    ontology: IsolationSourceOntology
    cache_db_path: Path
    configured_ontology_directory: str | None
    configured_cache_db_path: str | None

    @classmethod
    def load(cls, path: Path | str) -> "IsolationSourcePromptPolicy":
        """Strictly load isolation-source prompt policy from one YAML file."""
        policy_path = Path(path)
        return _parse_isolation_source_prompt_policy(load_policy_mapping(policy_path), policy_path)

    def serialize(self) -> str:
        """Return deterministic canonical JSON without changing legacy identities."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "prompt_version": self.prompt_version,
                "system_prompt": self.prompts.sample_system_template,
                "user_prompt": self.prompts.sample_user_template,
                "ontology_directory": self.ontology_directory.as_posix(),
                "cache_db_path": self.cache_db_path.as_posix(),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def as_legacy_mapping(self) -> dict[str, object]:
        """Return the established mapping shape used by configuration fingerprints."""
        configuration: dict[str, object] = {
            "prompt_version": self.prompt_version,
            "system_prompt": self.prompts.sample_system_template,
            "user_prompt": self.prompts.sample_user_template,
        }
        if self.configured_ontology_directory is not None:
            configuration["ontology_directory"] = self.configured_ontology_directory
        if self.configured_cache_db_path is not None:
            configuration["cache_db_path"] = self.configured_cache_db_path
        return configuration

    @property
    def effective_prompt_configuration(self) -> dict[str, object]:
        """Return the response-affecting prompt and decoding configuration."""
        return {
            "prompt_version": self.prompt_version,
            "system_prompt": self.effective_prompts.system,
            "user_prompt_template": self.effective_prompts.user_template,
            "request_parameters": dict(ISOLATION_SOURCE_LLM_PARAMETERS),
            "structured_output_mode": ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE.value,
        }

    @property
    def prompt_configuration_fingerprint(self) -> str:
        """Identify the effective isolation-source prompt contract from its parsed content."""
        return canonical_json_sha256(self.effective_prompt_configuration)

    @property
    def provenance(self) -> IsolationSourceProvenance:
        """Return the reference and prompt identity that must appear in the run report."""
        return IsolationSourceProvenance(
            vocabulary_version=self.ontology.vocabulary_version,
            vocabulary_fingerprint=self.ontology.vocabulary_fingerprint,
            mapping_set_version=self.ontology.mapping_set.mapping_set_version,
            mapping_set_fingerprint=self.ontology.mapping_set_fingerprint,
            prompt_version=self.prompt_version,
            prompt_configuration_fingerprint=self.prompt_configuration_fingerprint,
        )


def _isolation_source_policy_error(
    policy_path: Path, key: str, message: str
) -> PolicyConfigurationError:
    return PolicyConfigurationError(f"{policy_path}: {key}: {message}")


def _require_isolation_source_string(
    config: Mapping[object, object],
    key: str,
    policy_path: Path,
) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _isolation_source_policy_error(policy_path, key, "must be a non-empty string")
    return value


def _validate_isolation_source_prompt(
    template: str,
    *,
    policy_path: Path,
    key: str,
    required_fields: tuple[str, ...],
    uses_format: bool = False,
) -> None:
    if uses_format:
        try:
            parsed_fields = [
                (field_name, format_spec, conversion)
                for _, field_name, format_spec, conversion in string.Formatter().parse(template)
                if field_name is not None
            ]
        except ValueError as error:
            raise _isolation_source_policy_error(
                policy_path, key, f"malformed format string: {error}"
            ) from error
        expected_fields = [(field, "", None) for field in required_fields]
        if len(parsed_fields) == len(expected_fields) and set(parsed_fields) == set(
            expected_fields
        ):
            return
        fields = [field_name for field_name, _, _ in parsed_fields]
    else:
        braced_values = re.findall(r"(?<!{){([^{}\n]*)}(?!})", template)
        fields = [value for value in braced_values if not value.lstrip().startswith(('"', "'"))]
    if sorted(fields) != sorted(required_fields):
        placeholders = ", ".join(f"{{{field}}}" for field in required_fields)
        raise _isolation_source_policy_error(
            policy_path,
            key,
            f"must contain exactly one of each required placeholder ({placeholders}) "
            "and no other placeholders",
        )


def _isolation_source_resource_selection(
    config: Mapping[object, object],
    key: str,
    default: Path,
    policy_path: Path,
) -> tuple[str | None, str]:
    """Return configured spelling and effective path text for one resource."""
    if key not in config:
        return None, str(default)
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise _isolation_source_policy_error(policy_path, key, "must be a non-empty string")
    return value, value


def _parse_isolation_source_prompt_policy(
    config: Mapping[object, object],
    policy_path: Path,
) -> IsolationSourcePromptPolicy:
    allowed = {
        "schema_version",
        "prompt_version",
        "system_prompt",
        "user_prompt",
        "ontology_directory",
        "cache_db_path",
    }
    unknown = set(config) - allowed
    if unknown:
        key = sorted(str(value) for value in unknown)[0]
        raise _isolation_source_policy_error(policy_path, f"top level.{key}", "unknown policy key")

    schema_version = config.get("schema_version")
    if type(schema_version) is not int:
        raise _isolation_source_policy_error(
            policy_path, "schema_version", "must be integer version 3"
        )
    if schema_version != 3:
        raise _isolation_source_policy_error(
            policy_path,
            "schema_version",
            f"unsupported schema version {schema_version}; expected version 3; "
            "migrate the isolation-source prompt policy, then retry",
        )

    prompt_version = _require_isolation_source_string(config, "prompt_version", policy_path)
    sample_system = _require_isolation_source_string(config, "system_prompt", policy_path)
    sample_user = _require_isolation_source_string(config, "user_prompt", policy_path)
    _validate_isolation_source_prompt(
        sample_system,
        policy_path=policy_path,
        key="system_prompt",
        required_fields=("ontology_tree",),
    )
    _validate_isolation_source_prompt(
        sample_user,
        policy_path=policy_path,
        key="user_prompt",
        required_fields=("metadata",),
        uses_format=True,
    )

    configured_ontology_value, ontology_value = _isolation_source_resource_selection(
        config,
        "ontology_directory",
        DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY,
        policy_path,
    )
    ontology_directory = Path(ontology_value)
    if not ontology_directory.is_dir() or not os.access(ontology_directory, os.R_OK):
        raise _isolation_source_policy_error(
            policy_path,
            "ontology_directory",
            f"must select a readable directory: {ontology_directory}",
        )
    try:
        ontology = IsolationSourceOntology.load(ontology_directory)
        _validate_enrichment_vocabulary(ontology)
    except (IsolationSourceOntologyError, PolicyConfigurationError, ValueError) as error:
        raise _isolation_source_policy_error(
            policy_path,
            "ontology_directory",
            f"must select a readable isolation-source ontology directory: {error}",
        ) from error

    configured_cache_value, cache_value = _isolation_source_resource_selection(
        config,
        "cache_db_path",
        DEFAULT_ISOLATION_SOURCE_CACHE_DB,
        policy_path,
    )
    cache_path = Path(cache_value)
    cache_parent = cache_path.parent
    if not cache_parent.is_dir():
        raise _isolation_source_policy_error(
            policy_path,
            "cache_db_path",
            f"parent directory does not exist: {cache_parent}",
        )
    writable_cache_target = cache_path if cache_path.exists() else cache_parent
    if (cache_path.exists() and not cache_path.is_file()) or not os.access(
        writable_cache_target, os.W_OK
    ):
        raise _isolation_source_policy_error(
            policy_path,
            "cache_db_path",
            "must select a writable database file",
        )

    prompts = IsolationSourcePromptTemplates(
        sample_system_template=sample_system,
        sample_user_template=sample_user,
    )
    return IsolationSourcePromptPolicy(
        schema_version=3,
        prompt_version=prompt_version,
        prompts=prompts,
        effective_prompts=IsolationSourcePrompts(
            system=sample_system.replace("{ontology_tree}", render_ontology(ontology)),
            user_template=sample_user,
        ),
        ontology_directory=ontology_directory,
        ontology=ontology,
        cache_db_path=cache_path,
        configured_ontology_directory=configured_ontology_value,
        configured_cache_db_path=configured_cache_value,
    )


# --- Data structures ---


class IsolationSourceEvidenceLevel(StrEnum):
    """Whether BioSample evidence supports an isolation-source result."""

    SAMPLE = "sample"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class SelectedTerm:
    """One resolved isolation-source ontology selection."""

    term_id: str
    facet: str
    label: str


@dataclass(frozen=True, slots=True)
class IsolationSourceOntologyGapDiagnostic:
    """One classifier label absent from its declared facet vocabulary."""

    accession: str
    facet: str
    label: str


@dataclass(frozen=True, slots=True)
class StandardizedIsolationSource:
    """Isolation-source classification for one extracted metadata record."""

    selected_terms: tuple[SelectedTerm, ...]
    reasoning: list[dict]
    evidence_level: IsolationSourceEvidenceLevel = IsolationSourceEvidenceLevel.NONE
    classifier_term_ids: frozenset[str] = frozenset()
    host_recovery_eligible: bool = False
    identifier_disagreement: bool = False
    vocabulary_disagreement: bool = False
    request_fingerprint: str | None = None
    ontology_gap_diagnostics: tuple[IsolationSourceOntologyGapDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class IsolationSourceClassifierAnswer:
    """Validated classifier fields before ontology enrichment."""

    facet_values: dict[str, str | tuple[str, ...] | None]
    reasoning: str
    evidence_level: IsolationSourceEvidenceLevel


class IsolationSourceDiagnostic(StrEnum):
    """The fixed set of isolation-source results used in build statistics."""

    NO_CLASSIFICATION_INPUT = "no_classification_input"
    EXACT_MATCH = "exact_match"
    CACHE_HIT = "cache_hit"
    LLM_CALL = "llm_call"
    CLASSIFICATION_FAILURE = "classification_failure"
    IDENTIFIER_DISAGREEMENT = "identifier_disagreement"
    CROSSLINK_DISAGREEMENT = "crosslink_disagreement"
    UNSPECIFIED = "unspecified"


class _IsolationSourceClassificationError(RuntimeError):
    """A classifier response that remains unusable after validation retries."""

    def __init__(
        self,
        message: str,
        *,
        identifier_disagreement: bool = False,
        ontology_gap_diagnostics: tuple[IsolationSourceOntologyGapDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.identifier_disagreement = identifier_disagreement
        self.ontology_gap_diagnostics = ontology_gap_diagnostics


def _parse_supporting_pairs(
    accession: str,
    attributes: str,
    values: str,
) -> tuple[SupportingAttributeValuePair, ...]:
    """Parse aligned selected attribute-value pairs while preserving their raw pairing."""
    attribute_parts = split_pipe_separated(attributes)
    value_parts = split_pipe_separated(values)
    if len(attribute_parts) != len(value_parts):
        raise ValueError(
            f"Malformed isolation-source selected attribute-value pairs for accession {accession}: "
            f"{len(attribute_parts)} attributes for {len(value_parts)} values"
        )
    return tuple(
        SupportingAttributeValuePair(attribute.strip(), value.strip())
        for attribute, value in zip(attribute_parts, value_parts, strict=True)
        if value.strip()
    )


@dataclass(frozen=True, slots=True)
class IsolationSourceReasoningStep:
    """One step in the classifier's reasoning trace.

    `selected_terms` groups the exact selections contributed by this stage by facet.
    """

    node: str
    reasoning: str
    selected_terms: dict[str, tuple[str, ...]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "IsolationSourceReasoningStep":
        """Type one raw step from the classifier or the cache."""
        return cls(
            node=str(value.get("node", "")),
            reasoning=str(value.get("reasoning", "")),
            selected_terms={
                str(facet): tuple(str(item) for item in labels)
                for facet, labels in dict(value.get("selected_terms", {})).items()
            },
        )

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON representation used by run artifacts."""
        result: dict[str, object] = {
            "node": self.node,
            "reasoning": self.reasoning,
            "selected_terms": {
                facet: list(labels) for facet, labels in self.selected_terms.items()
            },
        }
        return result


@dataclass(frozen=True, slots=True)
class IsolationSourceOutcome:
    """Typed isolation-source classification for one extracted metadata record."""

    selected_terms: tuple[SelectedTerm, ...]
    evidence_level: IsolationSourceEvidenceLevel
    supporting_pairs: tuple[SupportingAttributeValuePair, ...]
    host_recovery_pairs: tuple[SupportingAttributeValuePair, ...]
    reasoning: tuple[IsolationSourceReasoningStep, ...]
    diagnostics: tuple[IsolationSourceDiagnostic, ...]
    exact_matches: int
    cache_hits: int
    llm_calls: int
    host_recovery_eligible: bool = False
    request_fingerprint: str | None = None
    ontology_gap_diagnostics: tuple[IsolationSourceOntologyGapDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class IsolationSourceRejection:
    """An extracted metadata record with no isolation-source classification input."""

    diagnostics: tuple[IsolationSourceDiagnostic, ...]
    ontology_gap_diagnostics: tuple[IsolationSourceOntologyGapDiagnostic, ...] = ()


# --- Cache ---


class SQLiteCache(SQLiteKVCache):
    """SQLite-backed store keyed by canonical LLM request fingerprints."""

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS cache (
            hash_id TEXT PRIMARY KEY,
            answer TEXT NOT NULL,
            reasoning TEXT,
            evidence_level TEXT NOT NULL DEFAULT 'none'
        )
    """

    def __init__(self, db_path: Path | str = DEFAULT_ISOLATION_SOURCE_CACHE_DB) -> None:
        super().__init__(db_path)

    def get(self, request_fingerprint: str) -> IsolationSourceClassifierAnswer | None:
        self.cursor.execute(
            "SELECT answer, reasoning, evidence_level FROM cache WHERE hash_id=?",
            (request_fingerprint,),
        )
        cache_entry = self.cursor.fetchone()
        if cache_entry is None:
            return None

        answer = json.loads(cache_entry[0])

        return IsolationSourceClassifierAnswer(
            facet_values={
                facet: tuple(value) if isinstance(value, list) else value
                for facet, value in answer.items()
            },
            reasoning=cache_entry[1] or "",
            evidence_level=IsolationSourceEvidenceLevel(cache_entry[2]),
        )

    def set(
        self,
        request_fingerprint: str,
        answer: IsolationSourceClassifierAnswer,
    ) -> None:
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO cache
                (hash_id, answer, reasoning, evidence_level)
            VALUES (?, ?, ?, ?)
            """,
            (
                request_fingerprint,
                json.dumps(answer.facet_values),
                answer.reasoning,
                answer.evidence_level.value,
            ),
        )
        self.conn.commit()


# --- LLM classifier ---


def _build_schema(
    ontology: IsolationSourceOntology,
    observe_unknown_label: Callable[[str, str], None],
) -> type[BaseModel]:
    """Build the ordered, facet-specific classifier response model."""

    class IsolationSourceClassificationBase(BaseModel):
        model_config = ConfigDict(title="IsolationClassification")

        reasoning: str = Field(..., description="Brief reason for the chosen terms.")

    permitted_evidence_levels = tuple(IsolationSourceEvidenceLevel)
    evidence_description = (
        "Evidence provenance for the selected terms: 'sample' when BioSample metadata "
        "supports them, and 'none' only when no facet applies. Permitted values: "
        + ", ".join(level.value for level in permitted_evidence_levels)
        + "."
    )
    evidence_literal = Literal.__getitem__(
        tuple(level.value for level in permitted_evidence_levels)
    )
    facet_fields: dict[str, tuple[object, Field]] = {}
    validators: dict[str, object] = {}
    facet_keys: list[str] = []
    for facet in ordered_facets(ontology):
        facet_keys.append(facet.key)
        labels_to_terms = {
            term.label: term for term in ontology.terms.values() if term.facet == facet.key
        }
        permitted_labels = tuple(labels_to_terms)
        label_literal = Literal.__getitem__(permitted_labels)
        description = (
            f"{facet.meaning} {facet.classifier_guidance} "
            "The permitted labels form a closed vocabulary: "
            + ", ".join(permitted_labels)
            + ". Leave this facet empty when none fits."
        )
        if facet.cardinality is FacetCardinality.SINGLE:
            facet_fields[facet.key] = (
                label_literal | None,
                Field(..., description=description),
            )
        else:
            facet_fields[facet.key] = (
                list[label_literal],
                Field(..., description=description),
            )

        def validate_facet(
            value: str | list[str] | None,
            *,
            facet_key: str = facet.key,
            terms_by_label: Mapping[str, IsolationSourceTerm] = labels_to_terms,
        ) -> str | list[str] | None:
            labels = [] if value is None else [value] if isinstance(value, str) else value
            invalid = list(dict.fromkeys(label for label in labels if label not in terms_by_label))
            if invalid:
                for label in invalid:
                    observe_unknown_label(facet_key, label)
                raise ValueError(
                    f"Unknown {facet_key} labels: " + ", ".join(repr(label) for label in invalid)
                )
            selected_ids = {terms_by_label[label].term_id for label in labels}
            for label in labels:
                term = terms_by_label[label]
                parent_id = term.parent_id
                while parent_id is not None:
                    if parent_id in selected_ids:
                        raise ValueError(
                            f"{term.label!r} cannot be returned with its ancestor "
                            f"{ontology.terms[parent_id].label!r} in {facet_key}"
                        )
                    parent_id = ontology.terms[parent_id].parent_id
            return value

        validators[f"validate_{facet.key}"] = field_validator(facet.key, mode="before")(
            validate_facet
        )

    def validate_evidence_and_facets(model: BaseModel) -> BaseModel:
        has_selection = any(getattr(model, facet_key) not in (None, []) for facet_key in facet_keys)
        evidence_is_none = model.evidence_level == IsolationSourceEvidenceLevel.NONE.value
        if has_selection == evidence_is_none:
            raise ValueError("Every facet must be empty if and only if evidence_level is 'none'")
        return model

    validators["validate_evidence_and_facets"] = model_validator(mode="after")(
        validate_evidence_and_facets
    )
    return create_model(
        "IsolationSourceClassification",
        __base__=IsolationSourceClassificationBase,
        __validators__=validators,
        evidence_level=(evidence_literal, Field(..., description=evidence_description)),
        **facet_fields,
    )


def _resolve_evidence_level(
    *,
    direct_term_ids: set[str],
    classifier_term_ids: set[str],
) -> IsolationSourceEvidenceLevel:
    """Derive whether BioSample evidence supports the final result."""
    return (
        IsolationSourceEvidenceLevel.SAMPLE
        if direct_term_ids or classifier_term_ids
        else IsolationSourceEvidenceLevel.NONE
    )


class LLMClassifier:
    """Resolve one record's values to ontology terms.

    Calls the model only when deterministic matching falls short.
    """

    def __init__(
        self,
        policy: IsolationSourcePromptPolicy,
        ontology: IsolationSourceOntology,
        cache_manager: SQLiteCache,
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
        read_cache: bool = True,
    ) -> None:
        self.logger = result_logger or logger
        self.policy = policy
        self.ont = ontology
        self.cache = cache_manager
        self.read_cache = read_cache
        self.stats = {"cache_hits": 0, "exact_matches": 0, "llm_calls": 0}

        settings = llm_settings or load_llm_settings()
        if client is _LOAD_CONFIGURED_CLIENT:
            raw_client, env_model = load_llm_client(settings)
        else:
            raw_client = client
            env_model = settings.model
        self._raw_client = raw_client
        # This classifier resolves the endpoint and therefore owns the endpoint used
        # in run identity. Answers from different endpoints are not interchangeable,
        # so callers must read the endpoint here instead of restating it.
        self.server = settings.server
        try:
            self.model = env_model or ""
            self.client = (
                instructor.from_openai(
                    raw_client,
                    mode=ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE,
                )
                if raw_client
                else None
            )

            self._ordered_facets = ordered_facets(ontology)
            self._facet_order = {
                facet.key: index for index, facet in enumerate(self._ordered_facets)
            }
            self._terms_by_facet_and_label = {
                facet.key: {
                    term.label: term for term in ontology.terms.values() if term.facet == facet.key
                }
                for facet in self._ordered_facets
            }
            self._children_by_parent: dict[str | None, list[str]] = {}
            for term in ontology.terms.values():
                self._children_by_parent.setdefault(term.parent_id, []).append(term.term_id)
            for children in self._children_by_parent.values():
                children.sort()
            self._exact_match_index: dict[str, IsolationSourceTerm] = {}
            for term in ontology.terms.values():
                self._exact_match_index.setdefault(normalize_keyword(term.label), term)
                for synonym in term.synonyms:
                    self._exact_match_index.setdefault(normalize_keyword(synonym), term)
            self._identifier_match_index = {
                term_id.upper(): term for term_id, term in ontology.terms.items()
            }
            self._identifier_match_index.update(ontology.resolved_mapping_terms)
            self._declared_identifier_prefixes = {
                prefix.casefold() for prefix in ontology.mapping_set.curie_map
            }
            self._declared_identifier_prefixes.update(
                term_id.partition(":")[0].casefold() for term_id in ontology.terms
            )
            self._active_ontology_gap_observation: ContextVar[
                tuple[str, list[IsolationSourceOntologyGapDiagnostic]] | None
            ] = ContextVar("active_isolation_source_ontology_gap_observation", default=None)
            self._response_schema = _build_schema(ontology, self._record_unknown_label)
            prompts = policy.effective_prompts
            self.system_prompt = prompts.system
            self.user_template = prompts.user_template
        except BaseException:
            if raw_client is not None:
                raw_client.close()
            raise

    def close(self) -> None:
        if self._raw_client is not None:
            self._raw_client.close()

    def _record_unknown_label(self, facet: str, label: str) -> None:
        """One classifier label absent from its declared facet vocabulary."""
        observation = self._active_ontology_gap_observation.get()
        if observation is None:
            return
        accession, diagnostics = observation
        diagnostics.append(IsolationSourceOntologyGapDiagnostic(accession, facet, label))

    def _direct_match(self, value: str) -> tuple[IsolationSourceTerm | None, bool]:
        """Resolve an ontology term when the entire value is a label, identifier, or
        matching label-identifier pair."""
        stripped = value.strip()
        label_term = self._exact_match_index.get(normalize_keyword(stripped))
        if label_term is not None:
            return label_term, False

        for pattern in (ONTOLOGY_ID_PATTERN, _BRACKETED_BARE_IDENTIFIER_PATTERN):
            identifier_shape = pattern.fullmatch(stripped)
            if identifier_shape is not None:
                return self._term_for_identifier(identifier_shape.group("identifier")), False

        for pattern in (
            _BRACKETED_IDENTIFIER_PATTERN,
            _PARENTHESIZED_IDENTIFIER_PATTERN,
            _SUFFIX_IDENTIFIER_PATTERN,
            _LEADING_IDENTIFIER_PATTERN,
        ):
            paired_shape = pattern.fullmatch(stripped)
            if paired_shape is None:
                continue
            label_term = self._exact_match_index.get(
                normalize_keyword(paired_shape.group("label").strip())
            )
            identifier_term = self._term_for_identifier(paired_shape.group("identifier"))
            if label_term is not None and label_term == identifier_term:
                return identifier_term, False
            disagreement = (
                label_term is not None
                and identifier_term is not None
                and label_term != identifier_term
            )
            return None, disagreement
        return None, False

    def _term_for_identifier(self, identifier: str) -> IsolationSourceTerm | None:
        """Resolve an identifier only if the ontology artifact declares its prefix."""
        parts = identifier.split(":")
        if len(parts) == 3 and parts[0].casefold() == parts[1].casefold():
            identifier = f"{parts[0]}:{parts[2]}"
        prefix, _, _ = identifier.partition(":")
        if prefix.casefold() not in self._declared_identifier_prefixes:
            return None
        return self._identifier_match_index.get(identifier.upper())

    def _resolved_terms(self, term_ids: set[str]) -> tuple[SelectedTerm, ...]:
        ordered_ids: list[str] = []
        for facet in self._ordered_facets:
            facet_ids = {
                term_id for term_id in term_ids if self.ont.terms[term_id].facet == facet.key
            }

            def append_preorder(term_id: str, selected_facet_ids: set[str]) -> None:
                if term_id in selected_facet_ids:
                    ordered_ids.append(term_id)
                for child_id in self._children_by_parent.get(term_id, ()):
                    append_preorder(child_id, selected_facet_ids)

            for root_id in self._children_by_parent.get(None, ()):
                if self.ont.terms[root_id].facet == facet.key:
                    append_preorder(root_id, facet_ids)

        return tuple(
            SelectedTerm(
                term_id=term_id,
                facet=self.ont.terms[term_id].facet,
                label=self.ont.terms[term_id].label,
            )
            for term_id in ordered_ids
        )

    def _reasoning_selection(self, term_ids: set[str]) -> dict[str, list[str]]:
        selected: dict[str, list[str]] = {}
        for term in self._resolved_terms(term_ids):
            selected.setdefault(term.facet, []).append(term.label)
        return selected

    def _resolve_classifier_answer(
        self,
        answer: IsolationSourceClassifierAnswer,
    ) -> tuple[dict[str, list[str]], set[str]]:
        """Resolve a raw classifier answer against the active ontology."""
        classifier_selection: dict[str, list[str]] = {}
        classifier_term_ids: set[str] = set()
        for facet in self._ordered_facets:
            value = answer.facet_values[facet.key]
            labels = [] if value is None else [value] if isinstance(value, str) else list(value)
            if labels:
                classifier_selection[facet.key] = labels
            for label in labels:
                term = self._terms_by_facet_and_label[facet.key][label]
                classifier_term_ids.add(term.term_id)
        return classifier_selection, classifier_term_ids

    def _standardize_classifier_answer(
        self,
        answer: IsolationSourceClassifierAnswer,
        *,
        direct_term_ids: set[str],
        identifier_disagreement: bool,
        request_fingerprint: str,
        ontology_gap_diagnostics: tuple[IsolationSourceOntologyGapDiagnostic, ...] = (),
    ) -> StandardizedIsolationSource:
        """Combine one raw answer with current deterministic evidence and enrichment."""
        classifier_selection, classifier_term_ids = self._resolve_classifier_answer(answer)
        evidence_level = _resolve_evidence_level(
            direct_term_ids=direct_term_ids,
            classifier_term_ids=classifier_term_ids,
        )
        classification = StandardizedIsolationSource(
            selected_terms=self._resolved_terms(direct_term_ids | classifier_term_ids),
            reasoning=[
                {
                    "node": "classifier",
                    "reasoning": answer.reasoning,
                    "selected_terms": classifier_selection,
                }
            ],
            evidence_level=evidence_level,
            classifier_term_ids=frozenset(classifier_term_ids),
            identifier_disagreement=identifier_disagreement,
            request_fingerprint=request_fingerprint,
            ontology_gap_diagnostics=ontology_gap_diagnostics,
        )
        return self._enrich(classification)

    def _enrich(self, classification: StandardizedIsolationSource) -> StandardizedIsolationSource:
        """Add cross-linked terms, derive the source type, include facet ancestors, and order all
        terms canonically."""
        selected_ids = {term.term_id for term in classification.selected_terms}
        original_source_ids = {
            term_id
            for term_id in selected_ids
            if self.ont.terms[term_id].facet == _SOURCE_TYPE_FACET
        }
        classifier_source_ids = {
            term_id
            for term_id in classification.classifier_term_ids
            if self.ont.terms[term_id].facet == _SOURCE_TYPE_FACET
        }
        reasoning = list(classification.reasoning)
        vocabulary_disagreement = False

        # A source term implies all of its crosslink targets. Collect them before changing the
        # selection so input order and ontology row order cannot affect the result.
        crosslink_target_ids = {
            target_id
            for source_id in selected_ids
            for target_id in self.ont.terms[source_id].crosslink_target_ids
        }
        source_crosslink_ids = {
            term_id
            for term_id in crosslink_target_ids
            if self.ont.terms[term_id].facet == _SOURCE_TYPE_FACET
        }
        if source_crosslink_ids:
            vocabulary_disagreement |= bool(
                classifier_source_ids and classifier_source_ids != source_crosslink_ids
            )
            selected_ids -= original_source_ids
        crosslink_additions = crosslink_target_ids - selected_ids
        selected_ids |= crosslink_target_ids
        if crosslink_additions:
            reasoning.append(
                {
                    "node": "crosslink",
                    "reasoning": "Vocabulary crosslinks assigned related terms.",
                    "selected_terms": self._reasoning_selection(crosslink_additions),
                }
            )

        source_candidates = set(source_crosslink_ids)
        for facet_key, source_term_id in _SOURCE_TYPE_BY_IMPLYING_FACET.items():
            if any(self.ont.terms[term_id].facet == facet_key for term_id in selected_ids):
                source_candidates.add(source_term_id)

        if source_candidates:

            def source_precedence(term_id: str) -> tuple[int, int, str]:
                ancestor_id: str | None = term_id
                while ancestor_id is not None:
                    if ancestor_id == _HOST_ASSOCIATED_TERM_ID:
                        return (0, int(term_id not in source_crosslink_ids), term_id)
                    ancestor_id = self.ont.terms[ancestor_id].parent_id
                return (
                    {
                        _ENVIRONMENTAL_TERM_ID: 1,
                        _FOOD_OR_FEED_TERM_ID: 2,
                    }.get(term_id, 3),
                    int(term_id not in source_crosslink_ids),
                    term_id,
                )

            derived_source_id = min(source_candidates, key=source_precedence)
            current_source_ids = {
                term_id
                for term_id in selected_ids
                if self.ont.terms[term_id].facet == _SOURCE_TYPE_FACET
            }
            vocabulary_disagreement |= bool(
                classifier_source_ids and derived_source_id not in classifier_source_ids
            )
            selected_ids -= current_source_ids
            selected_ids.add(derived_source_id)
            if current_source_ids != {derived_source_id}:
                reasoning.append(
                    {
                        "node": "source_type_derivation",
                        "reasoning": "Filled facets determined the broad source kind.",
                        "selected_terms": self._reasoning_selection({derived_source_id}),
                    }
                )

        ancestor_additions: set[str] = set()
        for term_id in tuple(selected_ids):
            parent_id = self.ont.terms[term_id].parent_id
            while parent_id is not None:
                if parent_id not in selected_ids:
                    ancestor_additions.add(parent_id)
                parent_id = self.ont.terms[parent_id].parent_id
        selected_ids |= ancestor_additions
        if ancestor_additions:
            reasoning.append(
                {
                    "node": "ancestor_expansion",
                    "reasoning": "Selected terms were expanded to their facet ancestors.",
                    "selected_terms": self._reasoning_selection(ancestor_additions),
                }
            )

        return replace(
            classification,
            selected_terms=self._resolved_terms(selected_ids),
            reasoning=reasoning,
            host_recovery_eligible=any(
                self.ont.terms[term_id].enables_host_recovery for term_id in selected_ids
            ),
            vocabulary_disagreement=vocabulary_disagreement,
        )

    @staticmethod
    def _format_metadata(attrs: list[str], vals: list[str]) -> str:
        return "Metadata:\n" + "\n".join(
            f"{attribute} = {value}" for attribute, value in zip(attrs, vals, strict=True)
        )

    def standardize_record(
        self,
        accession: str,
        attr_name: str,
        value: str,
    ) -> StandardizedIsolationSource:
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
            return StandardizedIsolationSource(
                selected_terms=(),
                reasoning=[
                    {
                        "node": "classifier",
                        "reasoning": "No non-empty selected values were provided.",
                        "selected_terms": {},
                    }
                ],
            )

        # Direct-match pass over each (attr, val) pair before calling the LLM.
        direct_term_ids: set[str] = set()
        consumed_value_count = 0
        identifier_disagreement = False
        for v in valid_vals:
            if normalize_keyword(v) in _NON_SOURCE_VALUES:
                consumed_value_count += 1
                continue
            term, value_disagreement = self._direct_match(v)
            identifier_disagreement |= value_disagreement
            if term is not None:
                direct_term_ids.add(term.term_id)
                consumed_value_count += 1
                self.stats["exact_matches"] += 1

        selected_term_ids: set[str] = set()
        classifier_term_ids: set[str] = set()
        reasoning_history: list[dict] = []

        # LLM processing is skipped only when every value resolved on its own. Partial
        # coverage still goes to the model, which sees the unresolved values in
        # context rather than having them dropped.
        direct_covers_all = bool(valid_vals) and consumed_value_count == len(valid_vals)
        evidence_level = IsolationSourceEvidenceLevel.NONE

        if direct_covers_all:
            selected_term_ids |= direct_term_ids
            evidence_level = _resolve_evidence_level(
                direct_term_ids=direct_term_ids,
                classifier_term_ids=set(),
            )
            reasoning_history.append(
                {
                    "node": "direct_match",
                    "reasoning": "All values resolved manually.",
                    "selected_terms": self._reasoning_selection(direct_term_ids),
                }
            )
        else:
            ontology_gap_diagnostics: list[IsolationSourceOntologyGapDiagnostic] = []
            response_schema = self._response_schema
            user_prompt = self.user_template.format(
                metadata=self._format_metadata(valid_attrs, valid_vals),
            )
            request = CanonicalLLMRequest(
                model=self.model,
                messages=(
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ),
                parameters=ISOLATION_SOURCE_LLM_PARAMETERS,
                response_schema_id=(
                    f"{ISOLATION_SOURCE_RESPONSE_SCHEMA_ID}:"
                    f"{ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE.value}:"
                    f"{canonical_json_sha256(response_schema.model_json_schema())}"
                ),
            )
            if self.read_cache:
                cached_answer = self.cache.get(request.fingerprint)
                if cached_answer:
                    self.stats["cache_hits"] += 1
                    return self._standardize_classifier_answer(
                        cached_answer,
                        direct_term_ids=direct_term_ids,
                        identifier_disagreement=identifier_disagreement,
                        request_fingerprint=request.fingerprint,
                    )

            if self.client is None:
                selected_term_ids |= direct_term_ids
                evidence_level = _resolve_evidence_level(
                    direct_term_ids=direct_term_ids,
                    classifier_term_ids=set(),
                )
                reasoning_history.append(
                    {
                        "node": "classifier",
                        "reasoning": "LLM classification is disabled.",
                        "selected_terms": {},
                    }
                )
            else:
                try:
                    self.stats["llm_calls"] += 1
                    with observe_llm_call(
                        accession=accession,
                        target=TARGET_SPECS[StandardizationTarget.ISOLATION_SOURCE].published_key,
                        model=self.model,
                    ) as call:
                        observation_token = self._active_ontology_gap_observation.set(
                            (accession, ontology_gap_diagnostics)
                        )
                        try:
                            resp = self.client.chat.completions.create(
                                model=request.model,
                                response_model=response_schema,
                                messages=list(request.messages),
                                **request.parameters,
                                max_retries=3,
                            )
                        finally:
                            self._active_ontology_gap_observation.reset(observation_token)
                    classifier_evidence_level = IsolationSourceEvidenceLevel(resp.evidence_level)
                    facet_values: dict[str, str | tuple[str, ...] | None] = {}
                    for facet in self._ordered_facets:
                        value = getattr(resp, facet.key)
                        facet_values[facet.key] = tuple(value) if isinstance(value, list) else value
                    classifier_answer = IsolationSourceClassifierAnswer(
                        facet_values=facet_values,
                        reasoning=resp.reasoning,
                        evidence_level=classifier_evidence_level,
                    )
                    call.accepted()
                except InstructorRetryException as e:
                    call.validation_retries_exhausted()
                    raise _IsolationSourceClassificationError(
                        f"Isolation-source LLM failed for accession {accession}",
                        identifier_disagreement=identifier_disagreement,
                        ontology_gap_diagnostics=tuple(ontology_gap_diagnostics),
                    ) from e
                except Exception as e:
                    if not isinstance(e, openai.APIError):
                        call.failed(LLMFailureCategory.INVALID_MODEL_RESPONSE)
                    raise RuntimeError(
                        f"Isolation-source LLM failed for accession {accession}"
                    ) from e

                self.cache.set(request.fingerprint, classifier_answer)
                return self._standardize_classifier_answer(
                    classifier_answer,
                    direct_term_ids=direct_term_ids,
                    identifier_disagreement=identifier_disagreement,
                    request_fingerprint=request.fingerprint,
                    ontology_gap_diagnostics=tuple(ontology_gap_diagnostics),
                )

        classification = StandardizedIsolationSource(
            selected_terms=self._resolved_terms(selected_term_ids),
            reasoning=reasoning_history,
            evidence_level=evidence_level,
            classifier_term_ids=frozenset(classifier_term_ids),
            identifier_disagreement=identifier_disagreement,
        )
        if not direct_covers_all:
            classification = replace(classification, request_fingerprint=request.fingerprint)
        return self._enrich(classification)


# --- Main class ---


class IsolationSourceStandardizer:
    """Standardize one extracted metadata record from BioSample evidence."""

    def __init__(
        self,
        policy: IsolationSourcePromptPolicy,
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
        read_llm_cache: bool = True,
    ) -> None:
        self.logger = result_logger or logger
        self.policy = policy

        self.cache = SQLiteCache(policy.cache_db_path)
        try:
            self.ontology = policy.ontology
            self.pipeline = LLMClassifier(
                policy,
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
        self.logger.info("IsolationSourceStandardizer initialised (LLMClassifier).")

    def standardize(
        self,
        extracted_record: Mapping[str, str],
        *,
        overflow: HostOverflowContext | None = None,
    ) -> IsolationSourceOutcome | IsolationSourceRejection:
        """Classify one extracted metadata record, including host overflow values.

        `overflow` contains values the host standardizer rejected as hosts but retained
        for isolation-source standardization. The classifier includes them in its input.
        `host_recovery_pairs` contains only the extracted metadata record's own pairs,
        preventing the recovery pass from reconsidering values the host standardizer
        already rejected.
        """
        accession = str(extracted_record.get("accession", "") or "")
        sample_attributes = str(extracted_record.get("iso_attr_orig", "") or "")
        sample_values = str(extracted_record.get("iso_val_orig", "") or "")
        host_recovery_pairs = _parse_supporting_pairs(accession, sample_attributes, sample_values)
        attributes = sample_attributes
        values = sample_values
        if overflow is not None and overflow.value.strip():
            attributes = "||".join(part for part in (attributes, overflow.attribute) if part)
            values = "||".join(part for part in (values, overflow.value) if part)

        supporting_pairs = _parse_supporting_pairs(accession, attributes, values)
        if not supporting_pairs:
            return IsolationSourceRejection((IsolationSourceDiagnostic.NO_CLASSIFICATION_INPUT,))

        before = dict(self.pipeline.stats)
        try:
            standardized = self.pipeline.standardize_record(
                accession,
                "||".join(pair.attribute for pair in supporting_pairs),
                "||".join(pair.value for pair in supporting_pairs),
            )
        except _IsolationSourceClassificationError as error:
            diagnostics = [IsolationSourceDiagnostic.CLASSIFICATION_FAILURE]
            if error.identifier_disagreement:
                diagnostics.append(IsolationSourceDiagnostic.IDENTIFIER_DISAGREEMENT)
            return IsolationSourceRejection(
                tuple(diagnostics),
                ontology_gap_diagnostics=error.ontology_gap_diagnostics,
            )
        exact_matches = self.pipeline.stats["exact_matches"] - before["exact_matches"]
        cache_hits = self.pipeline.stats["cache_hits"] - before["cache_hits"]
        llm_calls = self.pipeline.stats["llm_calls"] - before["llm_calls"]
        diagnostics = []
        if exact_matches:
            diagnostics.append(IsolationSourceDiagnostic.EXACT_MATCH)
        if cache_hits:
            diagnostics.append(IsolationSourceDiagnostic.CACHE_HIT)
        if llm_calls:
            diagnostics.append(IsolationSourceDiagnostic.LLM_CALL)
        if standardized.vocabulary_disagreement:
            diagnostics.append(IsolationSourceDiagnostic.CROSSLINK_DISAGREEMENT)
        if standardized.identifier_disagreement:
            diagnostics.append(IsolationSourceDiagnostic.IDENTIFIER_DISAGREEMENT)
        if not standardized.selected_terms:
            diagnostics.append(IsolationSourceDiagnostic.UNSPECIFIED)
        return IsolationSourceOutcome(
            selected_terms=standardized.selected_terms,
            evidence_level=standardized.evidence_level,
            supporting_pairs=supporting_pairs,
            host_recovery_pairs=host_recovery_pairs,
            reasoning=tuple(
                IsolationSourceReasoningStep.from_mapping(step) for step in standardized.reasoning
            ),
            diagnostics=tuple(diagnostics),
            exact_matches=exact_matches,
            cache_hits=cache_hits,
            llm_calls=llm_calls,
            host_recovery_eligible=standardized.host_recovery_eligible,
            request_fingerprint=standardized.request_fingerprint,
            ontology_gap_diagnostics=standardized.ontology_gap_diagnostics,
        )

    def refine_source_type_from_host_lineage(
        self,
        outcome: IsolationSourceOutcome,
        *,
        host_taxid: int,
        lineage_root_taxid: int,
        source_type_term_id: str,
    ) -> IsolationSourceOutcome:
        """Set the source type from host lineage unless the current type is non-host."""
        selected_ids = {term.term_id for term in outcome.selected_terms}
        source_type_ids = {
            term_id
            for term_id in selected_ids
            if self.ontology.terms[term_id].facet == _SOURCE_TYPE_FACET
        }

        def is_host_associated(term_id: str) -> bool:
            current_term_id: str | None = term_id
            while current_term_id is not None:
                if current_term_id == _HOST_ASSOCIATED_TERM_ID:
                    return True
                current_term_id = self.ontology.terms[current_term_id].parent_id
            return False

        if any(not is_host_associated(term_id) for term_id in source_type_ids):
            return outcome

        previous_labels = [self.ontology.terms[term_id].label for term_id in source_type_ids]
        selected_ids -= source_type_ids
        selected_ids.add(source_type_term_id)
        parent_id = self.ontology.terms[source_type_term_id].parent_id
        while parent_id is not None:
            selected_ids.add(parent_id)
            parent_id = self.ontology.terms[parent_id].parent_id

        derived_label = self.ontology.terms[source_type_term_id].label
        specific_previous_labels = [
            label for label in previous_labels if label != "host-associated"
        ]
        disagreement = bool(
            specific_previous_labels and derived_label not in specific_previous_labels
        )
        reasoning = (
            f"Standardized host taxid {host_taxid} descends from NCBI lineage root "
            f"{lineage_root_taxid}; selected {derived_label!r}."
        )
        if disagreement:
            reasoning += f" Replaced conflicting source-type selection {previous_labels!r}."
        return replace(
            outcome,
            selected_terms=self.pipeline._resolved_terms(selected_ids),
            reasoning=(
                *outcome.reasoning,
                IsolationSourceReasoningStep(
                    node="host_lineage_derivation",
                    reasoning=reasoning,
                    selected_terms={_SOURCE_TYPE_FACET: (derived_label,)},
                ),
            ),
            diagnostics=tuple(
                diagnostic
                for diagnostic in outcome.diagnostics
                if diagnostic is not IsolationSourceDiagnostic.UNSPECIFIED
            ),
            evidence_level=(
                IsolationSourceEvidenceLevel.SAMPLE
                if outcome.evidence_level is IsolationSourceEvidenceLevel.NONE
                else outcome.evidence_level
            ),
            host_recovery_eligible=True,
        )

    def close(self) -> None:
        try:
            self.pipeline.close()
        finally:
            self.cache.close()
