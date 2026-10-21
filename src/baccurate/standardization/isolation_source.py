"""
Maps isolation source annotations from sample metadata to curated ontology
terms via a single-call LLM classifier.

See docs/isolation_source.md for the full pipeline description.
"""

import json
import logging
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal

import instructor
import openai
from instructor.core import InstructorRetryException
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from baccurate.adapters.llm.client import LLMSettings, load_llm_client, load_llm_settings
from baccurate.adapters.llm.diagnostics import LLMFailureCategory, observe_llm_call
from baccurate.adapters.llm.request import CanonicalLLMRequest, canonical_json_sha256
from baccurate.paths import (
    DEFAULT_ISOLATION_SOURCE_CACHE_DB,
)
from baccurate.standardization._attribute_value_text import normalize_keyword, split_pipe_separated
from baccurate.standardization.host import HostOverflowContext
from baccurate.standardization.isolation_source_config import (
    ISOLATION_SOURCE_LLM_PARAMETERS,
    ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE,
    FacetConfiguration,
    IsolationSourceConfig,
)
from baccurate.standardization.isolation_source_ontology import (
    ANIMAL_HOST_TERM_ID,
    FOOD_OR_FEED_TERM_ID,
    HOST_ASSOCIATED_TERM_ID,
    HOST_DISEASE_FACET,
    HUMAN_HOST_TERM_ID,
    LABORATORY_TERM_ID,
    PLANT_HOST_TERM_ID,
    SOURCE_TYPE_FACET,
    IsolationSourceOntology,
    IsolationSourceTerm,
    _ancestor_ids,
)
from baccurate.standardization.supporting_attribute_value_pair import SupportingAttributeValuePair
from baccurate.targets import TARGET_SPECS, StandardizationTarget

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

# The value is part of existing request fingerprints and cache keys.
ISOLATION_SOURCE_RESPONSE_SCHEMA_ID = "baccurate.isolation.classification.v7"

HOST_SOURCE_TYPE_BY_LINEAGE_ROOT = (
    (9606, HUMAN_HOST_TERM_ID),
    (33208, ANIMAL_HOST_TERM_ID),
    (33090, PLANT_HOST_TERM_ID),
)
FOOD_EXCLUDED_FACETS = frozenset({"body_product", "body_site", HOST_DISEASE_FACET})
SAMPLED_FACETS = frozenset(
    {
        "body_product",
        "body_site",
        "lesion",
        "environmental_material",
        "sampled_object",
        "food_type",
    }
)


def _is_host_source_type(ontology: IsolationSourceOntology, term_id: str) -> bool:
    return HOST_ASSOCIATED_TERM_ID in (term_id, *_ancestor_ids(ontology.terms, term_id))


def _strain_key(text: str) -> str:
    return text.casefold().replace("\N{GREEK SMALL LETTER ALPHA}", "alpha")


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
    request_fingerprint: str | None = None
    ontology_gap_diagnostics: tuple[IsolationSourceOntologyGapDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class IsolationSourceClassifierAnswer:
    """Validated classifier fields before term derivation."""

    facet_values: dict[str, str | tuple[str, ...] | None]
    reasoning: str


class IsolationSourceDiagnostic(StrEnum):
    """The fixed set of isolation-source results used in build statistics."""

    NO_CLASSIFICATION_INPUT = "no_classification_input"
    EXACT_MATCH = "exact_match"
    CACHE_HIT = "cache_hit"
    LLM_CALL = "llm_call"
    CLASSIFICATION_FAILURE = "classification_failure"
    IDENTIFIER_DISAGREEMENT = "identifier_disagreement"
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


_ALTERNATIVES_PATTERN = re.compile(r"\bor\b|/")
_FIXED_PHRASES_WITHOUT_ALTERNATIVES = (
    "veterinary facility or diagnostic laboratory",
    "whole or parts",
    "natural or artificial",
    "with or without",
    "comminuted or otherwise",
    "import/export",
    "vtec/stec",
    "stec/vtec",
)


def _names_alternatives(value: str) -> bool:
    """Return whether a value joins alternatives with "or" or "/".

    Picklist and ontology phrases that contain "or" or "/" do not count.
    """
    text = value.casefold()
    for phrase in _FIXED_PHRASES_WITHOUT_ALTERNATIVES:
        text = text.replace(phrase, "")
    return bool(_ALTERNATIVES_PATTERN.search(text))


# --- Cache ---

_SAMPLE_DIGITS_PATTERN = re.compile(r"(?<![:\d])\d+")


def _strip_digits(value: str) -> str:
    """
    Normalize a value so that samples differing only in digits share a cache key
    (like "patient 1" and "patient 2").

    Skips digits after a colon, to avoid masking ontology patterns (ENVO:00002007)"""
    return _SAMPLE_DIGITS_PATTERN.sub("<NUM>", " ".join(value.lower().split()))


class SQLiteCache:
    """SQLite-backed store keyed by canonical LLM request fingerprints."""

    def __init__(self, db_path: Path | str = DEFAULT_ISOLATION_SOURCE_CACHE_DB) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                hash_id TEXT PRIMARY KEY,
                answer TEXT NOT NULL,
                reasoning TEXT
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get(self, request_fingerprint: str) -> IsolationSourceClassifierAnswer | None:
        self.cursor.execute(
            "SELECT answer, reasoning FROM cache WHERE hash_id=?",
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
        )

    def set(
        self,
        request_fingerprint: str,
        answer: IsolationSourceClassifierAnswer,
    ) -> None:
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO cache
                (hash_id, answer, reasoning)
            VALUES (?, ?, ?)
            """,
            (
                request_fingerprint,
                json.dumps(answer.facet_values),
                answer.reasoning,
            ),
        )
        self.conn.commit()


# --- LLM classifier ---


def _build_schema(
    ontology: IsolationSourceOntology,
    facets: tuple[FacetConfiguration, ...],
    observe_unknown_label: Callable[[str, str], None],
) -> type[BaseModel]:
    """Build the ordered, facet-specific classifier response model.

    The field order is the generation order of the constrained decoder.
    """

    class IsolationSourceClassificationBase(BaseModel):
        model_config = ConfigDict(title="IsolationClassification")

    facet_fields: dict[str, tuple[object, Field]] = {}
    validators: dict[str, object] = {}
    for facet in facets:
        labels_to_terms = {
            term.label: term for term in ontology.terms.values() if term.facet == facet.key
        }
        permitted_labels = tuple(labels_to_terms)
        label_literal = Literal.__getitem__(permitted_labels)
        description = (
            f"{ontology.facets[facet.key].meaning} {facet.classifier_guidance} "
            "The permitted labels form a closed vocabulary: "
            + ", ".join(permitted_labels)
            + ". Leave this facet empty when none fits."
        )
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
            if not isinstance(value, list):
                return value
            implied_ids: set[str] = set()
            for label in labels:
                parent_id = terms_by_label[label].parent_id
                while parent_id is not None:
                    implied_ids.add(parent_id)
                    parent_id = ontology.terms[parent_id].parent_id
            return [label for label in labels if terms_by_label[label].term_id not in implied_ids]

        validators[f"validate_{facet.key}"] = field_validator(facet.key, mode="before")(
            validate_facet
        )

    facet_fields["reasoning"] = (
        str,
        Field(
            ...,
            max_length=240,
            description="Brief evidence statement for the selected terms.",
        ),
    )

    return create_model(
        "IsolationSourceClassification",
        __base__=IsolationSourceClassificationBase,
        __validators__=validators,
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
        config: IsolationSourceConfig,
        ontology: IsolationSourceOntology,
        cache_manager: SQLiteCache,
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.config = config
        self.ont = ontology
        self.cache = cache_manager
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

            self._ordered_facets = config.facets
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
            self._exact_match_index = {
                key: (ontology.terms[term_id],) for key, term_id in ontology.match_keys.items()
            }
            self._exact_match_index.update(
                (normalize_keyword(value), tuple(ontology.terms[term_id] for term_id in term_ids))
                for value, term_ids in config.direct_match_values.items()
            )
            self._identifier_match_index = {
                term_id.upper(): term for term_id, term in ontology.terms.items()
            }
            self._identifier_match_index.update(ontology.resolved_mapping_terms)
            self._obsolete_replacements = {
                obsolete.term_id.upper(): tuple(
                    ontology.terms[replacement_id] for replacement_id in obsolete.replaced_by
                )
                for obsolete in ontology.obsolete_terms.values()
            }
            self._declared_identifier_prefixes = {
                prefix.casefold() for prefix in ontology.mapping_set.curie_map
            }
            self._declared_identifier_prefixes.update(
                term_id.partition(":")[0].casefold() for term_id in ontology.terms
            )
            self._active_ontology_gap_observation: ContextVar[
                tuple[str, list[IsolationSourceOntologyGapDiagnostic]] | None
            ] = ContextVar("active_isolation_source_ontology_gap_observation", default=None)
            self._response_schema = _build_schema(
                ontology, config.facets, self._record_unknown_label
            )
            prompts = config.effective_prompts
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

    def _direct_match(self, value: str) -> tuple[tuple[IsolationSourceTerm, ...], bool]:
        """Resolve ontology terms when the entire value is a label, identifier, or
        matching label-identifier pair. A direct-match value resolves to each of its terms, and a
        whole obsolete identifier to its replacements."""
        stripped = value.strip()
        label_terms = self._exact_match_index.get(normalize_keyword(stripped))
        if label_terms is not None:
            return label_terms, False

        for pattern in (ONTOLOGY_ID_PATTERN, _BRACKETED_BARE_IDENTIFIER_PATTERN):
            identifier_shape = pattern.fullmatch(stripped)
            if identifier_shape is not None:
                identifier = self._declared_identifier(identifier_shape.group("identifier"))
                if identifier in self._obsolete_replacements:
                    return self._obsolete_replacements[identifier], False
                term = self._identifier_match_index.get(identifier)
                return ((term,) if term is not None else ()), False

        for pattern in (
            _BRACKETED_IDENTIFIER_PATTERN,
            _PARENTHESIZED_IDENTIFIER_PATTERN,
            _SUFFIX_IDENTIFIER_PATTERN,
            _LEADING_IDENTIFIER_PATTERN,
        ):
            paired_shape = pattern.fullmatch(stripped)
            if paired_shape is None:
                continue
            label_terms = self._exact_match_index.get(
                normalize_keyword(paired_shape.group("label").strip())
            )
            identifier_term = self._term_for_identifier(paired_shape.group("identifier"))
            if label_terms is not None and label_terms == (identifier_term,):
                return label_terms, False
            disagreement = label_terms is not None and identifier_term is not None
            return (), disagreement
        return (), False

    def _declared_identifier(self, identifier: str) -> str | None:
        """Return the upper-case identifier if the ontology artifact declares its prefix."""
        parts = identifier.split(":")
        if len(parts) == 3 and parts[0].casefold() == parts[1].casefold():
            identifier = f"{parts[0]}:{parts[2]}"
        prefix, _, _ = identifier.partition(":")
        if prefix.casefold() not in self._declared_identifier_prefixes:
            return None
        return identifier.upper()

    def _term_for_identifier(self, identifier: str) -> IsolationSourceTerm | None:
        return self._identifier_match_index.get(self._declared_identifier(identifier))

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
        values: Sequence[str],
        ontology_gap_diagnostics: tuple[IsolationSourceOntologyGapDiagnostic, ...] = (),
    ) -> StandardizedIsolationSource:
        """Combine one raw answer with current deterministic evidence and term derivation."""
        classifier_selection, classifier_term_ids = self._resolve_classifier_answer(answer)
        reasoning = [
            {
                "node": "llm_reasoning",
                "reasoning": answer.reasoning,
                "selected_terms": classifier_selection,
            }
        ]
        if any(_names_alternatives(value) for value in values):
            ancestor_ids = set(self.ont.nearest_common_ancestors(classifier_term_ids))
            if ancestor_ids != classifier_term_ids:
                classifier_term_ids = ancestor_ids
                reasoning.append(
                    {
                        "node": "alternatives_common_ancestor",
                        "reasoning": "The values name alternatives, so the terms of one tree "
                        "are replaced by their nearest common ancestor.",
                        "selected_terms": self._reasoning_selection(ancestor_ids),
                    }
                )
        evidence_level = _resolve_evidence_level(
            direct_term_ids=direct_term_ids,
            classifier_term_ids=classifier_term_ids,
        )
        classification = StandardizedIsolationSource(
            selected_terms=self._resolved_terms(direct_term_ids | classifier_term_ids),
            reasoning=reasoning,
            evidence_level=evidence_level,
            classifier_term_ids=frozenset(classifier_term_ids),
            identifier_disagreement=identifier_disagreement,
            request_fingerprint=request_fingerprint,
            ontology_gap_diagnostics=ontology_gap_diagnostics,
        )
        return self._derive_terms(classification, frozenset(direct_term_ids))

    def _derive_terms(
        self,
        classification: StandardizedIsolationSource,
        direct_term_ids: frozenset[str] = frozenset(),
    ) -> StandardizedIsolationSource:
        """Apply the body term of a diagnosis, add the relation objects of the explicit terms,
        derive the source type, include the ancestors of all terms, and order all terms
        canonically."""
        explicit_ids = {term.term_id for term in classification.selected_terms}
        reasoning = list(classification.reasoning)

        is_food = FOOD_OR_FEED_TERM_ID in self.ont.derive(explicit_ids)
        if is_food:
            host_ids = {
                term_id
                for term_id in explicit_ids
                if self.ont.terms[term_id].facet in FOOD_EXCLUDED_FACETS
                or _is_host_source_type(self.ont, term_id)
            }
            explicit_ids -= host_ids
            if host_ids:
                reasoning.append(
                    {
                        "node": "food_host_removal",
                        "reasoning": "A food record gets no host source type, body product, "
                        "body site, or host disease.",
                        "selected_terms": {},
                    }
                )

        disease_body_ids = {
            self.config.disease_body_terms[term_id]
            for term_id in explicit_ids
            if term_id in self.config.disease_body_terms
        }
        explicit_facets = {self.ont.terms[term_id].facet for term_id in explicit_ids}
        if len(disease_body_ids) == 1 and explicit_facets.isdisjoint(SAMPLED_FACETS):
            explicit_ids |= disease_body_ids
            reasoning.append(
                {
                    "node": "disease_body_term",
                    "reasoning": "The record names nothing that was sampled, so the diagnosis "
                    "sets its body term.",
                    "selected_terms": self._reasoning_selection(disease_body_ids),
                }
            )
        elif disease_body_ids and "body_product" in explicit_facets:
            dropped_ids = {
                term_id
                for term_id in explicit_ids & (classification.classifier_term_ids - direct_term_ids)
                if self.ont.terms[term_id].facet == "body_site"
                and any(
                    body_id in (term_id, *_ancestor_ids(self.ont.terms, term_id))
                    or term_id in _ancestor_ids(self.ont.terms, body_id)
                    for body_id in disease_body_ids
                )
            }
            if dropped_ids:
                explicit_ids -= dropped_ids
                reasoning.append(
                    {
                        "node": "disease_body_site_removal",
                        "reasoning": "The record names a body product, so the body sites of the "
                        "diagnosis are removed.",
                        "selected_terms": {},
                    }
                )

        relation_additions = {
            object_id
            for explicit_id in explicit_ids
            for object_id in self.ont.relation_object_ids[explicit_id]
        } - explicit_ids
        if relation_additions:
            reasoning.append(
                {
                    "node": "relation",
                    "reasoning": "Vocabulary relations assigned related terms.",
                    "selected_terms": self._reasoning_selection(relation_additions),
                }
            )

        source_additions = self.ont.implied_source_type_ids(explicit_ids | relation_additions) - (
            explicit_ids | relation_additions
        )
        if source_additions:
            reasoning.append(
                {
                    "node": "source_type_derivation",
                    "reasoning": "source_type derived from other facets.",
                    "selected_terms": self._reasoning_selection(source_additions),
                }
            )

        selected_ids = {
            term_id
            for term_id in self.ont.derive(explicit_ids)
            if not (is_food and _is_host_source_type(self.ont, term_id))
        }
        return replace(
            classification,
            selected_terms=self._resolved_terms(selected_ids),
            reasoning=reasoning,
            host_recovery_eligible=not self.config.host_recovery_term_ids.isdisjoint(selected_ids),
        )

    @staticmethod
    def _format_metadata(attrs: list[str], vals: list[str]) -> str:
        """Render selected attribute-value pairs as one line per pair.

        Both sides are JSON string literals, because BioSample attribute names and
        values are submitted free text. Line breaks, quotation marks, and equals signs
        inside one pair would otherwise render as further `attribute = value` lines,
        and the classifier would read attributes the sample never declared.
        `ensure_ascii=False` keeps non-ASCII source descriptions readable to the model.
        """
        return "Metadata:\n" + "\n".join(
            f"{json.dumps(attribute, ensure_ascii=False)} = {json.dumps(value, ensure_ascii=False)}"
            for attribute, value in zip(attrs, vals, strict=True)
        )

    def _classification_request(
        self,
        attributes: list[str],
        values: list[str],
        response_schema_id: str,
    ) -> CanonicalLLMRequest:
        """Build one classification request from selected attribute-value pairs."""
        values = [value.replace("_", " ") for value in values]
        return CanonicalLLMRequest(
            model=self.model,
            messages=(
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_template.format(
                        metadata=self._format_metadata(attributes, values),
                    ),
                },
            ),
            parameters=ISOLATION_SOURCE_LLM_PARAMETERS,
            response_schema_id=response_schema_id,
        )

    def standardize_record(
        self,
        accession: str,
        attr_name: str,
        value: str,
        context: tuple[SupportingAttributeValuePair, ...] = (),
    ) -> StandardizedIsolationSource:
        """Classify one record through deterministic matching, cache, and model fallback.

        The model request also shows the `context` pairs, which never match directly.
        """

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
                        "node": "no_values",
                        "reasoning": "No non-empty selected values were provided.",
                        "selected_terms": {},
                    }
                ],
            )

        source_pairs = [
            (a, v)
            for a, v in zip(valid_attrs, valid_vals, strict=True)
            if normalize_keyword(v) not in self.config.non_source_values
        ]
        valid_attrs = [a for a, _ in source_pairs]
        valid_vals = [v for _, v in source_pairs]

        # Direct-match pass over each (attr, val) pair before calling the LLM.
        direct_term_ids: set[str] = set()
        consumed_value_count = 0
        identifier_disagreement = False
        for v in valid_vals:
            terms, value_disagreement = self._direct_match(v)
            identifier_disagreement |= value_disagreement
            if terms:
                direct_term_ids.update(term.term_id for term in terms)
                consumed_value_count += 1
                self.stats["exact_matches"] += 1

        selected_term_ids: set[str] = set()
        classifier_term_ids: set[str] = set()
        reasoning_history: list[dict] = []

        # LLM processing is skipped only when every value resolved on its own. Partial
        # coverage still goes to the model, which sees the unresolved values in
        # context rather than having them dropped.
        direct_covers_all = consumed_value_count == len(valid_vals)
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
            response_schema_id = (
                f"{ISOLATION_SOURCE_RESPONSE_SCHEMA_ID}:"
                f"{ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE.value}:"
                f"{canonical_json_sha256(response_schema.model_json_schema())}"
            )
            request_attrs = valid_attrs + [pair.attribute for pair in context]
            request_vals = valid_vals + [pair.value for pair in context]
            request = self._classification_request(request_attrs, request_vals, response_schema_id)
            cache_fingerprint = self._classification_request(
                request_attrs,
                [_strip_digits(value) for value in request_vals],
                response_schema_id,
            ).fingerprint
            cached_answer = self.cache.get(cache_fingerprint)
            if cached_answer:
                self.stats["cache_hits"] += 1
                return self._standardize_classifier_answer(
                    cached_answer,
                    direct_term_ids=direct_term_ids,
                    identifier_disagreement=identifier_disagreement,
                    request_fingerprint=cache_fingerprint,
                    values=valid_vals,
                )

            if self.client is None:
                selected_term_ids |= direct_term_ids
                evidence_level = _resolve_evidence_level(
                    direct_term_ids=direct_term_ids,
                    classifier_term_ids=set(),
                )
                reasoning_history.append(
                    {
                        "node": "llm_disabled",
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
                    facet_values: dict[str, str | tuple[str, ...] | None] = {}
                    for facet in self._ordered_facets:
                        value = getattr(resp, facet.key)
                        facet_values[facet.key] = tuple(value) if isinstance(value, list) else value
                    classifier_answer = IsolationSourceClassifierAnswer(
                        facet_values=facet_values,
                        reasoning=resp.reasoning,
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

                self.cache.set(cache_fingerprint, classifier_answer)
                return self._standardize_classifier_answer(
                    classifier_answer,
                    direct_term_ids=direct_term_ids,
                    identifier_disagreement=identifier_disagreement,
                    request_fingerprint=cache_fingerprint,
                    values=valid_vals,
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
            classification = replace(classification, request_fingerprint=cache_fingerprint)
        return self._derive_terms(classification)


# --- Main class ---


class IsolationSourceStandardizer:
    """Standardize one extracted metadata record from BioSample evidence."""

    def __init__(
        self,
        config: IsolationSourceConfig,
        result_logger: logging.Logger | None = None,
        client: object = _LOAD_CONFIGURED_CLIENT,
        llm_settings: LLMSettings | None = None,
    ) -> None:
        self.logger = result_logger or logger
        self.config = config

        self.cache = SQLiteCache(config.cache_db_path)
        try:
            self.ontology = config.ontology
            self.pipeline = LLMClassifier(
                config,
                self.ontology,
                self.cache,
                result_logger=self.logger,
                client=client,
                llm_settings=llm_settings,
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
        host_pairs: tuple[SupportingAttributeValuePair, ...] = (),
    ) -> IsolationSourceOutcome | IsolationSourceRejection:
        """Classify one extracted metadata record, including host overflow values.

        `overflow` contains values the host standardizer rejected as hosts but retained
        for isolation-source standardization. The classifier includes them in its input.
        `host_pairs` are the pairs of a standardized host. The model request shows them as
        context.
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
                host_pairs,
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
        """Add the host source type derived from host lineage."""
        selected_ids = {term.term_id for term in outcome.selected_terms}
        if FOOD_OR_FEED_TERM_ID in selected_ids:
            return outcome
        host_source_type_ids = {
            term_id for term_id in selected_ids if _is_host_source_type(self.ontology, term_id)
        }
        selected_ids -= host_source_type_ids
        selected_ids.add(source_type_term_id)
        parent_id = self.ontology.terms[source_type_term_id].parent_id
        while parent_id is not None:
            selected_ids.add(parent_id)
            parent_id = self.ontology.terms[parent_id].parent_id

        derived_label = self.ontology.terms[source_type_term_id].label
        specific_previous_labels = [
            self.ontology.terms[term_id].label
            for term_id in host_source_type_ids
            if term_id != HOST_ASSOCIATED_TERM_ID
        ]
        disagreement = bool(
            specific_previous_labels and derived_label not in specific_previous_labels
        )
        reasoning = f"Derived {derived_label!r} from standardized taxid {host_taxid}"
        if disagreement:
            reasoning += (
                f" Replaced conflicting host source-type selection {specific_previous_labels!r}."
            )
        return replace(
            outcome,
            selected_terms=self.pipeline._resolved_terms(selected_ids),
            reasoning=(
                *outcome.reasoning,
                IsolationSourceReasoningStep(
                    node="host_lineage_derivation",
                    reasoning=reasoning,
                    selected_terms={SOURCE_TYPE_FACET: (derived_label,)},
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

    def add_laboratory_from_strain(
        self,
        result: IsolationSourceOutcome | IsolationSourceRejection,
        organism: str,
        strain: str,
    ) -> IsolationSourceOutcome | IsolationSourceRejection:
        """Add `laboratory` when the organism name or the strain is a known laboratory strain."""
        patterns = [
            (species, pattern)
            for species, pattern in self.config.laboratory_strain_patterns
            if organism.startswith(species)
        ]
        if any(
            pattern.search(_strain_key(organism[len(species) :])) for species, pattern in patterns
        ):
            reasoning = f"Organism {organism!r} is a laboratory strain."
        elif any(pattern.search(_strain_key(strain)) for _, pattern in patterns):
            reasoning = f"Strain {strain!r} of {organism!r} is a laboratory strain."
        else:
            return result
        if isinstance(result, IsolationSourceRejection):
            result = IsolationSourceOutcome(
                selected_terms=(),
                evidence_level=IsolationSourceEvidenceLevel.NONE,
                supporting_pairs=(),
                host_recovery_pairs=(),
                reasoning=(),
                diagnostics=result.diagnostics,
                exact_matches=0,
                cache_hits=0,
                llm_calls=0,
                ontology_gap_diagnostics=result.ontology_gap_diagnostics,
            )
        selected_ids = {term.term_id for term in result.selected_terms} | {LABORATORY_TERM_ID}
        return replace(
            result,
            selected_terms=self.pipeline._resolved_terms(selected_ids),
            reasoning=(
                *result.reasoning,
                IsolationSourceReasoningStep(
                    node="laboratory_strain",
                    reasoning=reasoning,
                    selected_terms={SOURCE_TYPE_FACET: ("laboratory strain",)},
                ),
            ),
            diagnostics=tuple(
                diagnostic
                for diagnostic in result.diagnostics
                if diagnostic is not IsolationSourceDiagnostic.UNSPECIFIED
            ),
            evidence_level=IsolationSourceEvidenceLevel.SAMPLE,
        )

    def close(self) -> None:
        try:
            self.pipeline.close()
        finally:
            self.cache.close()
