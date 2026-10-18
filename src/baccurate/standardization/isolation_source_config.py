"""Load and validate the isolation-source config."""

import os
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import instructor

from baccurate.adapters.config_yaml import ConfigError, load_config_mapping
from baccurate.adapters.llm.request import canonical_json_sha256
from baccurate.paths import (
    DEFAULT_ISOLATION_SOURCE_CACHE_DB,
    DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY,
)
from baccurate.standardization._attribute_value_text import normalize_keyword
from baccurate.standardization.isolation_source_ontology import (
    ANIMAL_HOST_TERM_ID,
    ENVIRONMENTAL_TERM_ID,
    FOOD_OR_FEED_TERM_ID,
    HOST_ASSOCIATED_TERM_ID,
    HOST_DISEASE_FACET,
    HUMAN_HOST_TERM_ID,
    PLANT_HOST_TERM_ID,
    SOURCE_TYPE_FACET,
    IsolationSourceOntology,
    IsolationSourceOntologyError,
    _ancestor_ids,
    render_ontology,
)

ISOLATION_SOURCE_LLM_PARAMETERS: dict[str, object] = {"temperature": 0, "seed": 100}
# The configured endpoint was verified to enforce JSON Schema constraints. Pinning this
# mode makes facet enumerations generation constraints instead of retry-time validation only.
ISOLATION_SOURCE_STRUCTURED_OUTPUT_MODE = instructor.Mode.JSON_SCHEMA

DISEASE_BODY_FACETS = frozenset({"body_product", "body_site"})


def _validate_term_derivation_vocabulary(ontology: IsolationSourceOntology) -> None:
    """Validate the fixed term identities required by term derivation."""
    required_ids = {
        HOST_ASSOCIATED_TERM_ID,
        ANIMAL_HOST_TERM_ID,
        HUMAN_HOST_TERM_ID,
        PLANT_HOST_TERM_ID,
        ENVIRONMENTAL_TERM_ID,
        FOOD_OR_FEED_TERM_ID,
    }
    for term_id in sorted(required_ids):
        term = ontology.terms.get(term_id)
        if term is None:
            raise ValueError(f"term derivation requires the missing term {term_id!r}")
        if term.facet != SOURCE_TYPE_FACET:
            raise ValueError(
                f"term derivation requires term {term_id!r} in facet {SOURCE_TYPE_FACET!r}"
            )

    for root_id in (
        HOST_ASSOCIATED_TERM_ID,
        ENVIRONMENTAL_TERM_ID,
        FOOD_OR_FEED_TERM_ID,
    ):
        if ontology.terms[root_id].parent_id is not None:
            raise ValueError(f"required broad source-kind term {root_id!r} must be a root term")

    for host_term_id in (ANIMAL_HOST_TERM_ID, HUMAN_HOST_TERM_ID, PLANT_HOST_TERM_ID):
        if HOST_ASSOCIATED_TERM_ID not in _ancestor_ids(ontology.terms, host_term_id):
            raise ValueError(
                f"term derivation requires term {host_term_id!r} to descend from "
                f"{HOST_ASSOCIATED_TERM_ID!r}"
            )


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
class FacetConfiguration:
    """How the pipeline renders, asks for, and writes one vocabulary facet."""

    key: str
    output_column: str
    render_order: int
    classifier_guidance: str


@dataclass(frozen=True, slots=True)
class IsolationSourceConfig:
    """Validated isolation-source prompt, ontology, and cache config."""

    schema_version: int
    prompt_version: str
    prompts: IsolationSourcePromptTemplates
    effective_prompts: IsolationSourcePrompts
    ontology_directory: Path
    ontology: IsolationSourceOntology
    facets: tuple[FacetConfiguration, ...]
    host_recovery_term_ids: frozenset[str]
    direct_match_values: Mapping[str, tuple[str, ...]]
    non_source_values: frozenset[str]
    cache_db_path: Path
    laboratory_strain_patterns: tuple[tuple[str, re.Pattern[str]], ...] = ()
    disease_body_terms: Mapping[str, str] = MappingProxyType({})

    @classmethod
    def load(cls, path: Path | str) -> "IsolationSourceConfig":
        """Strictly load isolation-source config from one YAML file."""
        config_path = Path(path)
        return _parse_isolation_source_config(load_config_mapping(config_path), config_path)

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


def _isolation_source_config_error(config_path: Path, key: str, message: str) -> ConfigError:
    return ConfigError(f"{config_path}: {key}: {message}")


def _require_isolation_source_string(
    config: Mapping[object, object],
    key: str,
    config_path: Path,
) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _isolation_source_config_error(config_path, key, "must be a non-empty string")
    return value


def _validate_isolation_source_prompt(
    template: str,
    *,
    config_path: Path,
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
            raise _isolation_source_config_error(
                config_path, key, f"malformed format string: {error}"
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
        raise _isolation_source_config_error(
            config_path,
            key,
            f"must contain exactly one of each required placeholder ({placeholders}) "
            "and no other placeholders",
        )


def _isolation_source_resource_selection(
    config: Mapping[object, object],
    key: str,
    default: Path,
    config_path: Path,
) -> str:
    """Return the effective path text for one resource."""
    if key not in config:
        return str(default)
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise _isolation_source_config_error(config_path, key, "must be a non-empty string")
    return value


def _parse_facet_configurations(
    config: Mapping[object, object],
    ontology: IsolationSourceOntology,
    config_path: Path,
) -> tuple[FacetConfiguration, ...]:
    raw_facets = config.get("facets")
    if not isinstance(raw_facets, Mapping):
        raise _isolation_source_config_error(config_path, "facets", "must be a mapping")
    undeclared = sorted(str(key) for key in set(raw_facets) - set(ontology.facets))
    if undeclared:
        raise _isolation_source_config_error(
            config_path, f"facets.{undeclared[0]}", "the vocabulary does not declare this facet"
        )
    unconfigured = sorted(set(ontology.facets) - set(raw_facets))
    if unconfigured:
        raise _isolation_source_config_error(
            config_path, f"facets.{unconfigured[0]}", "a vocabulary facet must be configured"
        )
    fields = {"output_column", "render_order", "classifier_guidance"}
    facets = []
    for key, raw_facet in raw_facets.items():
        location = f"facets.{key}"
        if not isinstance(raw_facet, Mapping) or set(raw_facet) != fields:
            raise _isolation_source_config_error(
                config_path,
                location,
                f"must be a mapping with the keys {', '.join(sorted(fields))}",
            )
        if type(raw_facet["render_order"]) is not int:
            raise _isolation_source_config_error(
                config_path, f"{location}.render_order", "must be an integer"
            )
        for text_key in ("output_column", "classifier_guidance"):
            value = raw_facet[text_key]
            if not isinstance(value, str) or not value.strip():
                raise _isolation_source_config_error(
                    config_path, f"{location}.{text_key}", "must be a non-empty string"
                )
        facets.append(
            FacetConfiguration(
                key=key,
                output_column=raw_facet["output_column"].strip(),
                render_order=raw_facet["render_order"],
                classifier_guidance=raw_facet["classifier_guidance"].strip(),
            )
        )
    return tuple(sorted(facets, key=lambda facet: facet.render_order))


def _parse_host_recovery_term_ids(
    config: Mapping[object, object],
    ontology: IsolationSourceOntology,
    config_path: Path,
) -> frozenset[str]:
    raw_ids = config.get("host_recovery_term_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(term_id, str) for term_id in raw_ids):
        raise _isolation_source_config_error(
            config_path, "host_recovery_term_ids", "must be a list of term identifiers"
        )
    unknown = sorted(set(raw_ids) - set(ontology.terms))
    if unknown:
        raise _isolation_source_config_error(
            config_path, "host_recovery_term_ids", f"unknown term {unknown[0]!r}"
        )
    return frozenset(raw_ids)


def _parse_disease_body_terms(
    config: Mapping[object, object],
    ontology: IsolationSourceOntology,
    config_path: Path,
) -> Mapping[str, str]:
    raw_terms = config.get("disease_body_terms", {})
    if not isinstance(raw_terms, Mapping) or not all(
        isinstance(disease_id, str) and isinstance(body_id, str)
        for disease_id, body_id in raw_terms.items()
    ):
        raise _isolation_source_config_error(
            config_path, "disease_body_terms", "must map term identifiers to term identifiers"
        )
    for disease_id, body_id in raw_terms.items():
        location = f"disease_body_terms.{disease_id}"
        disease = ontology.terms.get(disease_id)
        if disease is None or disease.facet != HOST_DISEASE_FACET:
            raise _isolation_source_config_error(
                config_path, location, f"{disease_id!r} is not an active host-disease term"
            )
        body = ontology.terms.get(body_id)
        if body is None or body.facet not in DISEASE_BODY_FACETS:
            raise _isolation_source_config_error(
                config_path, location, f"{body_id!r} is not an active body term"
            )
    return MappingProxyType(dict(raw_terms))


def _parse_direct_match_values(
    config: Mapping[object, object],
    ontology: IsolationSourceOntology,
    config_path: Path,
) -> Mapping[str, tuple[str, ...]]:
    raw_values = config.get("direct_match_values")
    shape_error = _isolation_source_config_error(
        config_path,
        "direct_match_values",
        "must map values to a term identifier or a list of term identifiers",
    )
    if not isinstance(raw_values, Mapping):
        raise shape_error
    term_ids_by_value: dict[str, tuple[str, ...]] = {}
    term_ids_by_key = dict(ontology.match_keys)
    for value, raw_term_ids in raw_values.items():
        term_ids = [raw_term_ids] if isinstance(raw_term_ids, str) else raw_term_ids
        if not (
            isinstance(value, str)
            and isinstance(term_ids, list)
            and term_ids
            and all(isinstance(term_id, str) for term_id in term_ids)
        ):
            raise shape_error
        location = f"direct_match_values.{value}"
        for term_id in term_ids:
            if term_id not in ontology.terms:
                raise _isolation_source_config_error(
                    config_path, location, f"{term_id!r} is not an active term"
                )
        key = normalize_keyword(value)
        if key in term_ids_by_key:
            raise _isolation_source_config_error(
                config_path, location, f"match key {key!r} already names {term_ids_by_key[key]!r}"
            )
        term_ids_by_key[key] = ", ".join(term_ids)
        term_ids_by_value[value] = tuple(term_ids)
    return MappingProxyType(term_ids_by_value)


def _parse_non_source_values(
    config: Mapping[object, object],
    ontology: IsolationSourceOntology,
    direct_match_values: Mapping[str, tuple[str, ...]],
    config_path: Path,
) -> frozenset[str]:
    raw_values = config.get("non_source_values", [])
    if not isinstance(raw_values, list) or not all(isinstance(value, str) for value in raw_values):
        raise _isolation_source_config_error(
            config_path, "non_source_values", "must be a list of values"
        )
    term_keys = set(ontology.match_keys).union(map(normalize_keyword, direct_match_values))
    keys = [normalize_keyword(value) for value in raw_values]
    for value, key in zip(raw_values, keys, strict=True):
        if key in term_keys:
            raise _isolation_source_config_error(
                config_path, f"non_source_values.{value}", f"match key {key!r} already names a term"
            )
    return frozenset(keys)


def _parse_laboratory_strain_patterns(
    config: Mapping[object, object],
    config_path: Path,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    raw_names = config.get("laboratory_strain_names", {})
    if not isinstance(raw_names, dict) or not all(
        isinstance(species, str)
        and species.strip()
        and isinstance(names, list)
        and names
        and all(isinstance(name, str) and name.strip() for name in names)
        for species, names in raw_names.items()
    ):
        raise _isolation_source_config_error(
            config_path,
            "laboratory_strain_names",
            "must map each species to a list of strain names",
        )
    return tuple(
        (
            species,
            re.compile(
                "(?<![a-z0-9])(?:"
                + "|".join(
                    "[-_ ]?".join(map(re.escape, re.split(r"[-_ ]", name.casefold())))
                    for name in names
                )
                + ")(?![a-z0-9])"
            ),
        )
        for species, names in raw_names.items()
    )


def _parse_isolation_source_config(
    config: Mapping[object, object],
    config_path: Path,
) -> IsolationSourceConfig:
    allowed = {
        "schema_version",
        "prompt_version",
        "system_prompt",
        "user_prompt",
        "ontology_directory",
        "cache_db_path",
        "facets",
        "host_recovery_term_ids",
        "direct_match_values",
        "disease_body_terms",
        "non_source_values",
        "laboratory_strain_names",
    }
    unknown = set(config) - allowed
    if unknown:
        key = sorted(str(value) for value in unknown)[0]
        raise _isolation_source_config_error(config_path, f"top level.{key}", "unknown config key")

    schema_version = config.get("schema_version")
    if type(schema_version) is not int:
        raise _isolation_source_config_error(
            config_path, "schema_version", "must be integer version 3"
        )
    if schema_version != 3:
        raise _isolation_source_config_error(
            config_path,
            "schema_version",
            f"unsupported schema version {schema_version}; expected version 3; "
            "migrate the isolation-source config, then retry",
        )

    prompt_version = _require_isolation_source_string(config, "prompt_version", config_path)
    sample_system = _require_isolation_source_string(config, "system_prompt", config_path)
    sample_user = _require_isolation_source_string(config, "user_prompt", config_path)
    _validate_isolation_source_prompt(
        sample_system,
        config_path=config_path,
        key="system_prompt",
        required_fields=("ontology_tree",),
    )
    _validate_isolation_source_prompt(
        sample_user,
        config_path=config_path,
        key="user_prompt",
        required_fields=("metadata",),
        uses_format=True,
    )

    ontology_value = _isolation_source_resource_selection(
        config,
        "ontology_directory",
        DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY,
        config_path,
    )
    ontology_directory = Path(ontology_value)
    if not ontology_directory.is_dir() or not os.access(ontology_directory, os.R_OK):
        raise _isolation_source_config_error(
            config_path,
            "ontology_directory",
            f"must select a readable directory: {ontology_directory}",
        )
    try:
        ontology = IsolationSourceOntology.load(ontology_directory)
        _validate_term_derivation_vocabulary(ontology)
    except (IsolationSourceOntologyError, ConfigError, ValueError) as error:
        raise _isolation_source_config_error(
            config_path,
            "ontology_directory",
            f"must select a readable isolation-source ontology directory: {error}",
        ) from error

    facets = _parse_facet_configurations(config, ontology, config_path)
    host_recovery_term_ids = _parse_host_recovery_term_ids(config, ontology, config_path)
    direct_match_values = _parse_direct_match_values(config, ontology, config_path)
    non_source_values = _parse_non_source_values(config, ontology, direct_match_values, config_path)

    cache_value = _isolation_source_resource_selection(
        config,
        "cache_db_path",
        DEFAULT_ISOLATION_SOURCE_CACHE_DB,
        config_path,
    )
    cache_path = Path(cache_value)
    writable_cache_target = cache_path if cache_path.exists() else cache_path.parent
    if (cache_path.exists() and not cache_path.is_file()) or (
        writable_cache_target.exists() and not os.access(writable_cache_target, os.W_OK)
    ):
        raise _isolation_source_config_error(
            config_path,
            "cache_db_path",
            "must select a writable database file",
        )

    prompts = IsolationSourcePromptTemplates(
        sample_system_template=sample_system,
        sample_user_template=sample_user,
    )
    return IsolationSourceConfig(
        schema_version=3,
        prompt_version=prompt_version,
        prompts=prompts,
        effective_prompts=IsolationSourcePrompts(
            system=sample_system.replace(
                "{ontology_tree}",
                render_ontology(
                    ontology,
                    {
                        facet.key: f"{ontology.facets[facet.key].meaning} "
                        f"{facet.classifier_guidance}"
                        for facet in facets
                    },
                ),
            ),
            user_template=sample_user,
        ),
        ontology_directory=ontology_directory,
        ontology=ontology,
        facets=facets,
        host_recovery_term_ids=host_recovery_term_ids,
        direct_match_values=direct_match_values,
        non_source_values=non_source_values,
        cache_db_path=cache_path,
        laboratory_strain_patterns=_parse_laboratory_strain_patterns(config, config_path),
        disease_body_terms=_parse_disease_body_terms(config, ontology, config_path),
    )
