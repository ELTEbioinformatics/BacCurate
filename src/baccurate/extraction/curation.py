"""Validated curation schema for selecting BioSample metadata candidates."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml

from baccurate.extraction.metadata_types import (
    ATTRIBUTES,
    DATE_OTHER,
    DATE_SAMPLING,
    AttributeMatch,
)

_TOKEN_BOUNDARIES = re.compile(r"[\W_]+", re.UNICODE)
_TARGETS = frozenset(ATTRIBUTES)


class CurationSchemaError(ValueError):
    """Raised when the curation schema is missing or invalid."""


@dataclass(frozen=True, slots=True)
class CurationEvent:
    """A finding from evaluating one raw attribute-value pair."""

    kind: str
    target: str
    family: str
    normalized_attribute: str
    normalized_value: str = ""


@dataclass(frozen=True, slots=True)
class CurationDecision:
    """Candidate matches and review events for one unchanged raw pair."""

    attribute: str
    value: str
    matches: tuple[AttributeMatch, ...]
    events: tuple[CurationEvent, ...]
    xml_source: str = ""


@dataclass(frozen=True, slots=True)
class _TargetRules:
    accepted: Mapping[str, str]
    ignored: frozenset[str]
    discoveries: tuple[_DiscoveryRule, ...]
    value_rejections: tuple[_ValueRejectionRule, ...]


@dataclass(frozen=True, slots=True)
class _ValueRejectionRule:
    family: str
    exact: frozenset[str]
    explanation_prefixes: tuple[str, ...] = ()
    explanation_suffixes: frozenset[str] = frozenset()
    fuzzy_references: frozenset[str] = frozenset()

    def classify(self, normalized_value: str, presentation_value: str) -> str | None:
        if self.family == "universal_missing" and not normalized_value:
            return "rejected_value"
        if normalized_value in self.exact or self._matches_explanation(
            normalized_value, presentation_value
        ):
            return "rejected_value"
        if not self.fuzzy_references:
            return None
        return _fuzzy_classification(normalized_value, self.fuzzy_references)

    def _matches_explanation(self, normalized_value: str, presentation_value: str) -> bool:
        allowed_suffixes = self.exact | self.explanation_suffixes
        for prefix in self.explanation_prefixes:
            if presentation_value.startswith(f"{prefix}:"):
                suffix = _normalize_rejection_value(presentation_value.removeprefix(f"{prefix}:"))
                if suffix in allowed_suffixes:
                    return True
            for separator in (" due to ", " because "):
                if normalized_value.startswith(f"{prefix}{separator}"):
                    suffix = normalized_value.removeprefix(f"{prefix}{separator}")
                    if suffix in allowed_suffixes:
                        return True
        return False


@dataclass(frozen=True, slots=True)
class _DiscoveryRule:
    family: str
    tokens: frozenset[str]
    phrases: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]

    def matches(self, normalized_attribute: str) -> bool:
        tokens = frozenset(normalized_attribute.split())
        if tokens & self.tokens:
            return True
        padded = f" {normalized_attribute} "
        return any(f" {phrase} " in padded for phrase in self.phrases) or any(
            pattern.search(normalized_attribute) is not None for pattern in self.patterns
        )


@dataclass(frozen=True, slots=True)
class _LocationRescue:
    prefixes: tuple[str, ...]
    separators: tuple[str, ...]

    def matches(self, value: str) -> bool:
        normalized = _presentation_normalize(value)
        for prefix in self.prefixes:
            if not normalized.startswith(prefix):
                continue
            remainder = normalized[len(prefix) :].lstrip()
            for separator in self.separators:
                if remainder.startswith(separator) and remainder[len(separator) :].strip():
                    return True
        return False


class CurationSchema:
    """Match raw attributes against reviewed per-target accept/ignore/reject lists."""

    def __init__(
        self,
        targets: Mapping[str, _TargetRules],
        universal_missing: _ValueRejectionRule,
        location_rescue: _LocationRescue | None,
    ) -> None:
        self._targets = dict(targets)
        self._universal_missing = universal_missing
        self._location_rescue = location_rescue

    @classmethod
    def load(cls, config_path: Path | str) -> Self:
        path = Path(config_path)
        config = _load_config(path)
        _reject_unknown_keys(
            path,
            config,
            {"schema_version", "universal_missing", "location_rescue", "targets"},
            "top level",
        )
        if config.get("schema_version") != 3:
            raise CurationSchemaError(f"{path}: schema_version must be 3")
        targets = _mapping(path, "targets", config.get("targets"))
        unknown_targets = set(targets) - _TARGETS
        if unknown_targets:
            raise CurationSchemaError(
                f"{path}: targets has unsupported target(s): "
                f"{', '.join(sorted(str(target) for target in unknown_targets))}"
            )
        universal_missing = _compile_value_rejection_rule(
            path,
            "universal_missing",
            "universal_missing",
            config.get("universal_missing", {}),
        )
        location_rescue = _compile_location_rescue(path, config.get("location_rescue"))
        return cls(
            {
                target: _compile_target(path, target, target_config)
                for target, target_config in targets.items()
            },
            universal_missing,
            location_rescue,
        )

    def evaluate(self, *, attribute: str, value: str) -> CurationDecision:
        normalized_attribute = _normalize(attribute)
        normalized_value = _normalize_rejection_value(value)
        presentation_value = _presentation_normalize(value)
        identified = [
            AttributeMatch(target, self._targets[target].accepted[normalized_attribute])
            for target in ATTRIBUTES
            if target in self._targets and normalized_attribute in self._targets[target].accepted
        ]
        location_rules = self._targets.get("loc")
        if (
            self._location_rescue is not None
            and location_rules is not None
            and normalized_attribute not in location_rules.ignored
            and not any(match.target == "loc" for match in identified)
            and self._location_rescue.matches(value)
        ):
            identified.append(AttributeMatch("loc"))
        events: list[CurationEvent] = []
        matches = []
        for match in identified:
            rejection = self._universal_missing.classify(normalized_value, presentation_value)
            family = self._universal_missing.family
            if rejection is None:
                target_rejection = _target_rejection(
                    self._targets[match.target], normalized_value, presentation_value
                )
                if target_rejection is not None:
                    rejection, family = target_rejection
            if rejection is None:
                matches.append(match)
                continue
            events.append(
                CurationEvent(
                    rejection,
                    match.target,
                    family,
                    normalized_attribute,
                    normalized_value,
                )
            )
            if rejection == "uncertain_rejection":
                matches.append(match)

        matched_targets = {match.target for match in identified}
        for target in ATTRIBUTES:
            target_rules = self._targets.get(target)
            if (
                target_rules is None
                or target in matched_targets
                or normalized_attribute in target_rules.ignored
            ):
                continue
            for discovery in target_rules.discoveries:
                if discovery.matches(normalized_attribute):
                    events.append(
                        CurationEvent(
                            "unreviewed_attribute",
                            target,
                            discovery.family,
                            normalized_attribute,
                        )
                    )
                    break
        return CurationDecision(attribute, value, tuple(matches), tuple(events))


def _load_config(path: Path) -> Mapping[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CurationSchemaError(f"{path}: cannot read curation schema: {error}") from error
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise CurationSchemaError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(config, Mapping):
        raise CurationSchemaError(f"{path}: top level must be a mapping")
    return config


def _compile_target(path: Path, target: str, raw_config: object) -> _TargetRules:
    prefix = f"targets.{target}"
    config = _mapping(path, prefix, raw_config)
    _reject_unknown_keys(
        path,
        config,
        {
            "attribute_discovery",
            "accepted_attributes",
            "ignored_attributes",
            "value_rejections",
        },
        prefix,
    )
    accept_groups = _mapping(
        path,
        f"{prefix}.accepted_attributes",
        config.get("accepted_attributes", {}),
    )
    ignore_groups = _mapping(
        path,
        f"{prefix}.ignored_attributes",
        config.get("ignored_attributes", {}),
    )
    if target == "date":
        unknown_categories = set(accept_groups) - {"sampling", "fallback"}
        if unknown_categories:
            raise CurationSchemaError(
                f"{path}: {prefix}.accepted_attributes has invalid date category "
                f"{', '.join(sorted(str(item) for item in unknown_categories))!r}; "
                "expected sampling or fallback"
            )
    accepted = _accepted_decisions(path, target, accept_groups)
    ignored = frozenset(
        value
        for _family, value in _decision_values(path, f"{prefix}.ignored_attributes", ignore_groups)
    )
    overlap = set(accepted) & ignored
    if overlap:
        raise CurationSchemaError(
            f"{path}: {prefix} attribute {sorted(overlap)[0]!r} is both accepted and ignored"
        )
    discovery_groups = _mapping(
        path,
        f"{prefix}.attribute_discovery",
        config.get("attribute_discovery", {}),
    )
    discoveries = tuple(
        _compile_discovery(path, target, family, rule) for family, rule in discovery_groups.items()
    )
    rejection_groups = _mapping(
        path,
        f"{prefix}.value_rejections",
        config.get("value_rejections", {}),
    )
    value_rejections = tuple(
        _compile_value_rejection_rule(path, f"{prefix}.value_rejections.{family}", family, rule)
        for family, rule in rejection_groups.items()
    )
    return _TargetRules(accepted, ignored, discoveries, value_rejections)


def _compile_location_rescue(path: Path, raw_config: object) -> _LocationRescue | None:
    if raw_config is None:
        return None
    prefix = "location_rescue"
    config = _mapping(path, prefix, raw_config)
    _reject_unknown_keys(path, config, {"vocabulary_path", "aliases", "separators"}, prefix)
    raw_vocabulary_path = config.get("vocabulary_path")
    if not isinstance(raw_vocabulary_path, str) or not raw_vocabulary_path.strip():
        raise CurationSchemaError(f"{path}: {prefix}.vocabulary_path must be a non-empty string")
    vocabulary_path = Path(raw_vocabulary_path)
    if not vocabulary_path.is_absolute():
        vocabulary_path = path.parent / vocabulary_path
    try:
        vocabulary_text = vocabulary_path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise CurationSchemaError(
            f"{path}: {prefix}.vocabulary_path cannot read {vocabulary_path}: {error}"
        ) from error
    prefixes = {
        normalized
        for line in vocabulary_text.splitlines()
        if (normalized := _presentation_normalize(line))
    }
    if not prefixes:
        raise CurationSchemaError(
            f"{path}: {prefix}.vocabulary_path {vocabulary_path} must not be empty"
        )
    for alias in _literal_nonempty_string_list(
        path, f"{prefix}.aliases", config.get("aliases", [])
    ):
        normalized_alias = _presentation_normalize(alias)
        if not normalized_alias:
            raise CurationSchemaError(f"{path}: {prefix}.aliases must contain non-empty names")
        prefixes.add(normalized_alias)
    separators = tuple(
        _literal_nonempty_string_list(path, f"{prefix}.separators", config.get("separators"))
    )
    if not separators:
        raise CurationSchemaError(f"{path}: {prefix}.separators must not be empty")
    return _LocationRescue(tuple(sorted(prefixes, key=lambda item: (-len(item), item))), separators)


def _compile_value_rejection_rule(
    path: Path, prefix: str, family: object, raw_rule: object
) -> _ValueRejectionRule:
    if not isinstance(family, str) or not family:
        raise CurationSchemaError(f"{path}: {prefix} family name must be a string")
    rule = _mapping(path, prefix, raw_rule)
    _reject_unknown_keys(
        path,
        rule,
        {
            "exact",
            "observed_variants",
            "explanation_prefixes",
            "explanation_suffixes",
            "fuzzy",
        },
        prefix,
    )
    canonical_exact = frozenset(
        _normalize_rejection_value(value)
        for value in _literal_nonempty_string_list(path, f"{prefix}.exact", rule.get("exact", []))
    )
    observed_variants = frozenset(
        _normalize_rejection_value(value)
        for value in _literal_nonempty_string_list(
            path,
            f"{prefix}.observed_variants",
            rule.get("observed_variants", []),
        )
    )
    explanation_prefixes = tuple(
        _normalize(value)
        for value in _normalized_nonempty_string_list(
            path, f"{prefix}.explanation_prefixes", rule.get("explanation_prefixes", [])
        )
    )
    explanation_suffixes = frozenset(
        _normalize_rejection_value(value)
        for value in _normalized_nonempty_string_list(
            path, f"{prefix}.explanation_suffixes", rule.get("explanation_suffixes", [])
        )
    )
    fuzzy_setting = rule.get("fuzzy")
    if fuzzy_setting not in (None, "one_edit_unique"):
        raise CurationSchemaError(
            f"{path}: {prefix}.fuzzy must be 'one_edit_unique' when configured"
        )
    return _ValueRejectionRule(
        family,
        canonical_exact | observed_variants,
        explanation_prefixes,
        explanation_suffixes,
        canonical_exact if fuzzy_setting == "one_edit_unique" else frozenset(),
    )


def _accepted_decisions(path: Path, target: str, groups: Mapping[str, object]) -> dict[str, str]:
    categories = {"sampling": DATE_SAMPLING, "fallback": DATE_OTHER} if target == "date" else {}
    values = _decision_values(path, f"targets.{target}.accepted_attributes", groups)
    result = {}
    for family, normalized in values:
        if normalized in result:
            raise CurationSchemaError(
                f"{path}: targets.{target}.accepted_attributes repeats attribute {normalized!r}"
            )
        result[normalized] = categories.get(family, "")
    return result


def _decision_values(
    path: Path, prefix: str, groups: Mapping[str, object]
) -> list[tuple[str, str]]:
    result = []
    for family, raw_values in groups.items():
        if not isinstance(family, str) or not family:
            raise CurationSchemaError(f"{path}: {prefix} family names must be strings")
        for value in _normalized_nonempty_string_list(path, f"{prefix}.{family}", raw_values):
            result.append((family, _normalize(value)))
    return result


def _compile_discovery(path: Path, target: str, family: object, raw_rule: object) -> _DiscoveryRule:
    prefix = f"targets.{target}.attribute_discovery.{family}"
    if not isinstance(family, str) or not family:
        raise CurationSchemaError(
            f"{path}: targets.{target}.attribute_discovery family names must be strings"
        )
    rule = _mapping(path, prefix, raw_rule)
    _reject_unknown_keys(path, rule, {"tokens", "phrases", "regex"}, prefix)
    tokens = frozenset(
        _normalize(value)
        for value in _normalized_nonempty_string_list(
            path, f"{prefix}.tokens", rule.get("tokens", [])
        )
    )
    phrases = tuple(
        _normalize(value)
        for value in _normalized_nonempty_string_list(
            path, f"{prefix}.phrases", rule.get("phrases", [])
        )
    )
    patterns = []
    for index, pattern in enumerate(
        _literal_nonempty_string_list(path, f"{prefix}.regex", rule.get("regex", []))
    ):
        try:
            patterns.append(re.compile(pattern))
        except re.error as error:
            raise CurationSchemaError(
                f"{path}: {prefix}.regex[{index}] invalid regular expression {pattern!r}: {error}"
            ) from error
    return _DiscoveryRule(family, tokens, phrases, tuple(patterns))


def _target_rejection(
    target: _TargetRules, normalized_value: str, presentation_value: str
) -> tuple[str, str] | None:
    uncertain_family = None
    for family in target.value_rejections:
        outcome = family.classify(normalized_value, presentation_value)
        if outcome == "rejected_value":
            return outcome, family.family
        if outcome == "uncertain_rejection" and uncertain_family is None:
            uncertain_family = family.family
    if uncertain_family is not None:
        return "uncertain_rejection", uncertain_family
    return None


def _fuzzy_classification(normalized_value: str, references: frozenset[str]) -> str | None:
    """Apply whole-value fuzzy matching inside one closed vocabulary."""
    candidate_tokens = normalized_value.split()
    safe_matches = 0
    near_matches = 0
    for reference in references:
        reference_tokens = reference.split()
        if len(candidate_tokens) != len(reference_tokens):
            continue
        distances = []
        eligible = True
        for candidate, expected in zip(candidate_tokens, reference_tokens, strict=True):
            if candidate == expected:
                distances.append(0)
                continue
            if (
                not candidate.isalpha()
                or not expected.isalpha()
                or min(len(candidate), len(expected)) < 5
            ):
                eligible = False
                break
            distances.append(_optimal_string_alignment_distance_at_most_two(candidate, expected))
        if not eligible or any(distance > 2 for distance in distances):
            continue
        total_distance = sum(distances)
        changed_words = sum(distance > 0 for distance in distances)
        if total_distance == 1 and changed_words == 1:
            safe_matches += 1
        elif total_distance == 2:
            near_matches += 1
    if safe_matches == 1:
        return "rejected_value"
    if safe_matches > 1 or near_matches:
        return "uncertain_rejection"
    return None


def _optimal_string_alignment_distance_at_most_two(left: str, right: str) -> int:
    """Return optimal string alignment distance, with values above two capped at three."""
    if abs(len(left) - len(right)) > 2:
        return 3
    previous_previous: list[int] | None = None
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
            if (
                previous_previous is not None
                and left_index > 1
                and right_index > 1
                and left_char == right[right_index - 2]
                and left[left_index - 2] == right_char
            ):
                current[-1] = min(current[-1], previous_previous[right_index - 2] + 1)
        if min(current) > 2:
            return 3
        previous_previous, previous = previous, current
    return min(previous[-1], 3)


def _mapping(path: Path, prefix: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CurationSchemaError(f"{path}: {prefix} must be a mapping")
    return value


def _normalized_nonempty_string_list(path: Path, prefix: str, value: object) -> Sequence[str]:
    if not isinstance(value, list):
        raise CurationSchemaError(f"{path}: {prefix} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not _normalize(item):
            raise CurationSchemaError(f"{path}: {prefix}[{index}] must be a non-empty string")
    return value


def _literal_nonempty_string_list(path: Path, prefix: str, value: object) -> Sequence[str]:
    if not isinstance(value, list):
        raise CurationSchemaError(f"{path}: {prefix} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise CurationSchemaError(f"{path}: {prefix}[{index}] must be a non-empty string")
    return value


def _reject_unknown_keys(
    path: Path, mapping: Mapping[str, object], allowed: set[str], prefix: str
) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise CurationSchemaError(
            f"{path}: {prefix} has unknown key(s): {', '.join(sorted(str(key) for key in unknown))}"
        )


def _normalize(value: str) -> str:
    normalized = _presentation_normalize(value)
    return " ".join(_TOKEN_BOUNDARIES.sub(" ", normalized).split())


def _presentation_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in "'\"":
        normalized = normalized[1:-1].strip()
    return " ".join(normalized.split())


def _normalize_rejection_value(value: str) -> str:
    normalized = _normalize(value)
    return normalized or _presentation_normalize(value)
