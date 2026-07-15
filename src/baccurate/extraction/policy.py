"""Validated extraction policy for identifying BioSample metadata candidates."""

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


class MetadataPolicyConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class PolicyEvent:
    """One review or audit observation produced while evaluating a raw pair."""

    kind: str
    target: str
    family: str
    normalized_attribute: str
    normalized_value: str = ""


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Candidate matches and review events for one unchanged raw pair."""

    attribute: str
    value: str
    matches: tuple[AttributeMatch, ...]
    events: tuple[PolicyEvent, ...]
    xml_source: str = ""


@dataclass(frozen=True, slots=True)
class _TargetPolicy:
    accepted: Mapping[str, str]
    ignored: frozenset[str]
    discoveries: tuple[_DiscoveryRule, ...]
    rejections: tuple[_RejectionFamily, ...]


@dataclass(frozen=True, slots=True)
class _RejectionFamily:
    family: str
    exact: frozenset[str]
    explanation_prefixes: tuple[str, ...] = ()
    explanation_suffixes: frozenset[str] = frozenset()
    fuzzy: bool = False

    def classify(self, normalized_value: str, presentation_value: str) -> str | None:
        if self.family == "universal_missing" and not normalized_value:
            return "rejected_value"
        if normalized_value in self.exact or self._matches_explanation(
            normalized_value, presentation_value
        ):
            return "rejected_value"
        if not self.fuzzy:
            return None
        return _fuzzy_classification(normalized_value, self.exact)

    def _matches_explanation(self, normalized_value: str, presentation_value: str) -> bool:
        allowed_suffixes = self.exact | self.explanation_suffixes
        for prefix in self.explanation_prefixes:
            if presentation_value.startswith(f"{prefix}:"):
                suffix = _normalize_rejection_value(
                    presentation_value.removeprefix(f"{prefix}:")
                )
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


class ExtractionPolicy:
    """Identify candidates using reviewed, target-scoped attribute decisions."""

    def __init__(
        self,
        targets: Mapping[str, _TargetPolicy],
        universal_missing: _RejectionFamily,
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
        if config.get("schema_version") != 1:
            raise MetadataPolicyConfigError(f"{path}: schema_version must be 1")
        targets = _mapping(path, "targets", config.get("targets"))
        unknown_targets = set(targets) - _TARGETS
        if unknown_targets:
            raise MetadataPolicyConfigError(
                f"{path}: targets has unsupported target(s): "
                f"{', '.join(sorted(str(target) for target in unknown_targets))}"
            )
        universal_missing = _compile_rejection_family(
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

    def evaluate(self, *, attribute: str, value: str) -> PolicyDecision:
        if not isinstance(attribute, str) or not isinstance(value, str):
            raise TypeError("attribute and value must be strings")
        normalized_attribute = _normalize(attribute)
        normalized_value = _normalize_rejection_value(value)
        presentation_value = _presentation_normalize(value)
        identified = [
            AttributeMatch(target, self._targets[target].accepted[normalized_attribute])
            for target in ATTRIBUTES
            if target in self._targets and normalized_attribute in self._targets[target].accepted
        ]
        location_policy = self._targets.get("loc")
        if (
            self._location_rescue is not None
            and location_policy is not None
            and normalized_attribute not in location_policy.ignored
            and not any(match.target == "loc" for match in identified)
            and self._location_rescue.matches(value)
        ):
            identified.append(AttributeMatch("loc"))
        events: list[PolicyEvent] = []
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
                PolicyEvent(
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
            target_policy = self._targets.get(target)
            if (
                target_policy is None
                or target in matched_targets
                or normalized_attribute in target_policy.ignored
            ):
                continue
            for discovery in target_policy.discoveries:
                if discovery.matches(normalized_attribute):
                    events.append(
                        PolicyEvent(
                            "unreviewed_attribute",
                            target,
                            discovery.family,
                            normalized_attribute,
                        )
                    )
                    break
        return PolicyDecision(attribute, value, tuple(matches), tuple(events))


def _load_config(path: Path) -> Mapping[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MetadataPolicyConfigError(f"{path}: cannot read policy: {error}") from error
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise MetadataPolicyConfigError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(config, Mapping):
        raise MetadataPolicyConfigError(f"{path}: top level must be a mapping")
    return config


def _compile_target(path: Path, target: str, raw_config: object) -> _TargetPolicy:
    prefix = f"targets.{target}"
    config = _mapping(path, prefix, raw_config)
    _reject_unknown_keys(path, config, {"discovery", "accept", "ignore", "reject"}, prefix)
    accept_groups = _mapping(path, f"{prefix}.accept", config.get("accept", {}))
    ignore_groups = _mapping(path, f"{prefix}.ignore", config.get("ignore", {}))
    if target == "date":
        unknown_categories = set(accept_groups) - {"sampling", "fallback"}
        if unknown_categories:
            raise MetadataPolicyConfigError(
                f"{path}: {prefix}.accept has invalid date category "
                f"{', '.join(sorted(str(item) for item in unknown_categories))!r}; "
                "expected sampling or fallback"
            )
    accepted = _accepted_decisions(path, target, accept_groups)
    ignored = frozenset(
        value for _family, value in _decision_values(path, f"{prefix}.ignore", ignore_groups)
    )
    overlap = set(accepted) & ignored
    if overlap:
        raise MetadataPolicyConfigError(
            f"{path}: {prefix} attribute {sorted(overlap)[0]!r} is both accepted and ignored"
        )
    discovery_groups = _mapping(path, f"{prefix}.discovery", config.get("discovery", {}))
    discoveries = tuple(
        _compile_discovery(path, target, family, rule) for family, rule in discovery_groups.items()
    )
    rejection_groups = _mapping(path, f"{prefix}.reject", config.get("reject", {}))
    rejections = tuple(
        _compile_rejection_family(path, f"{prefix}.reject.{family}", family, rule)
        for family, rule in rejection_groups.items()
    )
    return _TargetPolicy(accepted, ignored, discoveries, rejections)


def _compile_location_rescue(path: Path, raw_config: object) -> _LocationRescue | None:
    if raw_config is None:
        return None
    prefix = "location_rescue"
    config = _mapping(path, prefix, raw_config)
    _reject_unknown_keys(path, config, {"vocabulary_path", "aliases", "separators"}, prefix)
    raw_vocabulary_path = config.get("vocabulary_path")
    if not isinstance(raw_vocabulary_path, str) or not raw_vocabulary_path.strip():
        raise MetadataPolicyConfigError(
            f"{path}: {prefix}.vocabulary_path must be a non-empty string"
        )
    vocabulary_path = Path(raw_vocabulary_path)
    if not vocabulary_path.is_absolute():
        vocabulary_path = path.parent / vocabulary_path
    try:
        vocabulary_text = vocabulary_path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise MetadataPolicyConfigError(
            f"{path}: {prefix}.vocabulary_path cannot read {vocabulary_path}: {error}"
        ) from error
    prefixes = {
        normalized
        for line in vocabulary_text.splitlines()
        if (normalized := _presentation_normalize(line))
    }
    if not prefixes:
        raise MetadataPolicyConfigError(
            f"{path}: {prefix}.vocabulary_path {vocabulary_path} must not be empty"
        )
    for alias in _raw_string_list(path, f"{prefix}.aliases", config.get("aliases", [])):
        normalized_alias = _presentation_normalize(alias)
        if not normalized_alias:
            raise MetadataPolicyConfigError(
                f"{path}: {prefix}.aliases must contain non-empty names"
            )
        prefixes.add(normalized_alias)
    separators = tuple(_raw_string_list(path, f"{prefix}.separators", config.get("separators")))
    if not separators:
        raise MetadataPolicyConfigError(f"{path}: {prefix}.separators must not be empty")
    return _LocationRescue(tuple(sorted(prefixes, key=lambda item: (-len(item), item))), separators)


def _compile_rejection_family(
    path: Path, prefix: str, family: object, raw_rule: object
) -> _RejectionFamily:
    if not isinstance(family, str) or not family:
        raise MetadataPolicyConfigError(f"{path}: {prefix} family name must be a string")
    rule = _mapping(path, prefix, raw_rule)
    _reject_unknown_keys(
        path,
        rule,
        {"exact", "explanation_prefixes", "explanation_suffixes", "fuzzy"},
        prefix,
    )
    exact = frozenset(
        _normalize_rejection_value(value)
        for value in _raw_string_list(path, f"{prefix}.exact", rule.get("exact", []))
    )
    explanation_prefixes = tuple(
        _normalize(value)
        for value in _string_list(
            path, f"{prefix}.explanation_prefixes", rule.get("explanation_prefixes", [])
        )
    )
    explanation_suffixes = frozenset(
        _normalize_rejection_value(value)
        for value in _string_list(
            path, f"{prefix}.explanation_suffixes", rule.get("explanation_suffixes", [])
        )
    )
    fuzzy_setting = rule.get("fuzzy")
    if fuzzy_setting not in (None, "one_edit_unique"):
        raise MetadataPolicyConfigError(
            f"{path}: {prefix}.fuzzy must be 'one_edit_unique' when configured"
        )
    return _RejectionFamily(
        family,
        exact,
        explanation_prefixes,
        explanation_suffixes,
        fuzzy_setting == "one_edit_unique",
    )


def _accepted_decisions(path: Path, target: str, groups: Mapping[str, object]) -> dict[str, str]:
    categories = {"sampling": DATE_SAMPLING, "fallback": DATE_OTHER} if target == "date" else {}
    values = _decision_values(path, f"targets.{target}.accept", groups)
    result = {}
    for family, normalized in values:
        if normalized in result:
            raise MetadataPolicyConfigError(
                f"{path}: targets.{target}.accept repeats attribute {normalized!r}"
            )
        result[normalized] = categories.get(family, "")
    return result


def _decision_values(
    path: Path, prefix: str, groups: Mapping[str, object]
) -> list[tuple[str, str]]:
    result = []
    for family, raw_values in groups.items():
        if not isinstance(family, str) or not family:
            raise MetadataPolicyConfigError(f"{path}: {prefix} family names must be strings")
        for value in _string_list(path, f"{prefix}.{family}", raw_values):
            result.append((family, _normalize(value)))
    return result


def _compile_discovery(path: Path, target: str, family: object, raw_rule: object) -> _DiscoveryRule:
    prefix = f"targets.{target}.discovery.{family}"
    if not isinstance(family, str) or not family:
        raise MetadataPolicyConfigError(
            f"{path}: targets.{target}.discovery family names must be strings"
        )
    rule = _mapping(path, prefix, raw_rule)
    _reject_unknown_keys(path, rule, {"tokens", "phrases", "regex"}, prefix)
    tokens = frozenset(
        _normalize(value)
        for value in _string_list(path, f"{prefix}.tokens", rule.get("tokens", []))
    )
    phrases = tuple(
        _normalize(value)
        for value in _string_list(path, f"{prefix}.phrases", rule.get("phrases", []))
    )
    patterns = []
    for index, pattern in enumerate(
        _raw_string_list(path, f"{prefix}.regex", rule.get("regex", []))
    ):
        try:
            patterns.append(re.compile(pattern))
        except re.error as error:
            raise MetadataPolicyConfigError(
                f"{path}: {prefix}.regex[{index}] invalid regular expression {pattern!r}: {error}"
            ) from error
    return _DiscoveryRule(family, tokens, phrases, tuple(patterns))


def _target_rejection(
    target: _TargetPolicy, normalized_value: str, presentation_value: str
) -> tuple[str, str] | None:
    uncertain_family = None
    for family in target.rejections:
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
        raise MetadataPolicyConfigError(f"{path}: {prefix} must be a mapping")
    return value


def _string_list(path: Path, prefix: str, value: object) -> Sequence[str]:
    if not isinstance(value, list):
        raise MetadataPolicyConfigError(f"{path}: {prefix} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not _normalize(item):
            raise MetadataPolicyConfigError(f"{path}: {prefix}[{index}] must be a non-empty string")
    return value


def _raw_string_list(path: Path, prefix: str, value: object) -> Sequence[str]:
    if not isinstance(value, list):
        raise MetadataPolicyConfigError(f"{path}: {prefix} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise MetadataPolicyConfigError(f"{path}: {prefix}[{index}] must be a non-empty string")
    return value


def _reject_unknown_keys(
    path: Path, mapping: Mapping[str, object], allowed: set[str], prefix: str
) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise MetadataPolicyConfigError(
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
