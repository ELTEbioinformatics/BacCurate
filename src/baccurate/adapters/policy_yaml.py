from collections.abc import Mapping
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode


class PolicyConfigurationError(ValueError):
    """An error in YAML syntax or policy structure."""


def _configuration_error(path: Path, policy_key: str, message: str) -> PolicyConfigurationError:
    return PolicyConfigurationError(f"{path}: {policy_key}: {message}")


def _reject_duplicate_keys(path: Path, node: Node, key_path: tuple[str, ...] = ()) -> None:
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = key_node.value if isinstance(key_node, ScalarNode) else "<non-scalar-key>"
            current_path = (*key_path, key)
            if key in seen:
                raise _configuration_error(
                    path,
                    ".".join(current_path),
                    "duplicate YAML key",
                )
            seen.add(key)
            _reject_duplicate_keys(path, value_node, current_path)
    elif isinstance(node, SequenceNode):
        for index, value_node in enumerate(node.value):
            _reject_duplicate_keys(path, value_node, (*key_path, str(index)))


def load_policy_mapping(path: Path | str) -> Mapping[object, object]:
    """Load one YAML mapping with duplicate-key and source-aware syntax checks."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        node = yaml.compose(text)
        if node is not None:
            _reject_duplicate_keys(source, node)
        value = yaml.safe_load(text)
    except PolicyConfigurationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise _configuration_error(source, "<yaml>", str(error)) from error

    if not isinstance(value, Mapping):
        raise _configuration_error(source, "<root>", "must be a YAML mapping")
    return value
