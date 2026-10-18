"""Standardization targets, the config files they need, their columns and the extraction flags."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ConfigSlot(StrEnum):
    """A config file with the rules for one pipeline step, for example `host.yaml`."""

    SELECTION = "selection"
    HOST = "host"
    LOCATION = "location"
    ISOLATION_SOURCE = "isolation_source"


CONFIG_FILENAMES: Mapping[ConfigSlot, str] = MappingProxyType(
    {
        ConfigSlot.SELECTION: "selection.yaml",
        ConfigSlot.HOST: "host.yaml",
        ConfigSlot.LOCATION: "location.yaml",
        ConfigSlot.ISOLATION_SOURCE: "isolation_source.yaml",
    }
)

COLLECTION_DATE_CATEGORY = "c"
FALLBACK_DATE_CATEGORY = "f"

LOCATION_NAME_MATCH = "name"
LOCATION_VALUE_MATCH = "value"


class StandardizationTarget(StrEnum):
    """A biological metadata concept BacCurate can standardize."""

    HOST = "host"
    DATE = "date"
    LOCATION = "loc"
    ISOLATION_SOURCE = "iso"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """The fixed facts about one target: its output key, LLM use, config files and columns."""

    published_key: str
    uses_llm: bool
    required_configs: tuple[ConfigSlot, ...]
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...] = ()

    @property
    def model_identifier_key(self) -> str | None:
        """Return the key for the LLM model name, or None for a target without an LLM."""
        return self.published_key if self.uses_llm else None


TARGET_SPECS: Mapping[StandardizationTarget, TargetSpec] = MappingProxyType(
    {
        StandardizationTarget.HOST: TargetSpec(
            published_key="host",
            uses_llm=False,
            required_configs=(ConfigSlot.HOST,),
            input_columns=("host_attr_orig", "host_val_orig"),
            output_columns=(
                "host_attr_orig",
                "host_val_orig",
                "host_taxid",
                "host_scientific_name",
                "host_common_names",
                "host_lineage_names",
                "host_lineage_taxids",
                "host_match_route",
                "host_diagnostics",
            ),
        ),
        StandardizationTarget.DATE: TargetSpec(
            published_key="date",
            uses_llm=False,
            required_configs=(),
            input_columns=(
                "biosample_last_update",
                "date_attr_orig",
                "date_val_orig",
                "date_category",
            ),
            output_columns=(
                "date_attr_orig",
                "date_val_orig",
                "date_start",
                "date_end",
                "date_record",
                "date_precision",
                "date_structure",
                "date_diagnostics",
            ),
        ),
        StandardizationTarget.LOCATION: TargetSpec(
            published_key="location",
            uses_llm=False,
            required_configs=(ConfigSlot.LOCATION,),
            input_columns=("loc_attr_orig", "loc_val_orig", "loc_matched_by"),
            output_columns=(
                "loc_attr_orig",
                "loc_val_orig",
                "loc_country",
                "loc_un_region",
                "loc_sublocation",
                "loc_latitude",
                "loc_longitude",
                "loc_resolution",
                "loc_diagnostics",
            ),
        ),
        StandardizationTarget.ISOLATION_SOURCE: TargetSpec(
            published_key="isolation_source",
            uses_llm=True,
            required_configs=(ConfigSlot.ISOLATION_SOURCE, ConfigSlot.HOST),
            input_columns=("iso_attr_orig", "iso_val_orig", "host_val_orig"),
        ),
    }
)

# This is not the enum order, but the final TSV column order.
# Every target block reads the same way: the submitted pairs, the standardized answer,
# the lookup that produced it, then the diagnostics.
DATASET_COLUMN_ORDER = (
    StandardizationTarget.DATE,
    StandardizationTarget.LOCATION,
    StandardizationTarget.ISOLATION_SOURCE,
    StandardizationTarget.HOST,
)


def required_config_slots(
    targets: Iterable[StandardizationTarget],
) -> frozenset[ConfigSlot]:
    """Return the config files that at least one of ``targets`` needs."""
    return frozenset(
        config_slot for target in targets for config_slot in TARGET_SPECS[target].required_configs
    )


def run_config_slots(
    targets: Iterable[StandardizationTarget],
    *,
    extraction_required: bool,
) -> tuple[ConfigSlot, ...]:
    """Return the config files that one run loads, in run-report order."""
    required = required_config_slots(targets)
    if extraction_required:
        required |= {ConfigSlot.SELECTION}
    return tuple(config_slot for config_slot in ConfigSlot if config_slot in required)
