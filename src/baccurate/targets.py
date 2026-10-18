"""Standardization targets, the policy files they need, their columns and the extraction flags."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class PolicySlot(StrEnum):
    """A policy file with the rules for one pipeline step, for example `host.yaml`."""

    SELECTION = "selection"
    HOST = "host"
    LOCATION = "location"
    ISOLATION_SOURCE = "isolation_source"


POLICY_FILENAMES: Mapping[PolicySlot, str] = MappingProxyType(
    {
        PolicySlot.SELECTION: "selection.yaml",
        PolicySlot.HOST: "host.yaml",
        PolicySlot.LOCATION: "location.yaml",
        PolicySlot.ISOLATION_SOURCE: "isolation_source.yaml",
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
    """The fixed facts about one target: its output key, LLM use, policy files and columns."""

    published_key: str
    uses_llm: bool
    required_policies: tuple[PolicySlot, ...]
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
            required_policies=(PolicySlot.HOST,),
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
            required_policies=(),
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
            required_policies=(PolicySlot.LOCATION,),
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
            required_policies=(PolicySlot.ISOLATION_SOURCE, PolicySlot.HOST),
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


def required_policy_slots(
    targets: Iterable[StandardizationTarget],
) -> frozenset[PolicySlot]:
    """Return the policy files that at least one of ``targets`` needs."""
    return frozenset(
        policy_slot for target in targets for policy_slot in TARGET_SPECS[target].required_policies
    )


def run_policy_slots(
    targets: Iterable[StandardizationTarget],
    *,
    extraction_required: bool,
) -> tuple[PolicySlot, ...]:
    """Return the policy files that one run loads, in run-report order."""
    required = required_policy_slots(targets)
    if extraction_required:
        required |= {PolicySlot.SELECTION}
    return tuple(policy_slot for policy_slot in PolicySlot if policy_slot in required)
