"""Standardization targets and the routing facts associated with them."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from baccurate.standardization_target.policy_slot import PolicySlot

COLLECTION_DATE_CATEGORY = "c"
FALLBACK_DATE_CATEGORY = "f"


class StandardizationTarget(StrEnum):
    """A biological metadata concept BacCurate can standardize."""

    HOST = "host"
    DATE = "date"
    LOCATION = "loc"
    ISOLATION_SOURCE = "iso"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Routing facts owned by one standardization target."""

    published_key: str
    uses_llm: bool
    required_policies: tuple[PolicySlot, ...]
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]

    @property
    def model_identifier_key(self) -> str | None:
        """Return the published key for LLM-backed targets."""
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
                "host_sci_name",
                "host_common_names",
                "host_lineage_names",
                "host_lineage_taxids",
                "host_match_quality_score",
                "host_needs_review",
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
                "date_category",
                "date_structure",
                "date_precision",
                "date_start",
                "date_end",
                "date_derivations",
                "date_attr_orig",
                "date_val_orig",
            ),
        ),
        StandardizationTarget.LOCATION: TargetSpec(
            published_key="location",
            uses_llm=False,
            required_policies=(PolicySlot.LOCATION,),
            input_columns=("loc_attr_orig", "loc_val_orig"),
            output_columns=(
                "loc_attr_orig",
                "loc_val_orig",
                "loc_selected_pair",
                "loc_resolution",
                "loc_country",
                "loc_un_region",
                "loc_sublocation",
                "loc_latitude",
                "loc_longitude",
                "loc_diagnostics",
            ),
        ),
        StandardizationTarget.ISOLATION_SOURCE: TargetSpec(
            published_key="isolation_source",
            uses_llm=True,
            required_policies=(PolicySlot.ISOLATION_SOURCE, PolicySlot.HOST),
            input_columns=("iso_attr_orig", "iso_val_orig", "host_val_orig"),
            output_columns=(
                "iso_attr_orig",
                "iso_val_orig",
                "iso_source_type",
                "iso_body_product",
                "iso_body_site",
                "iso_lesion",
                "iso_environmental_material",
                "iso_facility",
                "iso_sampled_object",
                "iso_food_type",
                "iso_term_ids",
            ),
        ),
    }
)

# This is not the enum order, but the final TSV column order.
DATASET_COLUMN_ORDER = (
    StandardizationTarget.DATE,
    StandardizationTarget.LOCATION,
    StandardizationTarget.ISOLATION_SOURCE,
    StandardizationTarget.HOST,
)


def required_policy_slots(
    targets: Iterable[StandardizationTarget],
) -> frozenset[PolicySlot]:
    """
    Return the policy slots required by the union of ``targets``.

    The frozenset supports membership testing and must not be used where order
    matters.
    """
    return frozenset(
        policy_slot for target in targets for policy_slot in TARGET_SPECS[target].required_policies
    )


def run_policy_slots(
    targets: Iterable[StandardizationTarget],
    *,
    extraction_required: bool,
) -> tuple[PolicySlot, ...]:
    """Return the policy slots one run loads, in run-report order."""
    required = required_policy_slots(targets)
    if extraction_required:
        required |= {PolicySlot.SELECTION}
    return tuple(policy_slot for policy_slot in PolicySlot if policy_slot in required)
