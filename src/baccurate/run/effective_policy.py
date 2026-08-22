"""Load only the policies needed for a BacCurate run."""

from dataclasses import dataclass
from pathlib import Path

from baccurate.extraction import SelectionSchema
from baccurate.standardization.host import HostPolicy
from baccurate.standardization.isolation_source import IsolationSourcePromptPolicy
from baccurate.standardization.location import LocationPolicy
from baccurate.standardization_target.policy_slot import POLICY_FILENAMES, PolicySlot
from baccurate.standardization_target.specifications import (
    StandardizationTarget,
    run_policy_slots,
)
from baccurate.taxon_registry.registry import TaxonRegistry


@dataclass(frozen=True, slots=True)
class EffectivePolicy:
    """
    Immutable domain policy selected for one BacCurate run.

    A selection schema is present exactly when the run must perform extraction, so a
    run that reuses an extracted bundle is never blocked by extraction policy it
    does not evaluate.
    """

    taxon_registry: TaxonRegistry
    selection_schema: SelectionSchema | None
    host_policy: HostPolicy | None
    location_policy: LocationPolicy | None
    isolation_source_prompt_policy: IsolationSourcePromptPolicy | None


def load_effective_policy(
    *,
    taxon_registry: TaxonRegistry,
    configuration_root: Path,
    requested_standardization_targets: tuple[str, ...],
    extraction_required: bool,
) -> EffectivePolicy:
    """
    Load the policy this run requires from ``configuration_root``.

    The target-taxon registry is supplied rather than loaded here because it also
    defines the accepted command keywords, so a run resolves it before it can parse
    the arguments that select the remaining policy.
    """
    targets = tuple(StandardizationTarget(target) for target in requested_standardization_targets)
    required_policies = run_policy_slots(targets, extraction_required=extraction_required)
    return EffectivePolicy(
        taxon_registry=taxon_registry,
        selection_schema=(
            SelectionSchema.load(configuration_root / POLICY_FILENAMES[PolicySlot.SELECTION_SCHEMA])
            if PolicySlot.SELECTION_SCHEMA in required_policies
            else None
        ),
        host_policy=(
            HostPolicy.load(
                configuration_root / POLICY_FILENAMES[PolicySlot.HOST],
                taxon_registry,
            )
            if PolicySlot.HOST in required_policies
            else None
        ),
        location_policy=(
            LocationPolicy.load(configuration_root / POLICY_FILENAMES[PolicySlot.LOCATION])
            if PolicySlot.LOCATION in required_policies
            else None
        ),
        isolation_source_prompt_policy=(
            IsolationSourcePromptPolicy.load(
                configuration_root / POLICY_FILENAMES[PolicySlot.ISOLATION_SOURCE]
            )
            if PolicySlot.ISOLATION_SOURCE in required_policies
            else None
        ),
    )
