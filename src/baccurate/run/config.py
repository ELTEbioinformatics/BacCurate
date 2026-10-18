"""Load only the configs needed for a BacCurate run."""

from dataclasses import dataclass
from pathlib import Path

from baccurate.extraction import SelectionConfig
from baccurate.standardization.host import HostConfig
from baccurate.standardization.isolation_source_config import (
    IsolationSourceConfig,
)
from baccurate.standardization.location import LocationConfig
from baccurate.targets import (
    CONFIG_FILENAMES,
    ConfigSlot,
    StandardizationTarget,
    run_config_slots,
)
from baccurate.taxon_registry.registry import TaxonRegistry


@dataclass(frozen=True, slots=True)
class RunConfig:
    """
    Immutable domain config selected for one BacCurate run.

    A selection config is present exactly when the run must perform extraction, so a
    run that reuses extracted metadata is never blocked by extraction config it
    does not evaluate.
    """

    taxon_registry: TaxonRegistry
    selection_config: SelectionConfig | None
    host_config: HostConfig | None
    location_config: LocationConfig | None
    isolation_source_config: IsolationSourceConfig | None


def load_run_config(
    *,
    taxon_registry: TaxonRegistry,
    configuration_root: Path,
    requested_standardization_targets: tuple[str, ...],
    extraction_required: bool,
) -> RunConfig:
    """
    Load the config this run requires from ``configuration_root``.

    The target-taxon registry is supplied rather than loaded here because it also
    defines the accepted command keywords, so a run resolves it before it can parse
    the arguments that select the remaining config.
    """
    targets = tuple(StandardizationTarget(target) for target in requested_standardization_targets)
    required_configs = run_config_slots(targets, extraction_required=extraction_required)
    return RunConfig(
        taxon_registry=taxon_registry,
        selection_config=(
            SelectionConfig.load(configuration_root / CONFIG_FILENAMES[ConfigSlot.SELECTION])
            if ConfigSlot.SELECTION in required_configs
            else None
        ),
        host_config=(
            HostConfig.load(
                configuration_root / CONFIG_FILENAMES[ConfigSlot.HOST],
                taxon_registry,
            )
            if ConfigSlot.HOST in required_configs
            else None
        ),
        location_config=(
            LocationConfig.load(configuration_root / CONFIG_FILENAMES[ConfigSlot.LOCATION])
            if ConfigSlot.LOCATION in required_configs
            else None
        ),
        isolation_source_config=(
            IsolationSourceConfig.load(
                configuration_root / CONFIG_FILENAMES[ConfigSlot.ISOLATION_SOURCE]
            )
            if ConfigSlot.ISOLATION_SOURCE in required_configs
            else None
        ),
    )
