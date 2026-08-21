"""Resolve a BacCurate command-line invocation without modifying the filesystem."""

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from baccurate.adapters.compressed_io import open_text
from baccurate.adapters.llm.client import LLMSettings, load_llm_settings
from baccurate.extraction import CurationSchema, resolve_taxon_assignment
from baccurate.paths import (
    CONFIG_DIR,
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_BIOSAMPLE_XML_INPUT,
    DEFAULT_EXTRACTED_TSV,
    DEFAULT_INDEX_TSV,
    DEFAULT_NAMES_DMP,
    DEFAULT_NODES_DMP,
    OUTPUT_DIR,
    TAXA_YAML,
)
from baccurate.run.effective_policy import EffectivePolicy, load_effective_policy
from baccurate.run.outputs import RunOutputs
from baccurate.standardization_target.policy_slot import POLICY_FILENAMES
from baccurate.standardization_target.specifications import (
    TARGET_SPECS,
    StandardizationTarget,
    run_policy_slots,
)
from baccurate.taxon_registry.registry import TaxonRegistry, load_taxon_registry
from baccurate.taxon_registry.species_label_matching import build_taxon_key_maps

STANDARDIZATION_TARGETS: tuple[StandardizationTarget, ...] = (
    StandardizationTarget.HOST,
    StandardizationTarget.DATE,
    StandardizationTarget.LOCATION,
    StandardizationTarget.ISOLATION_SOURCE,
)


@dataclass(frozen=True, slots=True)
class RunInvocation:
    """A fully resolved BacCurate run that is ready to create its outputs."""

    debug: bool
    skip_llm: bool
    names_dmp: Path
    nodes_dmp: Path
    registry: TaxonRegistry
    log_level: str
    active_targets: tuple[StandardizationTarget, ...]
    extracted_metadata_path: Path
    effective_policy: EffectivePolicy
    curation_schema: CurationSchema | None
    outputs: RunOutputs
    biosample_input_path: Path
    index_path: Path
    taxon_keys: list[str]
    extraction_taxon_keys: list[str] | None
    disable_progress: bool
    configuration_paths: list[Path]
    normalized_options: dict[str, object]
    llm_settings: LLMSettings
    model_identifiers: dict[str, str | None]
    biosample_snapshot_manifest: Path
    bioproject_snapshot_manifest: Path


def _discover_taxa(
    index_path: Path,
    taxon_registry: TaxonRegistry,
) -> list[str]:
    keys = set(taxon_registry.taxon_keys)
    genus_map, species_map = build_taxon_key_maps(taxon_registry)
    found = set()
    with open_text(index_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            assignment = resolve_taxon_assignment(row, keys, genus_map, species_map)
            if assignment is not None:
                found.add(assignment.taxon_key)
    return [taxon_key for taxon_key in taxon_registry.taxon_keys if taxon_key in found]


def _model_identifiers(
    targets: tuple[StandardizationTarget, ...],
    settings: LLMSettings,
) -> dict[str, str | None]:
    return {
        model_identifier_key: settings.model
        for target in targets
        if (model_identifier_key := TARGET_SPECS[target].model_identifier_key) is not None
    }


def resolve_invocation(
    argv: Sequence[str] | None = None,
    *,
    taxon_registry: TaxonRegistry | None = None,
) -> RunInvocation:
    arguments = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    # The registry names the accepted command keywords, so it is the one policy that
    # has to be resolved before the arguments selecting the rest can be parsed.
    registry = taxon_registry or load_taxon_registry(TAXA_YAML)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Directory containing YAML configs.",
    )
    parser.add_argument(
        "--standardize",
        dest="standardization_targets",
        nargs="*",
        choices=[target.value for target in STANDARDIZATION_TARGETS],
        help=(
            "Specific standardization targets to run. "
            "If omitted, all standardization targets are run."
        ),
    )
    parser.add_argument(
        "--extracted-metadata",
        type=Path,
        default=None,
        help=f"Path for the extracted metadata TSV (default: {DEFAULT_EXTRACTED_TSV}). ",
    )
    parser.add_argument(
        "names",
        nargs="*",
        choices=registry.keywords,
        metavar="TAXON",
        help="One or more taxa to process (see config/taxa.yaml). "
        "Taxon keys: "
        + ", ".join(registry.taxon_keys)
        + ". Containers (expand to their taxon keys): "
        + ", ".join(registry.container_keys)
        + ". If omitted, every taxon is processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base output directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Final dataset TSV (default: <output-dir>/<run-name>/<run-name>.tsv).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=("Directory name used for the output path. (default: current timestamp)."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Disable LLM client initialization and API calls.",
    )
    parser.add_argument(
        "--names-dmp",
        type=Path,
        default=DEFAULT_NAMES_DMP,
        help="NCBI names.dmp path for host lineage data generation.",
    )
    parser.add_argument(
        "--nodes-dmp",
        type=Path,
        default=DEFAULT_NODES_DMP,
        help="NCBI nodes.dmp path for host lineage data generation.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars (auto-disabled on non-TTY anyway).",
    )

    args = parser.parse_args(arguments)

    log_level = "DEBUG" if args.debug else "INFO"
    active_targets = (
        tuple(
            target
            for target in STANDARDIZATION_TARGETS
            if target.value in args.standardization_targets
        )
        if args.standardization_targets
        else STANDARDIZATION_TARGETS
    )
    extracted_metadata_path = args.extracted_metadata or DEFAULT_EXTRACTED_TSV
    # Selected policy is validated here so that invalid policy fails before this run
    # opens outputs, caches, or external services.
    effective_policy = load_effective_policy(
        taxon_registry=registry,
        configuration_root=args.config_dir,
        requested_standardization_targets=tuple(target.value for target in active_targets),
        extraction_required=not extracted_metadata_path.exists(),
    )
    curation_schema = effective_policy.curation_schema

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M")
    try:
        include_prompt_snapshot = not args.skip_llm and any(
            TARGET_SPECS[target].uses_llm for target in active_targets
        )
        outputs = RunOutputs.plan(
            output_dir=args.output_dir,
            run_name=run_name,
            output_file=args.output_file,
            include_isolation_source=StandardizationTarget.ISOLATION_SOURCE in active_targets,
            include_prompt_snapshot=include_prompt_snapshot,
            include_location=StandardizationTarget.LOCATION in active_targets,
        )
    except ValueError as exc:
        parser.error(str(exc))
    collision = outputs.collision()
    if collision is not None and not args.force:
        parser.error(f"output already exists: {collision}. Pass --force to replace it.")

    biosample_input_path = DEFAULT_BIOSAMPLE_XML_INPUT
    index_path = DEFAULT_INDEX_TSV

    names = (
        list(registry.expand(args.names)) if args.names else _discover_taxa(index_path, registry)
    )

    # Default location -> extract all taxa; explicit path -> extract only the named ones.
    extraction_names = None if args.extracted_metadata is None else names

    disable_progress = args.quiet
    configuration_paths = [
        TAXA_YAML,
        DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
        DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    ]
    # The taxon registry is resolved before argument parsing; remaining policy
    # filenames are resolved after target and extraction selection.
    configuration_paths.extend(
        args.config_dir / POLICY_FILENAMES[policy_slot]
        for policy_slot in run_policy_slots(
            active_targets,
            extraction_required=curation_schema is not None,
        )
    )
    normalized_options = {
        "taxa": list(names),
        "standardization_targets": [target.value for target in active_targets],
        "config_dir": str(args.config_dir),
        "output_dir": str(args.output_dir),
        "output_file": str(args.output_file) if args.output_file is not None else None,
        "run_name": run_name,
        "force": args.force,
        "skip_llm": args.skip_llm,
        "extracted_metadata": str(extracted_metadata_path),
        "names_dmp": str(args.names_dmp),
        "nodes_dmp": str(args.nodes_dmp),
        "debug": args.debug,
        "quiet": args.quiet,
    }
    llm_settings = load_llm_settings()
    model_identifiers = _model_identifiers(active_targets, llm_settings)
    return RunInvocation(
        debug=args.debug,
        skip_llm=args.skip_llm,
        names_dmp=args.names_dmp,
        nodes_dmp=args.nodes_dmp,
        registry=registry,
        log_level=log_level,
        active_targets=active_targets,
        extracted_metadata_path=extracted_metadata_path,
        effective_policy=effective_policy,
        curation_schema=curation_schema,
        outputs=outputs,
        biosample_input_path=biosample_input_path,
        index_path=index_path,
        taxon_keys=names,
        extraction_taxon_keys=extraction_names,
        disable_progress=disable_progress,
        configuration_paths=configuration_paths,
        normalized_options=normalized_options,
        llm_settings=llm_settings,
        model_identifiers=model_identifiers,
        biosample_snapshot_manifest=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
        bioproject_snapshot_manifest=DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    )
