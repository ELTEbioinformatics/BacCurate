"""Record the exact prompts a run's LLM pipelines send, so a finished run can be
audited or reproduced from its output directory without rerunning it."""

from collections.abc import Mapping
from pathlib import Path

from baccurate.paths import DEFAULT_ONTOLOGY_TSV
from baccurate.standardizers.isolation import (
    OntologyManager,
    effective_isolation_prompts,
)
from baccurate.standardizers.location import effective_location_prompts
from baccurate.utils.config import load_config


def write_prompt_snapshot(
    destination: Path,
    *,
    model_identifiers: Mapping[str, str | None],
    location_config: Path | None = None,
    isolation_config: Path | None = None,
) -> None:
    """Write one section per selected pipeline (location, isolation) to 'destination'.

    Each section holds:
    - model identifier
    - declared prompt_version
    - the exact system prompt and user-prompt template that is sent to the LLM
    """
    sections = []
    if location_config is not None:
        config = load_config(location_config)
        prompts = effective_location_prompts(config)
        sections.append(
            _section(
                "location",
                model_identifiers.get("location"),
                config,
                (
                    ("system_prompt", prompts.system),
                    ("user_prompt_template", prompts.user_template),
                ),
            )
        )
    if isolation_config is not None:
        config = load_config(isolation_config)
        ontology = OntologyManager(config.get("ontology_tsv_path", DEFAULT_ONTOLOGY_TSV))
        prompts = effective_isolation_prompts(config, ontology)
        sections.append(
            _section(
                "isolation",
                model_identifiers.get("isolation"),
                config,
                (
                    ("system_prompt", prompts.system),
                    ("user_prompt_template", prompts.user_template),
                    ("bioproject_system_prompt", prompts.bioproject_system),
                    ("bioproject_user_prompt", prompts.bioproject_user),
                ),
            )
        )
    destination.write_text(
        "Prompts\n\n" + "\n\n".join(sections) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _section(
    name: str,
    model_identifier: str | None,
    config: Mapping[str, object],
    prompts: tuple[tuple[str, str], ...],
) -> str:
    metadata = [f"[{name}]", f"model_identifier: {model_identifier or ''}"]
    if "prompt_version" in config:
        metadata.append(f"prompt_version: {config['prompt_version']}")
    fields = [f"{field} ({len(value)} characters):\n{value}" for field, value in prompts]
    return "\n".join(metadata) + "\n" + "\n".join(fields)
