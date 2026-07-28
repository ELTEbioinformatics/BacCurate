"""Record the exact prompts a run's LLM pipelines send, so a finished run can be
audited or reproduced from its output directory without rerunning it."""

from collections.abc import Mapping
from pathlib import Path

from baccurate.standardization.isolation_source import IsolationSourcePromptPolicy
from baccurate.standardization.location import LocationPolicy
from baccurate.standardization_target.specifications import TARGET_SPECS, StandardizationTarget


def write_prompt_snapshot(
    destination: Path,
    *,
    model_identifiers: Mapping[str, str | None],
    location_policy: LocationPolicy | None = None,
    isolation_source_prompt_policy: IsolationSourcePromptPolicy | None = None,
) -> None:
    """Write one section per selected standardization target to ``destination``.

    The supported prompt-backed targets are geographic location and isolation source.

    Each section holds:
    - model identifier
    - declared prompt_version
    - the exact system prompt and user-prompt template that is sent to the LLM
    """
    sections = []
    if location_policy is not None:
        prompts = location_policy.prompts
        sections.append(
            _section(
                "location",
                model_identifiers.get("location"),
                location_policy.prompt_version,
                (
                    ("system_prompt", prompts.system),
                    ("user_prompt_template", prompts.user_template),
                ),
            )
        )
    if isolation_source_prompt_policy is not None:
        prompts = isolation_source_prompt_policy.effective_prompts
        target_spec = TARGET_SPECS[StandardizationTarget.ISOLATION_SOURCE]
        sections.append(
            _section(
                target_spec.published_key,
                model_identifiers.get(target_spec.published_key),
                isolation_source_prompt_policy.prompt_version,
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
    prompt_version: object | None,
    prompts: tuple[tuple[str, str], ...],
) -> str:
    metadata = [f"[{name}]", f"model_identifier: {model_identifier or ''}"]
    if prompt_version is not None:
        metadata.append(f"prompt_version: {prompt_version}")
    fields = [f"{field} ({len(value)} characters):\n{value}" for field, value in prompts]
    return "\n".join(metadata) + "\n" + "\n".join(fields)
