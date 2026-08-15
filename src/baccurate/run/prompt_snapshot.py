"""Record the exact prompts a run's LLM pipelines send, so a finished run can be
audited or reproduced from its output directory without rerunning it."""

from collections.abc import Mapping
from pathlib import Path

from baccurate.standardization.isolation_source import IsolationSourcePromptPolicy
from baccurate.standardization_target.specifications import TARGET_SPECS, StandardizationTarget


def write_prompt_snapshot(
    destination: Path,
    *,
    model_identifiers: Mapping[str, str | None],
    isolation_source_prompt_policy: IsolationSourcePromptPolicy | None = None,
) -> None:
    """Write one section per prompt-backed target to ``destination``.

    Only isolation-source uses prompts. Each section records the model identifier,
    prompt_version, and the exact system/user prompts sent to the LLM.
    """
    sections = []
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
    prompt_version: str,
    prompts: tuple[tuple[str, str], ...],
) -> str:
    metadata = [
        f"[{name}]",
        f"model_identifier: {model_identifier or ''}",
        f"prompt_version: {prompt_version}",
    ]
    fields = [f"{field} ({len(value)} characters):\n{value}" for field, value in prompts]
    return "\n".join(metadata) + "\n" + "\n".join(fields)
