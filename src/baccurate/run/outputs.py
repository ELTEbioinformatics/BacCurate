"""Plan and manage the output files a single BacCurate run writes."""

from dataclasses import dataclass
from pathlib import Path

from baccurate.run.location_review_worklist import LOCATION_REVIEW_WORKLIST_FILENAME


@dataclass(frozen=True, slots=True)
class RunOutputs:
    """The complete set of files this run writes."""

    dataset: Path
    log: Path
    run_report: Path
    isolation_source_reasoning: Path | None
    prompt_snapshot: Path | None
    location_review_worklist: Path | None = None

    @classmethod
    def plan(
        cls,
        *,
        output_dir: Path,
        run_name: str,
        output_file: Path | None,
        include_isolation_source: bool,
        include_prompt_snapshot: bool = False,
        include_location: bool = False,
    ) -> "RunOutputs":
        if output_file is None:
            run_dir = output_dir / run_name
            dataset = run_dir / f"{run_name}.tsv"
        else:
            dataset = output_file
            run_dir = dataset.parent
        outputs = cls(
            dataset=dataset,
            log=dataset.with_suffix(".log"),
            run_report=run_dir / "run_report.json",
            isolation_source_reasoning=(
                run_dir / "isolation_source_reasoning.jsonl" if include_isolation_source else None
            ),
            prompt_snapshot=run_dir / "prompts.txt" if include_prompt_snapshot else None,
            location_review_worklist=(
                run_dir / LOCATION_REVIEW_WORKLIST_FILENAME if include_location else None
            ),
        )
        if len(set(outputs.paths())) != len(outputs.paths()):
            raise ValueError("output filename aliases another run output path")
        return outputs

    def paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (
                self.dataset,
                self.log,
                self.run_report,
                self.isolation_source_reasoning,
                self.prompt_snapshot,
                self.location_review_worklist,
            )
            if path is not None
        )

    def collision(self) -> Path | None:
        return next((path for path in self.paths() if path.exists()), None)

    def initialize(self) -> None:
        self.dataset.parent.mkdir(parents=True, exist_ok=True)
        for path in self.paths():
            path.write_text("", encoding="utf-8")
