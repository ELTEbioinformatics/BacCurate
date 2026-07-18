"""Plan and manage the output files a single pipeline run writes."""

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic

from baccurate.dataset_builder import DatasetBuildProgress, DatasetBuildReport
from baccurate.extraction import ExtractionReport
from baccurate.llm.diagnostics import LLMObservability
from baccurate.paths import REPO_ROOT
from baccurate.source_snapshot import (
    SourceSnapshotError,
    SourceSnapshotManifest,
    provenance_path_for,
    sha256_file,
    validate_derived_metadata_source,
)


def local_timestamp() -> str:
    """Return the current local time as an ISO 8601 string with UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class RunOutputs:
    """The complete set of files this run writes."""

    dataset: Path
    log: Path
    diagnostics: Path
    isolation_reasoning: Path | None

    @classmethod
    def plan(
        cls,
        *,
        output_dir: Path,
        run_name: str,
        output_file: Path | None,
        include_isolation: bool,
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
            diagnostics=run_dir / "diagnostics.json",
            isolation_reasoning=(
                run_dir / "isolation_reasoning.jsonl" if include_isolation else None
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
                self.diagnostics,
                self.isolation_reasoning,
            )
            if path is not None
        )

    def collision(self) -> Path | None:
        return next((path for path in self.paths() if path.exists()), None)

    def initialize(self) -> None:
        self.dataset.parent.mkdir(parents=True, exist_ok=True)
        for path in self.paths():
            path.write_text("", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Sanitized request and runtime inputs captured at run start."""

    requested_pathogens: tuple[str, ...]
    requested_attributes: tuple[str, ...]
    extracted_metadata: Path
    options: Mapping[str, object]
    configuration_paths: tuple[Path, ...]
    skip_llm: bool
    model_identifiers: Mapping[str, str | None]
    trace_llm_calls: bool = False


class RunStatus(StrEnum):
    """Persisted terminal and non-terminal run states."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunPhase(StrEnum):
    """Phases of a complete run."""

    STARTING = "starting"
    EXTRACTION = "extraction"
    DATASET_STREAMING = "dataset_streaming"
    COMPLETED = "completed"


class RunDiagnostics:
    """Write a run's status and diagnostics to disk, updating it as the run progresses."""

    def __init__(
        self,
        outputs: RunOutputs,
        context: RunContext,
    ) -> None:
        self._path = outputs.diagnostics
        self._started_monotonic = monotonic()
        self._phase_started_monotonic = self._started_monotonic
        self._llm_observability = LLMObservability(
            context.model_identifiers,
            trace_enabled=context.trace_llm_calls,
        )
        self._llm_observability.start()
        self._phase_seconds: dict[str, float] = {}
        self._timings: dict[str, object] = {
            "total_seconds": 0.0,
            "phase_seconds": self._phase_seconds,
        }
        timestamp = local_timestamp()
        self._document: dict[str, object] = {
            "schema_version": 2,
            "status": RunStatus.RUNNING.value,
            "phase": RunPhase.STARTING.value,
            "started_at": timestamp,
            "updated_at": timestamp,
            "finished_at": None,
            "error": None,
            "counters": {"processed": 0, "accepted": 0},
            "row_count": 0,
            "request": {
                "pathogens": list(context.requested_pathogens),
                "attributes": list(context.requested_attributes),
            },
            "outputs": {
                "dataset": str(outputs.dataset),
                "log": str(outputs.log),
                "diagnostics": str(outputs.diagnostics),
                "extracted_input": str(context.extracted_metadata),
                "isolation_reasoning": (
                    str(outputs.isolation_reasoning)
                    if outputs.isolation_reasoning is not None
                    else None
                ),
            },
            "timings": self._timings,
            "llm": self._llm_observability.snapshot(),
            "runtime": {
                "baccurate_version": _baccurate_version(),
                "git_commit": _git_commit(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "options": dict(context.options),
                "configuration_sha256": _configuration_hashes(context.configuration_paths),
                "skip_llm": context.skip_llm,
                "model_identifiers": dict(context.model_identifiers),
            },
        }
        self._write()

    def transition(self, phase: RunPhase) -> None:
        now = monotonic()
        self._record_phase_elapsed(now)
        self._phase_started_monotonic = now
        self._document["phase"] = phase.value
        self._document["updated_at"] = local_timestamp()
        self._write()

    def record_performed_extraction(
        self,
        report: ExtractionReport,
        *,
        elapsed_seconds: float,
    ) -> None:
        """Record what a completed extraction produced."""
        self._document["source"] = {
            "snapshot_id": report.source_snapshot_id,
            "metadata_reference_date": report.metadata_reference_date.isoformat(),
        }
        self._document["extraction"] = _extraction_document(
            mode="performed",
            source_xml_paths=report.source_xml_paths,
            extracted_metadata_path=report.extracted_metadata_path,
            elapsed_seconds=elapsed_seconds,
            extracted_record_count=report.extracted_record_count,
            curation_counts={
                "inspected_attributes": report.counters.inspected,
                "identified_candidates": report.counters.identified,
                "selected_candidates": report.counters.found,
                "automatically_rejected_candidates": report.counters.rejected,
                "unreviewed_candidates": report.counters.unreviewed,
                "multiply_identified_candidates": report.counters.multiply_matched,
            },
            automatic_rejection_counts=report.automatic_rejection_counts,
            unreviewed_count=report.unreviewed_count,
            uncertain_count=report.uncertain_count,
            review_artifact_paths=report.review_artifact_paths,
            source_record_path=report.source_record_path,
        )
        self._document["updated_at"] = local_timestamp()
        self._write()

    def begin_performed_extraction(
        self,
        *,
        source_xml_path: Path,
        extracted_metadata_path: Path,
        source_manifest_path: Path,
    ) -> None:
        """Record what is known before extraction starts."""
        try:
            manifest = SourceSnapshotManifest.load(source_manifest_path)
        except SourceSnapshotError:
            manifest = None
        if manifest is not None:
            self._record_source(manifest)
        self._document["extraction"] = _extraction_document(
            mode="performed",
            source_xml_paths=(source_xml_path,),
            extracted_metadata_path=extracted_metadata_path,
            source_record_path=provenance_path_for(extracted_metadata_path),
        )
        self._document["updated_at"] = local_timestamp()
        self._write()

    def record_reused_extraction(
        self,
        *,
        extracted_metadata_path: Path,
        source_manifest_path: Path,
    ) -> None:
        """Record what can be verified about a reused extracted TSV."""
        with extracted_metadata_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream, delimiter="\t")
            next(reader, None)
            extracted_record_count = sum(1 for row in reader if row)

        try:
            manifest = validate_derived_metadata_source(
                extracted_metadata_path,
                source_manifest_path,
            )
        except SourceSnapshotError:
            manifest = None
        if manifest is not None:
            self._record_source(manifest)
        source_xml_paths = tuple(Path(source.name) for source in manifest.files) if manifest else ()
        self._document["extraction"] = _extraction_document(
            mode="reused",
            source_xml_paths=source_xml_paths,
            extracted_metadata_path=extracted_metadata_path,
            extracted_record_count=extracted_record_count,
            source_record_path=provenance_path_for(extracted_metadata_path),
        )
        self._document["updated_at"] = local_timestamp()
        self._write()

    def _record_source(self, manifest: SourceSnapshotManifest) -> None:
        self._document["source"] = {
            "snapshot_id": manifest.snapshot_id,
            "metadata_reference_date": manifest.metadata_reference_date.isoformat(),
        }

    def finish(
        self,
        status: RunStatus,
        *,
        error: BaseException | None = None,
        progress: DatasetBuildProgress,
        report: DatasetBuildReport | None = None,
    ) -> None:
        finished_monotonic = monotonic()
        self._record_phase_elapsed(finished_monotonic)
        self._timings["total_seconds"] = round(finished_monotonic - self._started_monotonic, 6)
        extraction = self._document.get("extraction")
        if self._document["phase"] == RunPhase.EXTRACTION.value and isinstance(extraction, dict):
            extraction["elapsed_seconds"] = self._phase_seconds.get(RunPhase.EXTRACTION.value, 0.0)
        timestamp = local_timestamp()
        self._document["status"] = status.value
        self._document["updated_at"] = timestamp
        self._document["finished_at"] = timestamp
        self._document["counters"] = {
            "processed": progress.processed_rows,
            "accepted": progress.rows_written,
        }
        self._document["row_count"] = progress.rows_written
        if status is RunStatus.SUCCEEDED:
            self._document["phase"] = RunPhase.COMPLETED.value
        if error is not None:
            self._document["error"] = {
                "type": type(error).__name__,
                "message": _diagnostic_error_message(
                    error_type=type(error).__name__,
                    status=status,
                    phase=str(self._document["phase"]),
                ),
            }
        if report is not None:
            self._document["scientific"] = {
                name: _stable_json_value(attribute_report)
                for name, attribute_report in (
                    ("date", report.date),
                    ("location", report.location),
                    ("host", report.host),
                    ("isolation", report.isolation),
                )
                if attribute_report is not None
            }
        self._document["llm"] = self._llm_observability.snapshot(_llm_cache_hits(report))
        try:
            self._write()
        finally:
            self._llm_observability.close()

    def _record_phase_elapsed(self, now: float) -> None:
        phase = str(self._document["phase"])
        elapsed = now - self._phase_started_monotonic
        self._phase_seconds[phase] = round(self._phase_seconds.get(phase, 0.0) + elapsed, 6)

    def _write(self) -> None:
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        temporary.write_text(
            json.dumps(self._document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self._path)


def _stable_json_value(value):
    """Convert build-report values to stable JSON primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _stable_json_value(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            getattr(key, "value", str(key)): _stable_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_json_value(item) for item in value]
    return getattr(value, "value", value)


def processed_rows(report: DatasetBuildReport) -> int:
    """Return the processed-row count exposed by attribute reports."""
    counts = [
        attribute_report.aggregate.processed
        for attribute_report in (report.date, report.location, report.host, report.isolation)
        if attribute_report is not None
    ]
    return max(counts, default=report.rows_written)


def _llm_cache_hits(report: DatasetBuildReport | None) -> dict[str, int]:
    if report is None:
        return {}
    return {
        attribute: attribute_report.aggregate.cache_hits
        for attribute, attribute_report in (
            ("location", report.location),
            ("isolation", report.isolation),
        )
        if attribute_report is not None
    }


def _baccurate_version() -> str:
    try:
        return importlib.metadata.version("baccurate")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else None


def _configuration_hashes(paths: tuple[Path, ...]) -> dict[str, str | None]:
    hashes = {}
    for path in sorted(set(paths), key=str):
        try:
            hashes[str(path)] = sha256_file(path)
        except OSError:
            hashes[str(path)] = None
    return hashes


def _extraction_document(
    *,
    mode: str,
    source_xml_paths: tuple[Path, ...],
    extracted_metadata_path: Path,
    elapsed_seconds: float = 0.0,
    extracted_record_count: int | None = None,
    curation_counts: dict[str, int] | None = None,
    automatic_rejection_counts: dict[str, dict[str, int]] | None = None,
    unreviewed_count: int | None = None,
    uncertain_count: int | None = None,
    review_artifact_paths: Mapping[str, Path] | None = None,
    source_record_path: Path,
) -> dict[str, object]:
    return {
        "mode": mode,
        "source_xml_paths": [str(path) for path in source_xml_paths],
        "extracted_metadata_path": str(extracted_metadata_path),
        "elapsed_seconds": elapsed_seconds,
        "extracted_record_count": extracted_record_count,
        "curation_counts": curation_counts,
        "automatic_rejection_counts": automatic_rejection_counts,
        "unreviewed_count": unreviewed_count,
        "uncertain_count": uncertain_count,
        "review_artifact_paths": {
            name: str(path) for name, path in sorted((review_artifact_paths or {}).items())
        },
        "source_record_path": str(source_record_path),
    }


def _diagnostic_error_message(*, error_type: str, status: RunStatus, phase: str) -> str:
    """Build a short failure summary from the exception type, without its message."""
    readable_phase = phase.replace("_", " ")
    return (
        f"{readable_phase.capitalize()} {status.value} with {error_type}; "
        "see the run log for details"
    )
