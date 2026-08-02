"""Write the machine-readable report for a BacCurate run."""

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Literal

from baccurate.adapters.llm.diagnostics import LLMObservability
from baccurate.extraction import ExtractionReport
from baccurate.paths import (
    DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    DEFAULT_BIOPROJECT_XML_INPUT,
    REPO_ROOT,
)
from baccurate.provenance.source_snapshot import (
    SourceSnapshotError,
    SourceSnapshotManifest,
    provenance_path_for,
    sha256_file,
    validate_extracted_metadata_bundle,
)
from baccurate.run.outputs import RunOutputs
from baccurate.run.statistics import DatasetBuildProgress, DatasetBuildStatistics
from baccurate.standardization.isolation_source import IsolationSourceProvenance
from baccurate.standardization_target.specifications import TARGET_SPECS, StandardizationTarget


def _local_timestamp() -> str:
    """Return the current local time as an ISO 8601 string with UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Sanitized request and runtime inputs captured at run start."""

    requested_pathogens: tuple[str, ...]
    # These are standardization targets, not the target pathogens in the adjacent field.
    requested_standardization_targets: tuple[str, ...]
    extracted_metadata: Path
    options: Mapping[str, object]
    configuration_paths: tuple[Path, ...]
    skip_llm: bool
    model_identifiers: Mapping[str, str | None]
    trace_llm_calls: bool = False
    isolation_source_provenance: IsolationSourceProvenance | None = None


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


class RunReport:
    """Write a run report to disk, updating it as the run progresses."""

    def __init__(
        self,
        outputs: RunOutputs,
        context: RunContext,
    ) -> None:
        self._path = outputs.run_report
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
        timestamp = _local_timestamp()
        self._document: dict[str, object] = {
            "schema_version": 6,
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
                "standardization_targets": list(context.requested_standardization_targets),
            },
            "outputs": {
                "dataset": str(outputs.dataset),
                "log": str(outputs.log),
                "run_report": str(outputs.run_report),
                "extracted_input": str(context.extracted_metadata),
                "isolation_source_reasoning": (
                    str(outputs.isolation_source_reasoning)
                    if outputs.isolation_source_reasoning is not None
                    else None
                ),
                "prompt_snapshot": (
                    str(outputs.prompt_snapshot) if outputs.prompt_snapshot is not None else None
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
        if outputs.prompt_snapshot is not None:
            self._document["prompt_artifact"] = {
                "path": str(outputs.prompt_snapshot),
                "sha256": sha256_file(outputs.prompt_snapshot),
            }
        if context.isolation_source_provenance is not None:
            self._document["provenance"] = {
                "isolation_source": _stable_json_value(context.isolation_source_provenance)
            }
        self._write()

    def transition(self, phase: RunPhase) -> None:
        now = monotonic()
        self._record_phase_elapsed(now)
        self._phase_started_monotonic = now
        self._document["phase"] = phase.value
        self._document["updated_at"] = _local_timestamp()
        self._write()

    def record_performed_extraction(
        self,
        report: ExtractionReport,
        *,
        elapsed_seconds: float,
    ) -> None:
        """Record what a completed extraction produced."""
        self._update_provenance(
            _provenance_document(
                biosample_snapshot_id=report.biosample_snapshot_id,
                bioproject_snapshot_id=report.bioproject_snapshot_id,
                metadata_reference_date=report.metadata_reference_date,
            )
        )
        self._document["extraction"] = _extraction_document(
            mode="performed",
            prepared_input_paths=report.prepared_input_paths,
            extracted_metadata_path=report.extracted_metadata_path,
            elapsed_seconds=elapsed_seconds,
            extracted_record_count=report.extracted_record_count,
            curation_counts={
                "inspected_pairs": report.counters.inspected,
                "identified_pairs": report.counters.identified,
                "selected_pairs": report.counters.selected,
                "automatically_rejected_pairs": report.counters.automatically_rejected,
                "unreviewed_pairs": report.counters.unreviewed,
                "multiply_matched_pairs": report.counters.multiply_matched,
            },
            automatic_rejection_counts=report.automatic_rejection_counts,
            unreviewed_count=report.unreviewed_count,
            uncertain_count=report.uncertain_count,
            review_worklist_paths=report.review_worklist_paths,
            bundle_provenance_path=report.bundle_provenance_path,
        )
        self._document["updated_at"] = _local_timestamp()
        self._write()

    def begin_performed_extraction(
        self,
        *,
        biosample_input_path: Path,
        extracted_metadata_path: Path,
        biosample_manifest_path: Path,
        bioproject_input_path: Path = DEFAULT_BIOPROJECT_XML_INPUT,
        bioproject_manifest_path: Path = DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    ) -> None:
        """Record what is known before extraction starts."""
        try:
            biosample_manifest = SourceSnapshotManifest.load(biosample_manifest_path)
            bioproject_manifest = SourceSnapshotManifest.load(bioproject_manifest_path)
        except SourceSnapshotError:
            biosample_manifest = None
            bioproject_manifest = None
        if biosample_manifest is not None and bioproject_manifest is not None:
            self._record_provenance(biosample_manifest, bioproject_manifest)
        self._document["extraction"] = _extraction_document(
            mode="performed",
            prepared_input_paths=(biosample_input_path, bioproject_input_path),
            extracted_metadata_path=extracted_metadata_path,
            bundle_provenance_path=provenance_path_for(extracted_metadata_path),
        )
        self._document["updated_at"] = _local_timestamp()
        self._write()

    def record_reused_extraction(
        self,
        *,
        extracted_metadata_path: Path,
        biosample_manifest_path: Path,
        bioproject_manifest_path: Path = DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST,
    ) -> None:
        """Record what can be verified about a reused extracted TSV."""
        with extracted_metadata_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream, delimiter="\t")
            next(reader, None)
            extracted_record_count = sum(1 for row in reader if row)

        try:
            source_contract = validate_extracted_metadata_bundle(
                extracted_metadata_path,
                biosample_manifest_path,
                bioproject_manifest_path,
            )
            biosample_manifest = source_contract.biosample
            bioproject_manifest = source_contract.bioproject
        except SourceSnapshotError:
            # Keep both identities or neither, so a failed
            # BioProject load never leaves a half-populated provenance document.
            biosample_manifest = None
            bioproject_manifest = None
        manifests = (biosample_manifest, bioproject_manifest) if biosample_manifest else ()
        if manifests:
            self._record_provenance(biosample_manifest, bioproject_manifest)
        acquired_snapshot_files = tuple(
            snapshot_file.name for manifest in manifests for snapshot_file in manifest.files
        )
        self._document["extraction"] = _extraction_document(
            mode="reused",
            acquired_snapshot_files=acquired_snapshot_files,
            extracted_metadata_path=extracted_metadata_path,
            extracted_record_count=extracted_record_count,
            bundle_provenance_path=provenance_path_for(extracted_metadata_path),
        )
        self._document["updated_at"] = _local_timestamp()
        self._write()

    def _record_provenance(
        self,
        biosample_manifest: SourceSnapshotManifest,
        bioproject_manifest: SourceSnapshotManifest,
    ) -> None:
        self._update_provenance(
            _provenance_document(
                biosample_snapshot_id=biosample_manifest.snapshot_id,
                bioproject_snapshot_id=bioproject_manifest.snapshot_id,
                metadata_reference_date=biosample_manifest.metadata_reference_date,
            )
        )

    def _update_provenance(self, values: Mapping[str, object]) -> None:
        provenance = self._document.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise TypeError("Run report provenance must be a mapping")
        provenance.update(values)

    def finish(
        self,
        status: RunStatus,
        *,
        error: BaseException | None = None,
        progress: DatasetBuildProgress,
        statistics: DatasetBuildStatistics | None = None,
    ) -> None:
        finished_monotonic = monotonic()
        self._record_phase_elapsed(finished_monotonic)
        self._timings["total_seconds"] = round(finished_monotonic - self._started_monotonic, 6)
        extraction = self._document.get("extraction")
        if self._document["phase"] == RunPhase.EXTRACTION.value and isinstance(extraction, dict):
            extraction["elapsed_seconds"] = self._phase_seconds.get(RunPhase.EXTRACTION.value, 0.0)
        timestamp = _local_timestamp()
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
        if statistics is not None:
            self._document["scientific"] = {
                TARGET_SPECS[target].published_key: _stable_json_value(target_statistics)
                for target, target_statistics in (
                    (StandardizationTarget.DATE, statistics.date),
                    (StandardizationTarget.LOCATION, statistics.location),
                    (StandardizationTarget.HOST, statistics.host),
                    (StandardizationTarget.ISOLATION_SOURCE, statistics.isolation_source),
                )
                if target_statistics is not None
            }
        self._document["llm"] = self._llm_observability.snapshot(_llm_cache_hits(statistics))
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
    """Convert build-statistics values to stable JSON primitives."""
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


def _llm_cache_hits(statistics: DatasetBuildStatistics | None) -> dict[str, int]:
    if statistics is None:
        return {}
    return {
        TARGET_SPECS[target].published_key: target_statistics.aggregate.cache_hits
        for target, target_statistics in (
            (StandardizationTarget.LOCATION, statistics.location),
            (StandardizationTarget.ISOLATION_SOURCE, statistics.isolation_source),
        )
        if target_statistics is not None
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
    mode: Literal["performed", "reused"],
    extracted_metadata_path: Path,
    bundle_provenance_path: Path,
    prepared_input_paths: tuple[Path, ...] | None = None,
    acquired_snapshot_files: tuple[str, ...] | None = None,
    elapsed_seconds: float = 0.0,
    extracted_record_count: int | None = None,
    curation_counts: dict[str, int] | None = None,
    automatic_rejection_counts: dict[str, dict[str, int]] | None = None,
    unreviewed_count: int | None = None,
    uncertain_count: int | None = None,
    review_worklist_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    if mode not in ("performed", "reused"):
        raise ValueError(f"Unsupported extraction mode: {mode}")
    if (prepared_input_paths is None) == (acquired_snapshot_files is None):
        raise ValueError(
            "Exactly one of prepared_input_paths or acquired_snapshot_files is required"
        )
    if (mode == "performed") != (prepared_input_paths is not None):
        raise ValueError(f"Extraction mode {mode} does not match its provenance input")
    document: dict[str, object] = {"mode": mode}
    if prepared_input_paths is not None:
        document["prepared_input_paths"] = [str(path) for path in prepared_input_paths]
    else:
        document["acquired_snapshot_files"] = list(acquired_snapshot_files or ())
    document.update(
        {
            "extracted_metadata_path": str(extracted_metadata_path),
            "elapsed_seconds": elapsed_seconds,
            "extracted_record_count": extracted_record_count,
            "curation_counts": curation_counts,
            "automatic_rejection_counts": automatic_rejection_counts,
            "unreviewed_count": unreviewed_count,
            "uncertain_count": uncertain_count,
            "review_worklist_paths": {
                name: str(path) for name, path in sorted((review_worklist_paths or {}).items())
            },
            "bundle_provenance_path": str(bundle_provenance_path),
        }
    )
    return document


def _provenance_document(
    *,
    biosample_snapshot_id: str,
    bioproject_snapshot_id: str,
    metadata_reference_date: date,
) -> dict[str, object]:
    reference_date = metadata_reference_date.isoformat()
    return {
        "biosample": {
            "snapshot_id": biosample_snapshot_id,
            "metadata_reference_date": reference_date,
        },
        "bioproject": {"snapshot_id": bioproject_snapshot_id},
    }


def _diagnostic_error_message(*, error_type: str, status: RunStatus, phase: str) -> str:
    """Build a short failure summary from the exception type, without its message."""
    readable_phase = phase.replace("_", " ")
    return (
        f"{readable_phase.capitalize()} {status.value} with {error_type}; "
        "see the run log for details"
    )
