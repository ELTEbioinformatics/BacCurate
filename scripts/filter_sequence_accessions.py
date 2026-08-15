"""Create sequence-accession tables from retained NCBI reports.

Combines BioSample accessions from pathogen-key TSVs and AllTheBacteria metadata. It then filters retained SRA,
GenBank Assembly, and RefSeq Assembly reports for records linked to those BioSample accessions.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

# Allow package imports when a maintainer runs this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baccurate.provenance.source_snapshot import (
    SourceFile,
    SourceSnapshotManifest,
    sha256_file,
)

log = logging.getLogger("filter_sequence_accessions")

# --- Constants ---

RAW_DIR = Path("data/raw")
ID_LISTS_DIR = RAW_DIR / "id_lists"
OUTPUT_DIR = RAW_DIR / "sequence_accessions"
ATB_METADATA = RAW_DIR / "atb_2025-05.tsv"
SRA_INPUT = RAW_DIR / "SRA_Accessions.tab"
GENBANK_INPUT = RAW_DIR / "assembly_summary_genbank.txt"
REFSEQ_INPUT = RAW_DIR / "assembly_summary_refseq.txt"
SRA_REQUIRED_COLUMNS = {
    "Accession",
    "Status",
    "Type",
    "Visibility",
    "Loaded",
    "BioSample",
    "ReplacedBy",
}
ASSEMBLY_REQUIRED_COLUMNS = {"assembly_accession", "version_status", "biosample"}


@dataclass(frozen=True, slots=True)
class _SourceReport:
    """Configuration for filtering and writing one retained NCBI report."""

    label: str
    output_name: str
    sequence_accession_column: str
    manifest_path: Path
    snapshot_id_prefix: str
    provider: str
    source_url: str
    filter_report: Callable[[Path, set[str]], tuple[dict[str, set[str]], Counter[str], int, str]]


# --- Input loading ---


def _hashed_lines(path: Path, source_digest) -> Iterator[str]:
    with path.open("rb") as stream:
        for line in stream:
            source_digest.update(line)
            yield line.decode("utf-8")


def _load_pathogen_key_biosample_accessions(directory: Path) -> tuple[set[str], list[Path]]:
    biosample_accessions: set[str] = set()
    pathogen_key_tsv_paths: list[Path] = []
    for tsv_path in sorted(directory.glob("*.tsv")):
        with tsv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None or "accession" not in reader.fieldnames:
                continue
            biosample_accessions.update(row["accession"] for row in reader)
            pathogen_key_tsv_paths.append(tsv_path)
    return biosample_accessions, pathogen_key_tsv_paths


def _load_atb_biosample_accessions(metadata_path: Path) -> set[str]:
    with metadata_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return {row["accession"] for row in reader}


# --- Source filtering ---


def _filter_sra(
    source_path: Path,
    biosample_accessions: set[str],
) -> tuple[dict[str, set[str]], Counter[str], int, str]:
    # NCBI free-text fields can exceed Python's 128 KiB default.
    csv.field_size_limit(2**31 - 1)
    source_digest = hashlib.sha256()
    reader = csv.DictReader(_hashed_lines(source_path, source_digest), delimiter="\t")

    inclusion_checks = (
        ("Type != RUN", lambda row: row["Type"] == "RUN"),
        ("Status != live", lambda row: row["Status"] == "live"),
        ("Visibility != public", lambda row: row["Visibility"] == "public"),
        ("Loaded != 1", lambda row: row["Loaded"] == "1"),
        ("BioSample == -", lambda row: row["BioSample"] != "-"),
        ("ReplacedBy != -", lambda row: row["ReplacedBy"] == "-"),
        (
            "BioSample outside filter set",
            lambda row: row["BioSample"] in biosample_accessions,
        ),
    )
    exclusions: Counter[str] = Counter(dict.fromkeys((reason for reason, _ in inclusion_checks), 0))
    accepted_source_rows = 0
    sequence_accessions_by_biosample: dict[str, set[str]] = defaultdict(set)
    for row in reader:
        for exclusion_reason, is_included in inclusion_checks:
            if not is_included(row):
                exclusions[exclusion_reason] += 1
                break
        else:
            sequence_accessions_by_biosample[row["BioSample"]].add(row["Accession"])
            accepted_source_rows += 1
    return (
        sequence_accessions_by_biosample,
        exclusions,
        accepted_source_rows,
        source_digest.hexdigest(),
    )


def _filter_assembly(
    source_path: Path,
    biosample_accessions: set[str],
) -> tuple[dict[str, set[str]], Counter[str], int, str]:
    source_digest = hashlib.sha256()
    lines = _hashed_lines(source_path, source_digest)
    for line in lines:
        header = line.removeprefix("#").lstrip()
        if header.startswith("assembly_accession\t"):
            fieldnames = header.rstrip("\r\n").split("\t")
            break

    inclusion_checks = (
        ("version_status != latest", lambda row: row["version_status"] == "latest"),
        ("biosample == na", lambda row: row["biosample"] != "na"),
        (
            "biosample outside filter set",
            lambda row: row["biosample"] in biosample_accessions,
        ),
    )
    exclusions: Counter[str] = Counter(dict.fromkeys((reason for reason, _ in inclusion_checks), 0))
    accepted_source_rows = 0
    sequence_accessions_by_biosample: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(lines, fieldnames=fieldnames, delimiter="\t"):
        for exclusion_reason, is_included in inclusion_checks:
            if not is_included(row):
                exclusions[exclusion_reason] += 1
                break
        else:
            sequence_accessions_by_biosample[row["biosample"]].add(row["assembly_accession"])
            accepted_source_rows += 1
    return (
        sequence_accessions_by_biosample,
        exclusions,
        accepted_source_rows,
        source_digest.hexdigest(),
    )


# --- Output writing ---


def _write_intermediate(
    path: Path,
    sequence_accession_column: str,
    sequence_accessions_by_biosample: dict[str, set[str]],
) -> None:
    with (
        path.open("wb") as raw_stream,
        gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_stream,
    ):
        writer = csv.writer(text_stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("accession", sequence_accession_column))
        for biosample_accession in sorted(sequence_accessions_by_biosample):
            writer.writerow(
                (
                    biosample_accession,
                    ",".join(sorted(sequence_accessions_by_biosample[biosample_accession])),
                )
            )


def _write_manifest(
    path: Path,
    report: _SourceReport,
    *,
    retrieved_on: date,
    source_path: Path,
    source_sha256: str,
) -> None:
    manifest = SourceSnapshotManifest(
        snapshot_id=f"{report.snapshot_id_prefix}-{retrieved_on.isoformat()}",
        provider=report.provider,
        retrieved_on=retrieved_on,
        snapshot_as_of=None,
        source_url=report.source_url,
        file=SourceFile(name=source_path.name, sha256=source_sha256),
        notes=f"Wrote: {report.output_name}.",
    )
    path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )


def _temporary_path(directory: Path, name: str) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=f".{name}.", suffix=".tmp", delete=False
    ) as handle:
        return Path(handle.name)


def _write_source_outputs(
    *,
    report: _SourceReport,
    source_path: Path,
    output_dir: Path,
    retrieved_on: date,
    biosample_accessions: set[str],
) -> None:
    output_path = output_dir / report.output_name
    report.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_path(output_dir, report.output_name)
    temporary_manifest = _temporary_path(report.manifest_path.parent, report.manifest_path.name)
    try:
        sequence_accessions_by_biosample, exclusions, accepted_source_rows, source_sha256 = (
            report.filter_report(
                source_path,
                biosample_accessions,
            )
        )
        _write_intermediate(
            temporary_output,
            report.sequence_accession_column,
            sequence_accessions_by_biosample,
        )
        output_sha256 = sha256_file(temporary_output)
        _write_manifest(
            temporary_manifest,
            report,
            retrieved_on=retrieved_on,
            source_path=source_path,
            source_sha256=source_sha256,
        )
        log.info("%s source: %s | sha256: %s", report.label, source_path, source_sha256)
        log.info(
            "%s exclusions: %s",
            report.label,
            " | ".join(f"{key}: {value}" for key, value in exclusions.items()),
        )
        log.info("%s accepted source rows: %d", report.label, accepted_source_rows)
        log.info(
            "%s emitted BioSample accessions: %d",
            report.label,
            len(sequence_accessions_by_biosample),
        )
        log.info("%s intermediate: %s | sha256: %s", report.label, output_path, output_sha256)
        os.replace(temporary_output, output_path)
        os.replace(temporary_manifest, report.manifest_path)
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


SOURCE_REPORTS = {
    "sra": _SourceReport(
        label="SRA",
        output_name="sra_runs.tsv.gz",
        sequence_accession_column="sra_run_accessions",
        manifest_path=Path("config/sra_accessions_snapshot.yaml"),
        snapshot_id_prefix="ncbi-sra-accessions",
        provider="NCBI SRA",
        source_url="https://ftp.ncbi.nlm.nih.gov/sra/reports/Metadata/SRA_Accessions.tab",
        filter_report=_filter_sra,
    ),
    "genbank": _SourceReport(
        label="GenBank",
        output_name="genbank_assemblies.tsv.gz",
        sequence_accession_column="genbank_assembly_accessions",
        manifest_path=Path("config/assembly_summary_genbank_snapshot.yaml"),
        snapshot_id_prefix="ncbi-assembly-summary-genbank",
        provider="NCBI Assembly",
        source_url="https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_genbank.txt",
        filter_report=_filter_assembly,
    ),
    "refseq": _SourceReport(
        label="RefSeq",
        output_name="refseq_assemblies.tsv.gz",
        sequence_accession_column="refseq_assembly_accessions",
        manifest_path=Path("config/assembly_summary_refseq_snapshot.yaml"),
        snapshot_id_prefix="ncbi-assembly-summary-refseq",
        provider="NCBI Assembly",
        source_url="https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt",
        filter_report=_filter_assembly,
    ),
}


# --- Execution ---


def _setup_logging() -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        OUTPUT_DIR.parent / "filter_sequence_accessions.log",
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _close_logging() -> None:
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)


def main() -> int:
    """Filter all reports and write intermediate tables and source snapshot manifests."""
    report_paths = {
        "sra": SRA_INPUT,
        "genbank": GENBANK_INPUT,
        "refseq": REFSEQ_INPUT,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()
    try:
        for input_path in (
            ATB_METADATA,
            *report_paths.values(),
        ):
            if not input_path.is_file():
                raise FileNotFoundError(f"required input not found: {input_path}")
        if not ID_LISTS_DIR.is_dir():
            raise FileNotFoundError(f"required input directory not found: {ID_LISTS_DIR}")
        pathogen_key_biosample_accessions, pathogen_key_tsv_paths = (
            _load_pathogen_key_biosample_accessions(ID_LISTS_DIR)
        )
        atb_biosample_accessions = _load_atb_biosample_accessions(ATB_METADATA)
        biosample_accessions = pathogen_key_biosample_accessions | atb_biosample_accessions
        log.info("ATB metadata: %s | sha256: %s", ATB_METADATA, sha256_file(ATB_METADATA))
        for pathogen_key_tsv_path in pathogen_key_tsv_paths:
            log.info(
                "pathogen-key TSV: %s | sha256: %s",
                pathogen_key_tsv_path,
                sha256_file(pathogen_key_tsv_path),
            )
        log.info("BioSample accession filter set: %d", len(biosample_accessions))
        for report_key, source_path in report_paths.items():
            retrieved_on = datetime.fromtimestamp(source_path.stat().st_mtime, UTC).date()
            log.info("%s retrieved_on: %s", SOURCE_REPORTS[report_key].label, retrieved_on)
            _write_source_outputs(
                report=SOURCE_REPORTS[report_key],
                source_path=source_path,
                output_dir=OUTPUT_DIR,
                retrieved_on=retrieved_on,
                biosample_accessions=biosample_accessions,
            )
    except (OSError, ValueError) as error:
        log.error("filter failed: %s", error)
        return 1
    finally:
        _close_logging()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
