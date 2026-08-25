"""Write the files to be published on BacCurate.

Takes a run directory under output/ and writes 3 files named
baccurate_metadata_v<version>.<format>: TSV, JSONL, and Parquet, plus a
gzip copy of the two text formats. Output is in the same directory.

Usage:
    uv run python scripts/publish_dataset.py output/<run_dir>
"""

from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

import duckdb

# Allow package imports when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baccurate.extraction import SEQUENCE_ACCESSION_COLUMNS
from baccurate.provenance.source_snapshot import sha256_file

log = logging.getLogger("publish_dataset")

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Dataset conventions ---

LIST_COLUMNS = frozenset(
    {
        *SEQUENCE_ACCESSION_COLUMNS,
        "date_diagnostics",
        "date_attr_orig",
        "date_val_orig",
        "loc_attr_orig",
        "loc_val_orig",
        "loc_diagnostics",
        "host_common_names",
        "host_lineage_names",
        "host_lineage_taxids",
    }
)

SCALAR_TYPES = {
    "date_start": "DATE",
    "date_end": "DATE",
    "loc_latitude": "DOUBLE",
    "loc_longitude": "DOUBLE",
    "host_taxid": "INTEGER",
    "host_match_quality_score": "DOUBLE",
    "host_needs_review": "BOOLEAN",
}


def _is_list(column: str) -> bool:
    # Every isolation-source column is a list, including the facet columns
    # between iso_val_orig and iso_term_ids, which the ontology names at runtime.
    return column in LIST_COLUMNS or column.startswith("iso_")


def _column_expression(column: str) -> str:
    quoted = f'"{column}"'
    if _is_list(column):
        # A record with no standardized outcome writes "". An outcome that holds
        # no value for this field writes "NA". The published lists keep the two
        # apart as NULL against an empty list.
        return (
            f"CASE {quoted} WHEN '' THEN NULL WHEN 'NA' THEN []::VARCHAR[] "
            f"ELSE str_split({quoted}, '||') END AS {quoted}"
        )
    # A scalar column cannot hold an empty list, so both absence markers become NULL.
    value = f"NULLIF(NULLIF({quoted}, ''), 'NA')"
    cast = SCALAR_TYPES.get(column)
    return f"TRY_CAST({value} AS {cast}) AS {quoted}" if cast else f"{value} AS {quoted}"


# --- Inputs ---


def _find_dataset(run_dir: Path) -> Path:
    dataset = run_dir / f"{run_dir.name}.tsv"
    if not dataset.is_file():
        raise SystemExit(f"Cannot find the run dataset {dataset}")
    return dataset


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


# --- Outputs ---


def _compress(source: Path) -> Path:
    destination = source.with_suffix(source.suffix + ".gz")
    with source.open("rb") as raw, gzip.open(destination, "wb") as compressed:
        shutil.copyfileobj(raw, compressed, length=1024 * 1024)
    return destination


def _report(path: Path) -> None:
    log.info("%s  %.1f MB  sha256=%s", path.name, path.stat().st_size / 1e6, sha256_file(path))


def _log_lossy_casts(connection: duckdb.DuckDBPyConnection) -> None:
    checks = ", ".join(
        f"count(*) FILTER (WHERE \"{column}\" NOT IN ('', 'NA') "
        f'AND TRY_CAST("{column}" AS {cast}) IS NULL) AS "{column}"'
        for column, cast in SCALAR_TYPES.items()
    )
    counts = connection.sql(f"SELECT {checks} FROM raw").fetchone()
    for column, failures in zip(SCALAR_TYPES, counts, strict=True):
        if failures:
            log.warning("%s: %d values did not cast to %s", column, failures, SCALAR_TYPES[column])


def publish(run_dir: Path) -> None:
    dataset = _find_dataset(run_dir)
    version = package_version("baccurate")
    stem = f"baccurate_metadata_v{version}"
    tsv, jsonl, parquet = (
        run_dir / f"{stem}.{extension}" for extension in ("tsv", "jsonl", "parquet")
    )

    log.info("published at %s", datetime.now(UTC).isoformat(timespec="seconds"))
    log.info("version %s", version)
    log.info("branch %s", _git("rev-parse", "--abbrev-ref", "HEAD"))
    log.info("commit %s", _git("log", "-1", "--format=%H %cI %s"))
    log.info("source %s", dataset)

    connection = duckdb.connect()
    connection.read_csv(str(dataset), sep="\t", header=True, all_varchar=True).create_view("raw")
    columns = connection.sql("SELECT * FROM raw LIMIT 0").columns
    projection = ", ".join(_column_expression(column) for column in columns)
    connection.execute(f"CREATE VIEW published AS SELECT {projection} FROM raw")
    (rows,) = connection.sql("SELECT count(*) FROM published").fetchone()
    log.info("%d rows, %d columns", rows, len(columns))
    _log_lossy_casts(connection)

    shutil.copyfile(dataset, tsv)
    connection.execute(f"COPY published TO '{jsonl}' (FORMAT json)")
    connection.execute(f"COPY published TO '{parquet}' (FORMAT parquet, COMPRESSION zstd)")
    connection.close()

    for path in (tsv, _compress(tsv), jsonl, _compress(jsonl), parquet):
        _report(path)


def main() -> None:
    run_dir = Path(sys.argv[1])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "publish.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    publish(run_dir)


if __name__ == "__main__":
    main()
