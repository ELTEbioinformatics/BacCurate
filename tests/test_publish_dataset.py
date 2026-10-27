"""Behavior of the publish script that writes the released dataset files."""

from __future__ import annotations

import gzip
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import duckdb
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "publish_dataset.py"


@pytest.fixture(scope="module")
def publish_dataset() -> ModuleType:
    """Load the publish script as a module, the same way the command line runs it."""
    spec = importlib.util.spec_from_file_location("publish_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_writes_and_logs_only_the_served_files(
    publish_dataset: ModuleType,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    columns = ["acc", "text", "iso_val_orig", *publish_dataset.SCALAR_TYPES]
    rows = [
        [
            "SAMN1",
            "Homo sapiens",
            "blood||serum",
            "2020-01-02",
            "2020-01-03",
            "2019-06-01",
            "1.5",
            "2.5",
            "9606",
        ],
        ["SAMN2", "NA", "NA", "", "", "", "", "", ""],
    ]
    (run_dir / "run.tsv").write_text(
        "\n".join("\t".join(record) for record in (columns, *rows)) + "\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="publish_dataset"):
        publish_dataset.publish(run_dir)

    stem = f"baccurate_metadata_v{publish_dataset.package_version('baccurate')}"
    served = {f"{stem}.tsv.gz", f"{stem}.jsonl.gz", f"{stem}.parquet", "terms.tsv"}
    assert {path.name for path in run_dir.iterdir()} == {"run.tsv", *served}
    assert served <= {message.split()[0] for message in caplog.messages}
    parquet = run_dir / f"{stem}.parquet"
    schema = duckdb.sql(f"DESCRIBE SELECT * FROM '{parquet.as_posix()}'").fetchall()
    types = {row[0]: row[1] for row in schema}
    assert {column: types[column] for column in publish_dataset.SCALAR_TYPES} == dict(
        publish_dataset.SCALAR_TYPES
    )

    jsonl = gzip.decompress((run_dir / f"{stem}.jsonl.gz").read_bytes()).decode().splitlines()
    assert '"iso_val_orig":["blood","serum"]' in jsonl[0].replace(" ", "")
    assert '"iso_val_orig":[]' in jsonl[1].replace(" ", "")
    assert '"text":"Homosapiens"' in jsonl[0].replace(" ", "")
    assert '"text":null' in jsonl[1].replace(" ", "")


def test_publish_renames_the_source_database_columns_in_every_format(
    publish_dataset: ModuleType,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    columns = ["accession", "bioproject", "text", *publish_dataset.SCALAR_TYPES]
    blanks = ["" for _ in publish_dataset.SCALAR_TYPES]
    rows = [
        ["SAMN1", "PRJNA1||PRJNA2", "kept", *blanks],
        ["SAMN2", "PRJNA3", "kept", *blanks],
        ["SAMN3", "", "kept", *blanks],
    ]
    (run_dir / "run.tsv").write_text(
        "\n".join("\t".join(record) for record in (columns, *rows)) + "\n",
        encoding="utf-8",
    )

    publish_dataset.publish(run_dir)

    stem = f"baccurate_metadata_v{publish_dataset.package_version('baccurate')}"
    tsv = gzip.decompress((run_dir / f"{stem}.tsv.gz").read_bytes()).decode()
    header, *records = tsv.splitlines()
    assert header.split("\t")[:3] == ["biosample_accession", "bioproject_accessions", "text"]
    assert records[0].split("\t")[:3] == ["SAMN1", "PRJNA1||PRJNA2", "kept"]

    jsonl = gzip.decompress((run_dir / f"{stem}.jsonl.gz").read_bytes()).decode().splitlines()
    assert '"biosample_accession":"SAMN1"' in jsonl[0].replace(" ", "")
    assert '"bioproject_accessions":["PRJNA1","PRJNA2"]' in jsonl[0].replace(" ", "")
    assert '"bioproject_accessions":["PRJNA3"]' in jsonl[1].replace(" ", "")
    assert '"bioproject_accessions":null' in jsonl[2].replace(" ", "")


def test_find_dataset_rejects_a_run_directory_without_its_dataset(
    publish_dataset: ModuleType,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "other.tsv").write_text("acc\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        publish_dataset._find_dataset(run_dir)
