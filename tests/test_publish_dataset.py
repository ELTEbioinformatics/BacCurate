"""Behavior of the publish script that writes the released dataset files."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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


def test_column_expression_covers_list_scalar_and_text_columns(
    publish_dataset: ModuleType,
) -> None:
    list_column = publish_dataset._column_expression("iso_val_orig")
    assert "str_split" in list_column
    assert "'NA' THEN []::VARCHAR[]" in list_column

    scalar_column = publish_dataset._column_expression("date_start")
    assert scalar_column == (
        "TRY_CAST(NULLIF(NULLIF(\"date_start\", ''), 'NA') AS DATE) AS \"date_start\""
    )

    text_column = publish_dataset._column_expression("acc")
    assert text_column == "NULLIF(NULLIF(\"acc\", ''), 'NA') AS \"acc\""


def test_publish_writes_the_five_release_files(
    publish_dataset: ModuleType,
    tmp_path: Path,
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
            "1.5",
            "2.5",
            "9606",
            "0.9",
        ],
        ["SAMN2", "NA", "NA", "", "", "", "", "", ""],
    ]
    (run_dir / "run.tsv").write_text(
        "\n".join("\t".join(record) for record in (columns, *rows)) + "\n",
        encoding="utf-8",
    )

    publish_dataset.publish(run_dir)

    stem = f"baccurate_metadata_v{publish_dataset.package_version('baccurate')}"
    for name in (f"{stem}.tsv", f"{stem}.tsv.gz", f"{stem}.jsonl", f"{stem}.jsonl.gz"):
        assert (run_dir / name).is_file()
    parquet = run_dir / f"{stem}.parquet"
    assert parquet.is_file()

    jsonl = (run_dir / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"iso_val_orig":["blood","serum"]' in jsonl[0].replace(" ", "")
    assert '"iso_val_orig":[]' in jsonl[1].replace(" ", "")
    assert '"text":"Homosapiens"' in jsonl[0].replace(" ", "")
    assert '"text":null' in jsonl[1].replace(" ", "")


def test_find_dataset_rejects_a_run_directory_without_its_dataset(
    publish_dataset: ModuleType,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "other.tsv").write_text("acc\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        publish_dataset._find_dataset(run_dir)
