"""Integration tests for target-pathogen registry injection into maintainer tools."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from baccurate.pathogen_registry.registry import Pathogen, PathogenRegistry

ROOT = Path(__file__).parents[2]


def _load_maintainer_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load maintainer script: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_biosample_index = _load_maintainer_script("build_biosample_index")
parse_biosample_xml = _load_maintainer_script("parse_biosample_xml")


def _example_registry() -> PathogenRegistry:
    return PathogenRegistry(
        schema_version=1,
        target_pathogens={
            "zeta": Pathogen(
                "zeta",
                "Zeta example",
                30,
                "species",
                also_taxids=(31,),
            ),
            "alpha": Pathogen("alpha", "Alpha", 10, "genus"),
        },
        pathogen_groups={"examples": ("alpha", "zeta")},
    )


def test_parse_biosample_tool_generates_registry_classified_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _example_registry()

    def load_registry() -> PathogenRegistry:
        return registry

    dump = tmp_path / "biosamples.xml"
    dump.write_text(
        '<BioSampleSet><BioSample accession="SAMN1">'
        '<Description><Organism taxonomy_id="31" taxonomy_name="Zeta example"/>'
        "</Description></BioSample></BioSampleSet>",
        encoding="utf-8",
    )
    nodes = tmp_path / "nodes.dmp"
    nodes.write_text("1 | 1 |\n30 | 1 |\n31 | 30 |\n", encoding="utf-8")
    atb = tmp_path / "atb.tsv"
    atb.write_bytes(b"accession\tsylph_species\nSAMN2\tZeta example\n")
    id_lists = tmp_path / "id_lists"
    subset = tmp_path / "subset.xml"

    monkeypatch.setattr(parse_biosample_xml, "load_pathogen_registry", load_registry)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_biosample_xml.py",
            str(dump),
            "--atb",
            str(atb),
            "--nodes-dmp",
            str(nodes),
            "--merged-dmp",
            str(tmp_path / "missing-merged.dmp"),
            "--id-lists-dir",
            str(id_lists),
            "--subset",
            str(subset),
        ],
    )

    assert parse_biosample_xml.main() == 0
    assert (id_lists / "zeta.tsv").read_text(encoding="utf-8").splitlines() == [
        "accession\ttaxid\torganism",
        "SAMN1\t31\tZeta example",
    ]
    assert 'accession="SAMN1"' in subset.read_text(encoding="utf-8")


def test_index_tool_generates_registry_classified_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _example_registry()

    def load_registry() -> PathogenRegistry:
        return registry

    id_lists = tmp_path / "id_lists"
    id_lists.mkdir()
    (id_lists / "zeta.tsv").write_text(
        "accession\ttaxid\torganism\nSAMN1\t30\tZeta example\n",
        encoding="utf-8",
    )
    (id_lists / "alpha.tsv").write_text(
        "accession\ttaxid\torganism\nSAMN1\t10\tAlpha example\n",
        encoding="utf-8",
    )
    (id_lists / "stale.tsv").write_text(
        "accession\ttaxid\torganism\nSAMN3\t999\tStale example\n",
        encoding="utf-8",
    )
    atb = tmp_path / "atb.tsv"
    atb.write_text(
        "accession\tosf_tarball_filename\tsylph_species\nSAMN2\tpart.tar\tZeta example\n",
        encoding="utf-8",
    )
    output = tmp_path / "biosample_index.tsv"

    monkeypatch.setattr(build_biosample_index, "load_pathogen_registry", load_registry)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_biosample_index.py",
            str(atb),
            "--id-lists-dir",
            str(id_lists),
            "--output",
            str(output),
        ],
    )

    assert build_biosample_index.main() == 0
    rows = pd.read_csv(output, sep="\t", dtype=str, keep_default_na=False).set_index("accession")
    assert list(rows.index) == ["SAMN1", "SAMN2"]
    assert rows.loc["SAMN1", "pathogen_biosample"] == "zeta"
    assert rows.loc["SAMN2", "pathogen_ATB"] == "zeta"
