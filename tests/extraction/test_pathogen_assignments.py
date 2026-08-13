"""Tests for target pathogen assignment from data/raw/biosample_index.tsv."""
from pathlib import Path

from baccurate.extraction import TargetPathogenAssignment, load_pathogen_map


def test_load_pathogen_map_resolves_registered_assignments_before_filtering(
    tmp_path: Path,
) -> None:
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\tpathogen_biosample\tpathogen_ATB\n"
        "taxonomy-only\tecoli\tNA\n"
        "atb-only\tNA\tefaecium\n"
        "matching-dual\tecoli\tecoli\n"
        "conflicting-dual\tecoli\tefaecium\n"
        "unknown\tnot-registered\talso-not-registered\n"
        "fallback-from-unknown\tnot-registered\tefaecium\n",
        encoding="utf-8",
    )

    assignments = load_pathogen_map(index, {"ecoli", "efaecium"})

    assert assignments == {
        "taxonomy-only": TargetPathogenAssignment("ecoli", "biosample_taxonomy"),
        "atb-only": TargetPathogenAssignment("efaecium", "allthebacteria"),
        "matching-dual": TargetPathogenAssignment("ecoli", "biosample_taxonomy"),
        "conflicting-dual": TargetPathogenAssignment("ecoli", "biosample_taxonomy"),
        "fallback-from-unknown": TargetPathogenAssignment("efaecium", "allthebacteria"),
    }


def test_load_pathogen_map_filters_the_resolved_assignment(
    tmp_path: Path,
) -> None:
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        "accession\tpathogen_biosample\tpathogen_ATB\n"
        "selected-by-taxonomy\tecoli\tefaecium\n"
        "selected-by-atb\tNA\tecoli\n"
        "excluded-after-resolution\tefaecium\tecoli\n",
        encoding="utf-8",
    )

    assignments = load_pathogen_map(index, {"ecoli", "efaecium"}, names=["ecoli"])

    assert assignments == {
        "selected-by-taxonomy": TargetPathogenAssignment("ecoli", "biosample_taxonomy"),
        "selected-by-atb": TargetPathogenAssignment("ecoli", "allthebacteria"),
    }
