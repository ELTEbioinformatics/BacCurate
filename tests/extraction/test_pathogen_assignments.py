"""Tests for target pathogen assignment from data/raw/biosample_index.tsv."""

from pathlib import Path

from baccurate.extraction import TargetPathogenAssignment, load_pathogen_map

_HEADER = "accession\tpathogen_biosample\tpathogen_ATB\torganism_value\tsylph_species\n"


def test_load_pathogen_map_resolves_registered_assignments_before_filtering(
    tmp_path: Path,
) -> None:
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        _HEADER
        + "taxonomy-only\tecoli\tNA\tEscherichia coli\tNA\n"
        + "atb-only\tNA\tefaecium\tNA\tEnterococcus_B faecium\n"
        + "agreeing-dual\tecoli\tecoli\tEscherichia coli\tEscherichia coli\n"
        + "disagreeing-dual\tecoli\tefaecium\tEscherichia coli\tEnterococcus_B faecium\n"
        + "unknown\tnot-registered\talso-not-registered\tOther species\tOther species\n"
        + "off-target-sylph\tefaecium\tNA\tEnterococcus faecium\tEnterococcus_B lactis\n",
        encoding="utf-8",
    )

    assignments = load_pathogen_map(index, {"ecoli", "efaecium"})

    assert assignments == {
        "taxonomy-only": TargetPathogenAssignment(
            "ecoli", "biosample_taxonomy", ncbi_organism="Escherichia coli", sylph_species="NA"
        ),
        "atb-only": TargetPathogenAssignment(
            "efaecium", "allthebacteria", ncbi_organism="NA", sylph_species="Enterococcus_B faecium"
        ),
        # Record keeps the NCBI taxonomy route on agreement.
        "agreeing-dual": TargetPathogenAssignment(
            "ecoli",
            "biosample_taxonomy",
            ncbi_organism="Escherichia coli",
            sylph_species="Escherichia coli",
        ),
        # The sylph call wins across a genus, and the NCBI call is written alongside it.
        "disagreeing-dual": TargetPathogenAssignment(
            "efaecium",
            "allthebacteria",
            ncbi_organism="Escherichia coli",
            sylph_species="Enterococcus_B faecium",
        ),
        # A sylph call outside the registry does not remove a record.
        "off-target-sylph": TargetPathogenAssignment(
            "efaecium",
            "biosample_taxonomy",
            ncbi_organism="Enterococcus faecium",
            sylph_species="Enterococcus_B lactis",
        ),
    }


def test_load_pathogen_map_filters_the_resolved_assignment(
    tmp_path: Path,
) -> None:
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        _HEADER
        + "selected-by-atb\tefaecium\tecoli\tEnterococcus faecium\tEscherichia coli\n"
        + "selected-by-taxonomy\tecoli\tNA\tEscherichia coli\tNA\n"
        + "excluded-after-resolution\tecoli\tefaecium\tEscherichia coli\tEnterococcus_B faecium\n",
        encoding="utf-8",
    )

    assignments = load_pathogen_map(index, {"ecoli", "efaecium"}, names=["ecoli"])

    assert assignments == {
        "selected-by-atb": TargetPathogenAssignment(
            "ecoli",
            "allthebacteria",
            ncbi_organism="Enterococcus faecium",
            sylph_species="Escherichia coli",
        ),
        "selected-by-taxonomy": TargetPathogenAssignment(
            "ecoli", "biosample_taxonomy", ncbi_organism="Escherichia coli", sylph_species="NA"
        ),
    }
