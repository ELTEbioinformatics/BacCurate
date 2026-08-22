"""Tests for taxon assignment from data/raw/biosample_index.tsv."""

from pathlib import Path

from baccurate.extraction import TaxonAssignment, load_taxon_map
from baccurate.taxon_registry.registry import Taxon, TaxonRegistry

_HEADER = "accession\ttaxon_biosample\torganism_value\tsylph_species\n"

_ECOLI = Taxon("ecoli", "Escherichia coli", 562, "species")
_EFAECIUM = Taxon("efaecium", "Enterococcus faecium", 1352, "species")


def _registry(*included_taxa: Taxon) -> TaxonRegistry:
    return TaxonRegistry(
        schema_version=1,
        included_taxa={taxon.key: taxon for taxon in included_taxa},
        containers={},
    )


def test_load_taxon_map_resolves_registered_assignments_before_filtering(
    tmp_path: Path,
) -> None:
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        _HEADER
        + "taxonomy-only\tecoli\tEscherichia coli\tNA\n"
        + "atb-only\tNA\tNA\tEnterococcus_B faecium\n"
        + "agreeing-dual\tecoli\tEscherichia coli\tEscherichia coli\n"
        + "disagreeing-dual\tecoli\tEscherichia coli\tEnterococcus_B faecium\n"
        + "unknown\tnot-registered\tOther species\tOther species\n"
        + "off-target-sylph\tefaecium\tEnterococcus faecium\tEnterococcus_B lactis\n",
        encoding="utf-8",
    )

    assignments = load_taxon_map(index, _registry(_ECOLI, _EFAECIUM))

    assert assignments == {
        "taxonomy-only": TaxonAssignment("ecoli", "biosample_taxonomy", sylph_species="NA"),
        "atb-only": TaxonAssignment(
            "efaecium", "allthebacteria", sylph_species="Enterococcus_B faecium"
        ),
        # Record keeps the NCBI taxonomy route on agreement.
        "agreeing-dual": TaxonAssignment(
            "ecoli",
            "biosample_taxonomy",
            sylph_species="Escherichia coli",
        ),
        # The sylph call wins across a genus.
        "disagreeing-dual": TaxonAssignment(
            "efaecium",
            "allthebacteria",
            sylph_species="Enterococcus_B faecium",
        ),
        # A sylph call outside the registry does not remove a record.
        "off-target-sylph": TaxonAssignment(
            "efaecium",
            "biosample_taxonomy",
            sylph_species="Enterococcus_B lactis",
        ),
    }


def test_load_taxon_map_filters_the_resolved_assignment(
    tmp_path: Path,
) -> None:
    index = tmp_path / "biosample_index.tsv"
    index.write_text(
        _HEADER
        + "selected-by-atb\tefaecium\tEnterococcus faecium\tEscherichia coli\n"
        + "selected-by-taxonomy\tecoli\tEscherichia coli\tNA\n"
        + "excluded-after-resolution\tecoli\tEscherichia coli\tEnterococcus_B faecium\n",
        encoding="utf-8",
    )

    assignments = load_taxon_map(index, _registry(_ECOLI, _EFAECIUM), names=["ecoli"])

    assert assignments == {
        "selected-by-atb": TaxonAssignment(
            "ecoli",
            "allthebacteria",
            sylph_species="Escherichia coli",
        ),
        "selected-by-taxonomy": TaxonAssignment("ecoli", "biosample_taxonomy", sylph_species="NA"),
    }
