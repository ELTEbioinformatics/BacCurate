"""Preserve taxon registry behavior when Phase 1 takes ownership."""

from pathlib import Path

from baccurate.taxon_registry.registry import (
    Taxon,
    TaxonRegistry,
    load_taxon_registry,
)
from baccurate.taxon_registry.species_label_matching import (
    NA,
    build_taxon_key_maps,
    sylph_to_taxon_key,
)

ROOT = Path(__file__).parents[2]
TAXON_REGISTRY_PATH = ROOT / "config" / "taxa.yaml"


def _example_registry() -> TaxonRegistry:
    return TaxonRegistry(
        schema_version=1,
        included_taxa={
            "zeta": Taxon("zeta", "Zeta example", 30, "species", container="examples"),
            "alpha": Taxon("alpha", "Alpha", 10, "genus", container="examples", also_taxids=(11,)),
        },
        containers={"examples": ("alpha", "zeta")},
    )


def test_production_taxon_registry_is_preserved() -> None:
    expected = {
        "abaumannii": Taxon("abaumannii", "Acinetobacter baumannii", 470, "species"),
        "ecoli": Taxon("ecoli", "Escherichia coli", 562, "species", also_taxids=(620,)),
        "saureus": Taxon("saureus", "Staphylococcus aureus", 1280, "species"),
        "paeruginosa": Taxon("paeruginosa", "Pseudomonas aeruginosa", 287, "species"),
        "enterobacter": Taxon("enterobacter", "Enterobacter", 547, "genus"),
        "kpneumoniae": Taxon("kpneumoniae", "Klebsiella pneumoniae", 573, "species", "kpsc"),
        "kquasipneumoniae": Taxon(
            "kquasipneumoniae",
            "Klebsiella quasipneumoniae",
            1463165,
            "species",
            "kpsc",
        ),
        "kvariicola": Taxon("kvariicola", "Klebsiella variicola", 244366, "species", "kpsc"),
        "kquasivariicola": Taxon(
            "kquasivariicola",
            "Klebsiella quasivariicola",
            2026240,
            "species",
            "kpsc",
        ),
        "kafricana": Taxon("kafricana", "Klebsiella africana", 2489010, "species", "kpsc"),
        "efaecium": Taxon("efaecium", "Enterococcus faecium", 1352, "species", "enterococcus"),
        "efaecalis": Taxon("efaecalis", "Enterococcus faecalis", 1351, "species", "enterococcus"),
    }

    registry = load_taxon_registry(TAXON_REGISTRY_PATH)

    assert registry.schema_version == 1
    assert registry.included_taxa == expected
    assert registry.containers == {
        "kpsc": (
            "kpneumoniae",
            "kquasipneumoniae",
            "kvariicola",
            "kquasivariicola",
            "kafricana",
        ),
        "enterococcus": ("efaecium", "efaecalis"),
    }


def test_taxon_key_table_uses_the_supplied_registry() -> None:
    registry = _example_registry()
    assert registry.taxon_key_table() == (
        "taxon_key\ttaxids\trank\tcontainer\n"
        "zeta\t30\tspecies\texamples\n"
        "alpha\t10 11\tgenus\texamples"
    )


def test_species_label_mappings_use_the_supplied_registry() -> None:
    genus_map, species_map = build_taxon_key_maps(_example_registry())

    assert genus_map == {"alpha": "alpha"}
    assert species_map == {("zeta", "example"): "zeta"}
    assert {
        classification: sylph_to_taxon_key(classification, genus_map, species_map)
        for classification in (
            "Zeta example",
            "Zeta example_A",
            "Alpha species",
            "Alpha_B species",
            "Notatarget species",
        )
    } == {
        "Zeta example": "zeta",
        "Zeta example_A": "zeta",
        "Alpha species": "alpha",
        "Alpha_B species": "alpha",
        "Notatarget species": NA,
    }
