"""Preserve pathogen registry behavior when Phase 1 takes ownership."""

from pathlib import Path

from baccurate.pathogen_registry.registry import (
    Pathogen,
    PathogenRegistry,
    load_pathogen_registry,
)
from baccurate.pathogen_registry.species_label_matching import (
    NA,
    build_pathogen_key_maps,
    sylph_to_pathogen_key,
)

ROOT = Path(__file__).parents[2]
PATHOGEN_REGISTRY_PATH = ROOT / "config" / "pathogens.yaml"


def _example_registry() -> PathogenRegistry:
    return PathogenRegistry(
        schema_version=1,
        target_pathogens={
            "zeta": Pathogen("zeta", "Zeta example", 30, "species", container="examples"),
            "alpha": Pathogen(
                "alpha", "Alpha", 10, "genus", container="examples", also_taxids=(11,)
            ),
        },
        containers={"examples": ("alpha", "zeta")},
    )


def test_production_target_pathogen_registry_is_preserved() -> None:
    expected = {
        "abaumannii": Pathogen("abaumannii", "Acinetobacter baumannii", 470, "species"),
        "ecoli": Pathogen("ecoli", "Escherichia coli", 562, "species", also_taxids=(620,)),
        "saureus": Pathogen("saureus", "Staphylococcus aureus", 1280, "species"),
        "paeruginosa": Pathogen("paeruginosa", "Pseudomonas aeruginosa", 287, "species"),
        "enterobacter": Pathogen("enterobacter", "Enterobacter", 547, "genus"),
        "kpneumoniae": Pathogen("kpneumoniae", "Klebsiella pneumoniae", 573, "species", "kpsc"),
        "kquasipneumoniae": Pathogen(
            "kquasipneumoniae",
            "Klebsiella quasipneumoniae",
            1463165,
            "species",
            "kpsc",
        ),
        "kvariicola": Pathogen("kvariicola", "Klebsiella variicola", 244366, "species", "kpsc"),
        "kquasivariicola": Pathogen(
            "kquasivariicola",
            "Klebsiella quasivariicola",
            2026240,
            "species",
            "kpsc",
        ),
        "kafricana": Pathogen("kafricana", "Klebsiella africana", 2489010, "species", "kpsc"),
        "efaecium": Pathogen("efaecium", "Enterococcus faecium", 1352, "species", "enterococcus"),
        "efaecalis": Pathogen(
            "efaecalis", "Enterococcus faecalis", 1351, "species", "enterococcus"
        ),
    }

    registry = load_pathogen_registry(PATHOGEN_REGISTRY_PATH)

    assert registry.schema_version == 1
    assert registry.target_pathogens == expected
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


def test_pathogen_key_table_uses_the_supplied_registry() -> None:
    registry = _example_registry()
    assert registry.pathogen_key_table() == (
        "pathogen_key\ttaxids\trank\tcontainer\n"
        "zeta\t30\tspecies\texamples\n"
        "alpha\t10 11\tgenus\texamples"
    )


def test_species_label_mappings_use_the_supplied_registry() -> None:
    genus_map, species_map = build_pathogen_key_maps(_example_registry())

    assert genus_map == {"alpha": "alpha"}
    assert species_map == {("zeta", "example"): "zeta"}
    assert {
        classification: sylph_to_pathogen_key(classification, genus_map, species_map)
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
