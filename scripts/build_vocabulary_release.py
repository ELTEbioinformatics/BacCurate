"""Write the SKOS release file of the isolation-source vocabulary.

Loads the vocabulary from data/reference/ontology/ and writes
data/reference/ontology/baccurate.ttl.
"""

from __future__ import annotations

from baccurate.paths import DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY
from baccurate.standardization.isolation_source_ontology import IsolationSourceOntology
from baccurate.standardization.isolation_source_skos import build_skos_graph

RELEASE_PATH = DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY / "baccurate.ttl"


def main() -> None:
    ontology = IsolationSourceOntology.load(DEFAULT_ISOLATION_SOURCE_ONTOLOGY_DIRECTORY)
    turtle = build_skos_graph(ontology).serialize(format="turtle")
    RELEASE_PATH.write_text(turtle, encoding="utf-8", newline="\n")
    print(f"Wrote {RELEASE_PATH}")


if __name__ == "__main__":
    main()
