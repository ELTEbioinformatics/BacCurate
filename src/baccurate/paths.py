"""Default filesystem paths."""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

PATHOGENS_YAML = CONFIG_DIR / "pathogens.yaml"
DEFAULT_SOURCE_SNAPSHOT_MANIFEST = CONFIG_DIR / "source_snapshot.yaml"

RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
REFERENCE_DIR = DATA_DIR / "reference"


@dataclass(frozen=True)
class RawInputPaths:
    xml: Path
    index: Path


def raw_input_paths(uncompressed: bool = False) -> RawInputPaths:
    """Return the canonical compressed or optional plain-text raw inputs."""
    compression_suffix = "" if uncompressed else ".gz"
    return RawInputPaths(
        xml=RAW_DIR / f"biosamples.xml{compression_suffix}",
        index=RAW_DIR / f"biosample_index.tsv{compression_suffix}",
    )


DEFAULT_XML_INPUT = raw_input_paths().xml
DEFAULT_INDEX_TSV = raw_input_paths().index

DEFAULT_EXTRACTED_TSV = OUTPUT_DIR / "extracted_metadata.tsv"

DEFAULT_TAXONOMY_DIR = REFERENCE_DIR / "taxonomy"
DEFAULT_TAXIDS_NCBI = DEFAULT_TAXONOMY_DIR / "taxids_ncbi.tsv"
DEFAULT_TAXIDS_CURATED = DEFAULT_TAXONOMY_DIR / "taxids_curated.tsv"
DEFAULT_NAMES_DMP = DEFAULT_TAXONOMY_DIR / "names.dmp"
DEFAULT_NODES_DMP = DEFAULT_TAXONOMY_DIR / "nodes.dmp"
DEFAULT_ONTOLOGY_TSV = REFERENCE_DIR / "ontology_terms.tsv"
DEFAULT_GEO_LOC_LIST = REFERENCE_DIR / "geo_loc_list.txt"

DEFAULT_ISO_CACHE_DB = CACHE_DIR / "llm_iso_cache.db"
DEFAULT_LOC_CACHE_DB = CACHE_DIR / "llm_loc_cache.db"
