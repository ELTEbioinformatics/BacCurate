"""Default filesystem paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

PATHOGENS_YAML = CONFIG_DIR / "pathogens.yaml"
DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST = CONFIG_DIR / "biosample_snapshot.yaml"
DEFAULT_BIOPROJECT_SNAPSHOT_MANIFEST = CONFIG_DIR / "bioproject_snapshot.yaml"

RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
REFERENCE_DIR = DATA_DIR / "reference"


DEFAULT_BIOSAMPLE_XML_INPUT = RAW_DIR / "biosamples.xml.gz"
DEFAULT_BIOPROJECT_XML_INPUT = RAW_DIR / "bioproject.xml.gz"
DEFAULT_XML_INPUT = DEFAULT_BIOSAMPLE_XML_INPUT
DEFAULT_INDEX_TSV = RAW_DIR / "biosample_index.tsv.gz"

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
