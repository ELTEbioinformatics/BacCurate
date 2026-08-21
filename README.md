# BacCurate

[BacCurate](https://baccurate.org/) turns heterogeneous public sequencing metadata into a
standardized resource for comparative genomics, genomic epidemiology, and One Health research.

This repository contains the source code that extracts and harmonizes
[BioSample](https://www.ncbi.nlm.nih.gov/biosample) metadata. The dataset itself, and the
documentation, are on the [website](https://baccurate.org/).

### Samples covered

As of the 2026-07-09 BioSample snapshot, ~1.5 million BioSample records are covered, including all
_ESKAPEE_ pathogens:

- _Enterococcus faecium_
- _Enterococcus faecalis_
- _Staphylococcus aureus_
- _Klebsiella pneumoniae species complex_
- _Acinetobacter baumannii_
- _Pseudomonas aeruginosa_
- _Enterobacter_ spp.
- _Escherichia coli_

### Methodology in brief

1. The NCBI BioSample XML dump is parsed and the records of the registered taxa are indexed.
2. The attribute-value pairs of interest from each record are selected.
3. Values are resolved against the reference vocabularies in [`data/reference/`](data/reference):
   - **Isolation source**: Large Language Model (LLM) assisted mapping to a purpose-built ontology
     of 103 terms (`ontology/terms.tsv`), with [SSSOM](https://mapping-commons.github.io/sssom/)
     mappings to external ontologies.
   - **Host organism**: mapped to [NCBI taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy)
     (`taxonomy/`) taxids and scientific names.
   - **Geographic location**: mapped to the
     [INSDC Geographic Location Name List](https://www.ncbi.nlm.nih.gov/genbank/collab/country/) and
     [Natural Earth](https://www.naturalearthdata.com/) map units for coordinate-derived countries.
   - **Collection dates**: parsed with custom-built rules and normalized to
     [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html).

LLMs are only used for isolation-source standardization, everything else is deterministic and driven
by hand-curated rules.

For more information on the methods and the interpretation of the output data,
[read the full documentation here](https://baccurate.org/#/documentation) (_still WIP_).

## Installation

Requires Python 3.12 or later.

### 1. Clone the repository

```bash
git clone https://github.com/ELTEbioinformatics/BacCurate.git
cd BacCurate
```

### 2. Install dependencies and setup environment

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

If you'd rather not install `uv`, `pip install -e .` also works in a standard `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

<!-- prettier-ignore -->
> [!NOTE]
> The commands below use `uv`. With a `venv` activated instead, drop the `uv run` prefix.

Verify the installation:

```bash
uv run baccurate --help
```

### 3. Configuring API credentials (optional)

By default, isolation-source standardization utilizes an OpenAI compatible API.

Create a `.env` file in the repository root with your API credentials:

```
API_KEY="sk-..."
SERVER="https://api.openai.com/v1"
LLM_MODEL="gpt-4o-mini"
```

Use `--skip-llm` to process without LLMs.

### 4. Setting up input data

The starting dataset is assembled from:

- [NCBI BioSample metadata](https://ftp.ncbi.nlm.nih.gov/biosample/) (`biosample_set.xml.gz`)
- [AllTheBacteria](https://allthebacteria.org/) metadata (sylph/GTDB species profiling)
- [NCBI Genbank metadata](https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_genbank.txt)
- [NCBI RefSeq metadata](https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt)
- [NCBI SRA metadata](https://ftp.ncbi.nlm.nih.gov/sra/reports/Metadata/SRA_Accessions.tab)

Place these in `data/raw`.

The [NCBI taxonomy dump](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/) (`nodes.dmp`, `names.dmp`,
`merged.dmp`) goes in `data/reference/taxonomy/`. Then run:

```bash
uv run python scripts/parse_biosample_xml.py
uv run python scripts/filter_sequence_accessions.py
uv run python scripts/build_biosample_index.py
```

## Usage

### Running all pipelines for each taxon

```bash
uv run baccurate
```

Outputs are in `output/<timestamp>/`.

### Specifying taxa

You can provide taxon keywords defined in `config/taxa.yaml` to only process select taxa.

```bash
# Acinetobacter baumannii and Escherichia coli
uv run baccurate abaumannii ecoli
```

### Specifying pipelines

The `--standardize` option accepts `host`, `date`, `loc`, and `iso` (_host organism_, _collection
date_, _geographical location_ and _isolation-source_).

```bash
# Collection date and geographical location standardization
uv run baccurate abaumannii --standardize date loc
```

### Re-extracting metadata

Extraction (selecting the attribute-value pairs of interest from the raw BioSample metadata) runs
only if `output/extracted_metadata.tsv` does not exist.

To re-extract, either delete it or specify a new filename:

```bash
uv run baccurate --extracted-metadata extracted_metadata_new.tsv
```

### Debug mode

Enable verbose logging:

```bash
uv run baccurate --debug
```
