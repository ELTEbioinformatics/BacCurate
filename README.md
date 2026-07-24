# BacCurate

[BacCurate](https://baccurate.org/) turns heterogeneous public sequencing metadata into a
standardized, confidence-scored resource for comparative genomics, genomic epidemiology, and One
Health research.

This repository contains the source code that extracts and harmonizes the metadata from the
[NCBI BioSample](https://www.ncbi.nlm.nih.gov/biosample) database.

### Pathogens covered (includes all ESKAPEE)

- _Enterococcus faecium_
- _Enterococcus faecalis_
- _Staphylococcus aureus_
- _Klebsiella pneumoniae species complex_
- _Acinetobacter baumannii_
- _Pseudomonas aeruginosa_
- _Enterobacter_ spp.
- _Escherichia coli_

### Standardized attributes

- Collection date
- Host organism
- Geographical location
- Isolation source

## Installation

Requires Python 3.12 or later.

### 1. Clone the repository

```bash
git clone https://github.com/ELTEbioinformatics/BacCurate.git
cd BacCurate
```

### 2. Install dependencies and setup environment

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

For development:

```bash
uv sync --extra dev
```

If you'd rather not install `uv`, `pip install -e .` also works in a standard `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Verify the installation

```bash
uv run python main.py --help
```

### 3. (Optional) Configuring environment variables

The standardization scripts can call LLM APIs as a fallback. To use this feature, create a `.env`
file in the repository root with your API credentials:

```
API_KEY="openai_api_key"
SERVER="url"
LLM_MODEL="model-name"
```

### 4. Setting up input data

The starting dataset is assembled locally from three public sources:

- the [NCBI BioSample metadata](https://ftp.ncbi.nlm.nih.gov/biosample/) (`biosample_set.xml.gz`),
- the [NCBI taxonomy dump](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/) (`nodes.dmp`, `names.dmp`,
  `merged.dmp`),
- the [AllTheBacteria](https://allthebacteria.org/) metadata (sylph/GTDB species profiling).

With those in place, run:

```bash
python scripts/parse_biosample_xml.py
python scripts/build_biosample_index.py data/raw/atb_2025-05.tsv
```

See [docs/data_acquisition.md](docs/data_acquisition.md) for more information.

## Usage

To process all available attributes (host, date, loc, iso) for one or more pathogens:

```bash
uv run baccurate abaumannii ecoli
```

You must provide at least one pathogen keyword. Keywords are defined in `config/pathogens.yaml`.
Example: `ecoli`

### Running specific pipelines only

Available options are: `host`, `date`, `loc`, `iso`.

```bash
# Run date and location standardization
uv run baccurate abaumannii --attribute date loc
```

### Re-extracting metadata

By default, the pipeline skips extraction (searching for valid attributes) if
`extracted_metadata.tsv` already exists. To force re-extraction, use the --extracted-metadata flag:

```bash
uv run baccurate abaumannii --extracted-metadata extracted_metadata_new.tsv
```

### Debug mode

Enable verbose logging:

```bash
uv run baccurate abaumannii --debug
```
