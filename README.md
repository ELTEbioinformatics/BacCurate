# BacCurate
[BacCurate](https://baccurate.org/) standardizes metadata for highly virulent and antibiotic-resistant
bacterial pathogens into a searchable resource. This repository contains
the source code that extracts and harmonizes the metadata from the
[NCBI BioSample](https://www.ncbi.nlm.nih.gov/biosample) database.

### Pathogens covered (currently all ESKAPEE)
- *Enterococcus faecium*
- *Staphylococcus aureus*
- *Klebsiella pneumoniae*
- *Acinetobacter baumannii*
- *Pseudomonas aeruginosa*
- *Enterobacter* spp.
- *Escherichia coli*

### Standardized attributes
- Collection date
- Host organism
- Geographical location
- Isolation source

### [See the documentation about the standardization process here.](https://github.com/ELTEbioinformatics/BacCurate/tree/main/docs)
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
The standardization scripts can call LLM APIs as a fallback. To use this feature, create a `.env` file in the repository root with your API credentials:
```
API_KEY="openai_api_key"
SERVER="url"
LLM_MODEL="model-name"
```
### 4. Setting up input data
Under construction

## Usage
To process all available attributes (host, date, loc, iso) for one or more pathogens:
```bash
uv run baccurate abaumannii ecoli
```
You must provide at least one pathogen keyword. The valid keywords are defined in
`config/pathogens.yaml` (`abaumannii`, `ecoli`, `kpneumoniae`, `saureus`,
`paeruginosa`, `efaecium`, `enterobacter`). Each is also the input-file name and
the `pathogen` output-column value.

### Running specific pipelines only
Available options are: `host`, `date`, `loc`, `iso`.
```bash
# Run date and location standardization
uv run baccurate abaumannii --attribute date loc
```
### Re-extracting metadata
By default, the pipeline skips extraction (searching for valid attributes) if `extracted_metadata.tsv` already exists. To force re-extraction, use the --extracted-metadata flag:
```bash
uv run baccurate abaumannii --extracted-metadata extracted_metadata_new.tsv
```
### Debug mode
Enable verbose logging:
```bash
uv run baccurate abaumannii --debug
```
