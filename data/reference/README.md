# `data/reference/`

External or curated reference data. Updated by hand or by ad-hoc scripts in `/scripts/`.

- `ontology_terms.tsv` - curated isolation-source ontology used by the iso standardizer.
- `taxonomy/`
  - `taxids_ncbi.tsv` - NCBI host taxonomy lookup table. Curated host matching 
  - policy lives in `config/host.yaml`.
  - `names.dmp`, `nodes.dmp` - NCBI taxonomy dumps used for host lineage enrichment. Download from NCBI
