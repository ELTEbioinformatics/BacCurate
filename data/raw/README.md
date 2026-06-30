
Thid folder includes immutable inputs to the pipeline.

- `biosamples.xml` - Downloaded BioSample XML metadata.


- `biosample_index.tsv` - One row per BioSample accession. Includes the following columns:
    - `accession` - The join key across the pipeline.
    - `in_ATB` - True if the accession is present in ATB.
    - `pathogen_biosample` - pathogen key via a `config/pathogens.yaml`
    - `pathogen_ATB` - pathogen key derived from ATB's sequence classification,
      or `NA` when `in_ATB` is False.
    - `organism_value` - raw `Organism` value from the XML
