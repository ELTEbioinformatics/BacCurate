
This folder includes immutable inputs to the pipeline.

- `biosamples.xml.gz` - Filtered [BioSample XML metadata](https://ftp.ncbi.nlm.nih.gov/biosample/), streamed directly by the pipeline.


- `biosample_index.tsv.gz` - One row per BioSample accession, streamed directly by the pipeline. Includes the following columns:
    - `accession` - The join key across the pipeline.
    - `in_ATB` - True if the accession is present in AllTheBacteria collection.
    - `pathogen_biosample` - pathogen key (`config/pathogens.yaml`) from BioSample `Organism` field   ,
      or `NA` for records only via ATB.
    - `pathogen_ATB` - pathogen key derived from ATB's sylph/GTDB classification,
      or `NA` when the sylph species maps to no target.
    - `taxid` - the record's NCBI taxonomy id from the `Organism` field (often strain-level),
      or `NA` for ATB-only records.
    - `organism_value` - raw `Organism` value from the XML (for debug). `NA` for ATB-only records.
    - `osf_tarball_filename` - the ATB assembly tarball for the accession, or `NA`.

  See `docs/data_acquisition.md` for how this index is generated.
