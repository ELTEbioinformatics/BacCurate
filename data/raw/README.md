
This folder contains immutable pipeline inputs.

- `biosamples.xml.gz` - Filtered [BioSample XML metadata](https://ftp.ncbi.nlm.nih.gov/biosample/)
  streamed directly by the pipeline.

- `biosample_index.tsv.gz` - Contains one row for each BioSample accession that the NCBI
  taxonomy queries or the 2025-05 AllTheBacteria release names. Its columns are:
  - `accession` - Pipeline-wide join key.
  - `taxon_biosample` - Taxon key from the BioSample `Organism`, or `NA` for records that
    appear only through AllTheBacteria.
  - `sylph_species` - Raw GTDB-style classification label from AllTheBacteria. `NA` when
    AllTheBacteria does not include the accession.
  - `taxid` - NCBI Taxonomy ID from the BioSample `Organism` field, or `NA` for records that appear 
    only through AllTheBacteria.
  - `organism_value` - Unchanged BioSample `Organism` value, or `NA` for records that appear only
    through AllTheBacteria.
  - `osf_tarball_filename` - AllTheBacteria assembly tarball filename, or `NA`.
  - `sra_run_accessions` - SRA run accessions joined with `||`, or `NA`.
  - `genbank_assembly_accessions` - Genome assembly accessions joined with `||`, or
    `NA`.
  - `refseq_assembly_accessions` - RefSeq genome assembly accessions joined with `||`, or
    `NA`.

- `id_lists/` - One `<taxon_key>.tsv` per NCBI taxonomy query, written by
  `scripts/parse_biosample_xml.py`. Its `manifest.tsv` records the taxon keys and their
  taxids. `scripts/build_biosample_index.py` reads that manifest.

- `sequence_accessions/` - Compressed two-column intermediate files generated
  from the SRA, GenBank Assembly, and RefSeq Assembly bulk reports.

- `SRA_Accessions.tab`, `assembly_summary_genbank.txt`, and `assembly_summary_refseq.txt` - Additional metadata 
  downloaded from NCBI and processed by `scripts/filter_sequence_accessions.py`.

