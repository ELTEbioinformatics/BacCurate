
This folder contains immutable pipeline inputs.

- `biosamples.xml.gz` - Filtered [BioSample XML metadata](https://ftp.ncbi.nlm.nih.gov/biosample/)
  streamed directly by the pipeline.

- `biosample_index.tsv.gz` - Contains one row for each BioSample accession. The pipeline streams it
  directly. Its columns are:

  - `accession` - BioSample accession and pipeline-wide join key.
  - `in_ATB` - `True` if the BioSample accession is present in AllTheBacteria.
  - `pathogen_biosample` - Pathogen key from the BioSample `Organism`, or `NA` for records that
    appear only through AllTheBacteria.
  - `pathogen_ATB` - Pathogen key from the AllTheBacteria classification, or `NA` when its
    `sylph_species` maps to no target pathogen.
  - `sylph_species` - GTDB-style classification label.
  - `taxid` - NCBI Taxonomy ID from the BioSample `Organism`, or `NA` for records that appear only
    through AllTheBacteria.
  - `organism_value` - Unchanged BioSample `Organism` value, or `NA` for records that appear only
    through AllTheBacteria.
  - `osf_tarball_filename` - AllTheBacteria assembly tarball filename, or `NA`.
  - `sra_run_accessions` - Retrievable public SRA run accessions joined with `||`, or `NA`.
  - `genbank_assembly_accessions` - Current GenBank genome assembly accessions joined with `||`, or
    `NA`.
  - `refseq_assembly_accessions` - Current RefSeq genome assembly accessions joined with `||`, or
    `NA`.

- `sequence_accessions/` - Compressed two-column intermediate files generated
  from the SRA, GenBank Assembly, and RefSeq Assembly bulk reports.

- `SRA_Accessions.tab`, `assembly_summary_genbank.txt`, and `assembly_summary_refseq.txt` - Additional metadata 
  downloaded from NCBI and processed by `scripts/filter_sequence_accessions.py`.

