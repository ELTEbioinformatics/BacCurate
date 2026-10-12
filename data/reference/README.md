# `data/reference/`

- `ne_10m_admin_0_map_units.geojson` - Natural Earth 1:10m admin-0 map units, `v5.1.2`. Credits: [Natural Earth](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_10m_admin_0_map_units.geojson)
- `ontology/` - the isolation-source vocabulary. [`ontology/README.md`](ontology/README.md) describes its model.
  - `facets.yaml` - facet definitions and other metadata.
  - `terms.tsv` - ontology term identities, labels, parents, definitions, and scope notes.
  - `synonyms.tsv` - typed synonyms of the terms.
  - `obsolete_terms.tsv` - retired term identifiers and the active terms that replace them.
  - `relations.tsv` - typed relations between terms.
  - `mappings.sssom.tsv` - external ontology mappings.
  - `mappings.sssom.yml` - mapping-set metadata and prefix declarations.
  - `baccurate.ttl` - the SKOS release file, generated from the source files.
  - `CHANGELOG.md` - the changes of each vocabulary version.
  - `LICENSE` - the CC BY 4.0 license of the vocabulary.
- `data/reference/geo_loc_list.txt` - The [INSDC Geographic Location Name List](https://www.ncbi.nlm.nih.gov/genbank/collab/country/).
- `taxonomy/`
  - `taxids_ncbi.tsv` - Taxonomy lookup table and dmp files from [NCBI Taxonomy FTP taxdmp.zip](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/)
