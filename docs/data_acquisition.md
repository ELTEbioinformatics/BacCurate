# Data acquisition

NCBI BioSample metadata and the NCBI Taxonomy dump were downloaded on 2026-07-09 and the 2025-05
release of AllTheBacteria was used.

## Classification

BioSample records were chosen if they met either of two criteria:

1. `Organism` taxid fell within a target taxon's subtree
2. `accession` matched with a sylph/GTDB species profiling (from AllTheBacteria metadata).

The final dataset is the union of BioSample records identified by either criterion, with the sylph
profiling taking precedence of the taxon identification.

## Included taxa

| Taxon                                     | NCBI taxid  |
| ----------------------------------------- | ----------- |
| _Acinetobacter baumannii_                 | 470         |
| _Escherichia coli_ (including _Shigella_) | 562 (+ 620) |
| _Staphylococcus aureus_                   | 1280        |
| _Pseudomonas aeruginosa_                  | 287         |
| _Enterobacter_ genus                      | 547         |
| _Klebsiella pneumoniae_ species complex   | see below   |
| _Enterococcus faecium_ / _E. faecalis_    | 1352 / 1351 |

### _Klebsiella pneumoniae_ species complex

Defined per Lam et al. (2021) as:

| Species                      | NCBI taxid |
| ---------------------------- | ---------- |
| _Klebsiella pneumoniae_      | 573        |
| _Klebsiella quasipneumoniae_ | 1463165    |
| _Klebsiella variicola_       | 244366     |
| _Klebsiella quasivariicola_  | 2026240    |
| _Klebsiella africana_        | 2489010    |

### Inclusion of _Shigella_ under _Escherichia coli_

To align our dataset with the GTDB-adopted genomic species definition, the union of taxids 562 and
620 were reported both as _Escherichia coli_.

The reasoning behind is that _Shigella_ species are typically distinguished from other _E. coli_
lineages by an acquired virulence plasmid rather than by genome-level divergence, and the genus name
persists mostly through nomenclatural convention (Pupo et al., 2000).

### _Enterococcus_

_Enterococcus faecium_ and _E. faecalis_ were the two species retrieved from the _Enterococcus_
genus.

## References

- AllTheBacteria species identification: <https://allthebacteria.org/docs/species_id/>
- Lam, M. M. C., Wick, R. R., Watts, S. C., Cerdeira, L. T., Wyres, K. L., & Holt, K. E. (2021). **A
  genomic surveillance framework and genotyping tool for Klebsiella pneumoniae and its related
  species complex.** Nature Communications, 12(1). https://doi.org/10.1038/s41467-021-24448-3
- Hunt, M., Torres, M. D. T., Alikhan, N.-F., Anderson, D., Andreani, M. L., Blom, J., Bouras, G.,
  Brinkman, F. S. L., Carroll, L. M., Croxen, M. A., Floto, R. A., Hall, M. B., Hawkey, J.,
  Horsfield, S. T., Jia, B., Lacey, J. A., Lee, H.-S., Lima, L., MacAlasdair, N., … Iqbal, Z.
  (2024). **AllTheBacteria: a community resource empowers biology and discovers novel peptide
  antibiotics.** openRxiv. https://doi.org/10.1101/2024.03.08.584059
- Parks, D. H., Chuvochina, M., Reeves, P. R., Beatson, S. A., & Hugenholtz, P. (2021).
  **Reclassification of Shigella species as later heterotypic synonyms of Escherichia coli in the
  Genome Taxonomy Database.** openRxiv. https://doi.org/10.1101/2021.09.22.461432
- Pupo, G. M., Lan, R., & Reeves, P. R. (2000). **Multiple independent origins of Shigella clones of
  Escherichia coli and convergent evolution of many of their characteristics.** Proceedings of the
  National Academy of Sciences, 97(19), 10567–10572. https://doi.org/10.1073/pnas.180094797
