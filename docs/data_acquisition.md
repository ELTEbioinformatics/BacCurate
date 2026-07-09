# Data acquisition

NCBI BioSample metadata and the NCBI Taxonomy dump were downloaded on 2026-07-09 and
the 2025-05 release of AllTheBacteria was used.

## Classification

BioSample records were chosen if they met either of two criteria:

1. `Organism` taxid fell within a target taxon's subtree
2. `accession` matched with a sylph/GTDB species profiling (from AllTheBacteria metadata).

The final dataset is the union of BioSample records identified by either criterion,
with the sylph profiling taking precedence of the taxon identification.

## Included taxa

| Taxon                                     | NCBI taxid   |
|-------------------------------------------|--------------|
| *Acinetobacter baumannii*                 | 470          |
| *Escherichia coli* (including *Shigella*) | 562 (+ 620)  |
| *Staphylococcus aureus*                   | 1280         |
| *Pseudomonas aeruginosa*                  | 287          |
| *Enterobacter* genus                      | 547          |
| *Klebsiella pneumoniae* species complex   | see below    |
| *Enterococcus faecium* / *E. faecalis*    | 1352 / 1351  |


### *Klebsiella pneumoniae* species complex

Defined per Lam et al. (2021) as:

| Species                      | NCBI taxid |
|------------------------------|------------|
| *Klebsiella pneumoniae*      | 573        |
| *Klebsiella quasipneumoniae* | 1463165    |
| *Klebsiella variicola*       | 244366     |
| *Klebsiella quasivariicola*  | 2026240    |
| *Klebsiella africana*        | 2489010    |


### Inclusion of *Shigella* under *Escherichia coli*

To align our dataset with the GTDB-adopted genomic species definition, 
the union of taxids 562 and 620 were reported both as *Escherichia coli*.

The reasoning behind is that *Shigella* species are typically distinguished from other *E.
coli* lineages by an acquired virulence plasmid rather than by
genome-level divergence, and the genus name persists mostly through
nomenclatural convention (Pupo et al., 2000).

### *Enterococcus*

*Enterococcus faecium* and *E. faecalis* were the
two species retrieved from the *Enterococcus* genus.


## References

- AllTheBacteria species identification:
  <https://allthebacteria.org/docs/species_id/>
- Lam, M. M. C., Wick, R. R., Watts, S. C., Cerdeira, L. T., Wyres, K. L., & Holt, 
K. E. (2021). **A genomic surveillance framework and genotyping tool for Klebsiella 
pneumoniae and its related species complex.** Nature Communications, 12(1). 
https://doi.org/10.1038/s41467-021-24448-3
- Hunt, M., Torres, M. D. T., Alikhan, N.-F., Anderson, D., Andreani, M. L., Blom, J., 
Bouras, G., Brinkman, F. S. L., Carroll, L. M., Croxen, M. A., Floto, R. A., Hall, 
M. B., Hawkey, J., Horsfield, S. T., Jia, B., Lacey, J. A., Lee, H.-S., Lima, 
L., MacAlasdair, N., … Iqbal, Z. (2024). 
**AllTheBacteria: a community resource empowers biology and discovers novel peptide antibiotics.**
openRxiv. https://doi.org/10.1101/2024.03.08.584059
- Parks, D. H., Chuvochina, M., Reeves, P. R., Beatson, S. A., & Hugenholtz, P. (2021). **Reclassification of Shigella 
species as later heterotypic synonyms of Escherichia coli in the Genome Taxonomy Database.** openRxiv. https://doi.org/10.1101/2021.09.22.461432
- Pupo, G. M., Lan, R., & Reeves, P. R. (2000). **Multiple independent origins of 
Shigella clones of Escherichia coli and convergent evolution of many of their 
characteristics.** Proceedings of the National Academy of Sciences, 97(19), 10567–10572. 
https://doi.org/10.1073/pnas.180094797
