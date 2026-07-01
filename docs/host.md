# Host standardization

Map host annotations from sample metadata to NCBI Taxonomy ID and scientific names (binomial nomenclature).

[Source](https://github.com/kadan02/BacCurate/blob/main/src/baccurate/standardizers/host.py)

## Contents

- [Usage](#usage)
- [Configuration](#configuration)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Data usage recommendations](#data-usage-recommendations)
- [Methods](#methods)
  - [Workflow](#workflow)
  - [Taxonomy reference](#taxonomy-reference)
  - [Matching cascade](#matching-cascade)
  - [Reliability score](#reliability-score)
  - [Isolation-source preemption](#isolation-source-preemption)
  - [Manual overrides](#manual-overrides)

## Usage

Run the host pipeline for one or more pathogens with the `host` attribute:

```bash
uv run baccurate <pathogen> --attribute host
```
See the [main README](../README.md#usage) for installation and the full set of options.

## Configuration

To blacklist/whitelist the attributes to process, see `config/host.yaml`. 

Additional parsing configurations:

```yaml
ignored_substrings:    # stripped before matching
  - "healthy"
  - ", juvenile"
  - ", adult"

iso_keywords:          # if substring matches, record passed to iso-source
  - food
  - soil
  - meat

overrides:            # manually override given taxid's outputs
  taxid_overrides:
    "Squirrel monkey": 9521
    "poultry flies": null
```

## Inputs

A TSV file with one row per record:

| Column      | Description                                                  |
|-------------|--------------------------------------------------------------|
| `accession` | Record ID                                                    |
| `host_attr_orig` | `\|\|`-separated attribute names                             |
| `host_val_orig`  | `\|\|`-separated values, paired by position with `host_attr_orig` |

## Outputs

| Column                | Description                                                                       |
|-----------------------|-----------------------------------------------------------------------------------|
| `accession`           | Record ID                                                                         |
| `host_taxid`          | NCBI taxonomy ID                                                                  |
| `host_sci_name`       | Linnaean/Binomial name                                                            |
| `host_score`          | See [below](#reliability-score)                                                   |
| `host_low_conf`       | Boolean review flag                                                               |
| `host_common_names`   | Comma-separated NCBI common names for the host taxid                              |
| `host_lineage_names`  | Comma-separated ancestor scientific names, root-to-tip, limited to standard ranks |
| `host_lineage_taxids` | Comma-separated NCBI taxids, paired by position with `host_lineage_names`         |
| `host_attr_orig`      | Unstandardized input attribute(s)                                                   |
| `host_val_orig`       | Unstandardized input value(s)                                                       |

`host_overflow.tsv` lists records forwarded to the
isolation-source pipeline. A record may appear in both files when it has multiple input rows
and at least one row hit an iso-source keyword while another produced a host match.

## Data usage recommendations

Filtering on `reliability_score >= 0.9 AND
low_confidence == False` retains matches that came from a taxid,
scientific name, NCBI synonym, locally curated keyword, or NCBI
`genbank_common_name`, with no subset-match or cross-attribute
ambiguity.

Scores 0.70 (NCBI broad `common_name` or multi-word subset) and 0.50
(single-word subset) should be reviewed before being trusted in downstream analysis.

The `low_confidence` flag is orthogonal to the score: a 1.00 match
can be flagged when two input attributes for the same accession
resolved to different taxa (the highest-scoring one is kept and the
disagreement is flagged), and this is worth filtering on when
attribute consistency matters.

The `attribute` and `value` columns record which input field and
which raw string produced the match, so any output row can be traced
back to its source annotation.

## Methods

### Workflow

![Flowchart](charts/host.png)

The first two checks (iso-source keywords and manual overrides) are
preemptive. The remaining tiers are tried in descending score order.

### Taxonomy reference

The controlled vocabulary is split across two tables. `taxids_ncbi.tsv`
was generated from NCBI's taxonomy dump and carries scientific names,
synonyms, `genbank_common_name`, and `common_name`. `taxids_curated.tsv`
is manually maintained for this dataset and lists keywords (e.g.
`cattle` → *Bos taurus*) keyed by taxid. Entries were added when a
"common" term is missing from NCBI (in a lot of cases, typos) or 
when a common name/synonym would resolve to a taxon other than the one intended in 
this dataset.

### Value mapping

Each `||`-separated value is normalized (lowercased, punctuation
stripped, whitespace collapsed) and mapped to the reference. 
Within a record, all candidates are collected and ranked by
`(score, taxonomic specificity, attribute priority, source position)`,
where `host_taxid` outranks `host`, which outranks other attributes.
Taxonomic specificity is taken from the row order of the taxonomy reference, 
which is sorted from most-specific (subspecies, species)
to least-specific (genus and above), so when two candidates are tied on
score the more specific taxon wins. Subspecies are indexed for exact
matching only, not for subset matching, to avoid trinomial false
positives (e.g. *Gallus gallus gallus* matching `Gallus gallus`).

The two lowest tiers are *subset* matches, not fuzzy matches in the
edit-distance sense: matching is on whole words after normalization.

### Reliability score

| Score | Tier                                                                |
|------:|---------------------------------------------------------------------|
|  1.00 | Direct taxid, exact scientific name, exact synonym, manual override |
|  0.95 | Locally curated keyword                                             |
|  0.90 | NCBI `genbank_common_name`                                          |
|  0.70 | NCBI `common_name`, or multi-word subset match                      |
|  0.50 | Single-word subset match                                            |

Score reflects how the match was made, not how taxonomically specific the
result is.

A `low_confidence` flag is set independently of score on any subset
match, on subset matches that resolved across multiple distinct taxa,
or when multiple input attributes resolved to different taxa.


### Isolation-source preemption

Host and isolation-source annotations are frequently conflated in
the metadata. Values matching any configured isolation-source indicating
keyword (e.g. `food`, `soil`, `meat`) are forwarded to
`host_overflow.tsv` and skip host matching entirely.
Matching is whole-word on the normalized value, so `food` matches
`duck food` but not `seafood`. A multi-word keyword matches when all
of its words are present (order-independent).

### Manual overrides

`config/host.yaml` accepts `taxid_overrides`, a map from normalized
input value to either a taxid (force match) or `null` (reject and
forward). Overrides take precedence over all matching tiers and are
intended for cases where rule-based matching is structurally wrong -
for example, `"Squirrel monkey"` resolving to *Sciurus* via single-word
subset matching rather than to *Saimiri*. Override taxids are validated
against the loaded taxonomy at startup.
