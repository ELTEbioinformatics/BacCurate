# Changelog

## 1.0.0rc2 (2026-10-21)

### Terms

- `compost`: removed the `manure` parent.
- `river water`: changed the narrow synonym to `fresh water` instead of `water`.
- `infection`: changed to bacterial only. Aim is that non-bacterial infections, such as "COVID-19", don't get annotated with `infection`.
- `disease` and `infection`: scope notes also cover diseases or infections with no narrower term, such as cancer.
- `drainage fluid`: Changed to only be assigned for a fluid without a term of its own.
- New definitions for `infection` and 7 kinds of `infection`.

### Mappings

| Term                               | Object           | Before        | After          |
| ---------------------------------- | ---------------- | ------------- | -------------- |
| `infection`                        | `MONDO:0005113`  | None          | `exactMatch`   |
| `surface`                          | `ENVO:00010504`  | `closeMatch`  | None           |
| `skin`                             | `UBERON:0000014` | `exactMatch`  | None           |
| `skin`                             | `UBERON:0002097` | None          | `exactMatch`   |
| `water`                            | `ENVO:01000599`  | `narrowMatch` | None           |
| `fresh water`                      | `ENVO:01000599`  | None          | `narrowMatch`  |
| `thorax`                           | `UBERON:0001443` | `exactMatch`  | `broadMatch`   |
| `infection`                        | `MONDO:0005550`  | `exactMatch`  | `broadMatch`   |
| `respiratory tract infection`      | `MONDO:0024355`  | `exactMatch`  | `broadMatch`   |
| `pneumonia`                        | `MONDO:0005249`  | `exactMatch`  | `broadMatch`   |
| `urinary tract infection`          | `MONDO:0100338`  | `exactMatch`  | `broadMatch`   |
| `central nervous system infection` | `MONDO:0024619`  | `exactMatch`  | `broadMatch`   |
| `endocarditis`                     | `MONDO:0000565`  | `exactMatch`  | `broadMatch`   |
| `eye infection`                    | `MONDO:0043885`  | `exactMatch`  | `broadMatch`   |
| `mastitis`                         | `MONDO:0006849`  | `exactMatch`  | `broadMatch`   |
| `gastrointestinal infection`       | `MONDO:0000916`  | `narrowMatch` | `relatedMatch` |
| `skin and soft tissue infection`   | `MONDO:0021201`  | `narrowMatch` | `relatedMatch` |
| `ear infection`                    | `MONDO:0004795`  | `narrowMatch` | `relatedMatch` |
| `ear infection`                    | `MONDO:0024330`  | `narrowMatch` | `relatedMatch` |
| `hospital`                         | `ENVO:03501158`  | `narrowMatch` | `relatedMatch` |
| `sepsis`                           | `MONDO:0005229`  | `closeMatch`  | `exactMatch`   |

### Labeling rules (`config/isolation_source.yaml`)

- Additional instructions for the model to use the nearest term for values that don't have an exact term. 
  - Example: "Bed rail" → `inanimate object`.
- Additional instructions for the model for alternatives ("or", "and/or", "/") to keep only shared terms. 
  - Example: `rectal swab or faeces` → `digestive tract`.
- Changed the `diagnosis → body term` rule to be more conservative. Dropped the following from this rule:
    - `pneumonia` 
    - `respiratory tract infection`
    - `meningitis`
    - `central nervous system infection`
    - `bone and joint infection`
- And changed `bronchiectasis` → `lower respiratory tract` instead of `bronchus`

## 1.0.0rc1 (2026-10-12)

This is the first release of the vocabulary. The changes below compare it with `1.0.0-beta`, the
version in BacCurate before this release (which had the internal version `2.3.0`).

### Structure

- Each parent now has a `parent_relation`: `kind_of` (68 terms) or `part_of` (33 terms).
- Typed relations in `relations.tsv` replace the `crosslink_targets` column. The relation types are `located_in`,
  `produced_by`, `collected_from`, `has_part`, `part_of`, and `implies_source_type`.
- Synonyms are in `synonyms.tsv`, each with one of the types `exact`, `narrow`, `broad`, and
  `related`.
- `definition` and `scope_note` replace `curation_note`.
- Obsolete terms are in `obsolete_terms.tsv`, replacing `retired_term_ids`.
- The pipeline settings of the facets (`output_column`, `render_order`, and `classifier_guidance`),
  the `enables_host_recovery` column, and the direct-match values moved to
  `config/isolation_source.yaml`. `cardinality` was removed.
- Term IRIs use `https://w3id.org/baccurate/BACC_`.
- `baccurate.ttl` is the SKOS release file, generated from the source files.

### Added facet

`host_disease`:  A diagnosis or the health state of the host. This facet binds to the MIxS field `host_disease_stat` 
and the GENEPIO field `host disease`. These bindings previously belonged to `lesion`, which now has no standard 
bindings.

### Added terms

Added 53 new terms. The parent of each term is in brackets.

- `source_type`: `BACC:0000122` `human host` (`host-associated`).
- `body_product`: `BACC:0000119` `drainage fluid` (`body fluid`).
- `lesion`: `BACC:0000120` `cyst`.
- `facility`: `BACC:0000123` `veterinary facility`, and three kinds of `healthcare facility`:
  `BACC:0000124` `hospital`, `BACC:0000125` `primary care facility`, and `BACC:0000126`
  `long-term care facility`.
- `body_site`, body regions: `BACC:0000154` `head`, `BACC:0000155` `neck`, `BACC:0000156` `thorax`,
  `BACC:0000157` `back`, `BACC:0000109` `abdomen`, `BACC:0000110` `inguinal region` (`abdomen`),
  `BACC:0000111` `axilla`, `BACC:0000121` `perineum`, `BACC:0000106` `limb`, `BACC:0000107` `lower limb`
  (`limb`), and `BACC:0000108` `upper limb` (`limb`).
- `body_site`, organs and systems: `BACC:0000113` `digestive system`, `BACC:0000114` `gallbladder`
  (`digestive system`), `BACC:0000112` `spleen`, `BACC:0000115` `nasopharynx` (`pharynx`),
  `BACC:0000116` `oropharynx` (`pharynx`), `BACC:0000117` `skin and soft tissue`, `BACC:0000118`
  `muscle tissue` (`soft tissue`), `BACC:0000127` `mammary gland`, and `BACC:0000158`
  `skeletal system`.
- `host_disease`: `BACC:0000128` `disease` and `BACC:0000153` `colonization` are the roots.
  - Kinds of `disease`: `BACC:0000129` `infection`, `BACC:0000149` `cystic fibrosis`, `BACC:0000150`
    `bronchiectasis`, `BACC:0000151` `chronic obstructive pulmonary disease`, and `BACC:0000152`
    `atopic dermatitis`.
  - Kinds of `infection`: `BACC:0000130` `bacteremia`, `BACC:0000131` `sepsis`, `BACC:0000132`
    `respiratory tract infection`, `BACC:0000134` `urinary tract infection`, `BACC:0000135`
    `gastrointestinal infection`, `BACC:0000136` `intra-abdominal infection`, `BACC:0000137`
    `skin and soft tissue infection`, `BACC:0000138` `surgical site infection`, `BACC:0000139`
    `device-associated infection`, `BACC:0000140` `bone and joint infection`, `BACC:0000141`
    `central nervous system infection`, `BACC:0000143` `endocarditis`, `BACC:0000144` `eye infection`,
    `BACC:0000145` `ear infection`, `BACC:0000146` `reproductive tract infection`, `BACC:0000147`
    `mastitis`, and `BACC:0000148` `colibacillosis`.
  - `BACC:0000133` `pneumonia` is a kind of `respiratory tract infection`. `BACC:0000142` `meningitis` is
    a kind of `central nervous system infection`.

Notes on the new terms:

- `veterinary facility` is for a veterinary clinic or a veterinary hospital. It is not a kind of
  `healthcare facility`.
- The eight body regions of Terminologia Anatomica have a term: `head`, `neck`, `thorax`, `back`, `abdomen`,
  `perineum`, `upper limb`, and `lower limb`. `head` and `thorax` do not imply organs, and `back` does
  not imply the vertebral column or spinal cord.
- `colonization` is for carriage, a screening sample, asymptomatic bacteriuria, and a healthy host.

### Changed terms

- Label: `BACC:0000008` is `laboratory strain`, not `laboratory`. The old label read like a place,
  so it was confused with `laboratory facility` and with the culture of every isolate.
- Label: `BACC:0000089` is `plumbing drain`, not `drain`.
- Parent: `digestive tract` and `liver` are part of `digestive system`.
- Parent: `bone` and `synovial joint` are part of `skeletal system`.
- Parent: `skin` and `soft tissue` are kinds of `skin and soft tissue`.
- Parent: `intensive care unit` is part of `hospital`, not of `healthcare facility`.
- Parent: `bronchoalveolar lavage fluid` has no parent. It was a child of `respiratory secretion`. It
  has the relation `has_part` `respiratory secretion`.

### Obsolete terms

| ID             | Label                 | Replaced by                                                          |
|----------------|-----------------------|----------------------------------------------------------------------|
| `BACC:0000023` | `intestinal content`  | `BACC:0000022` `digestive tract content`, `BACC:0000039` `intestine` |
| `BACC:0000024` | `caecal content`      | `BACC:0000022` `digestive tract content`, `BACC:0000040` `caecum`    |
| `BACC:0000078` | `biofilm`             | None                                                                 |
| `BACC:0000081` | `laboratory facility` | None                                                                 |

`BACC:0000005` (`built environment`) and `BACC:0000006` (`natural environment`) were already retired in
`1.0.0-beta`. They are now in `obsolete_terms.tsv`, with the replacement `BACC:0000004`
(`environmental`).

### Synonyms, definitions, and scope notes

- The vocabulary has 213 synonyms: 88 `exact`, 112 `narrow`, 5 `broad`, and 8 `related`.
- Values that are not names of their term, for example host names, became direct-match values in
  `config/isolation_source.yaml` and left the vocabulary.
- 61 terms have a definition, and 23 terms have a scope note. Before, 5 terms had a curation note.

### Mappings

The mapping set has 166 mappings: 106 exact, 32 narrow, 18 close, 8 broad, and 2 related.

- Added: 52 mappings of the new terms, 24 of them to MONDO (release 2026-10-06).
- Four of these replace narrow matches of `healthcare facility`. `hospital` has `ENVO:00002173` as
  `exactMatch`, and `ENVO:03501137` and `ENVO:03501158` as `narrowMatch`. `veterinary facility` has
  `ENVO:03501348` as `exactMatch`.
- Removed: the 22 mappings of the source types `environmental`, `food or feed`, and `laboratory strain`.
  Reason: A source type is a category of samples, not an entity.
- Removed: `biofilm` `skos:exactMatch` `ENVO:00002034` and `laboratory facility` `skos:exactMatch`
  `ENVO:01001406`, with their terms.
- Changed predicates:

  | Term         | Object           | Before        | After          |
  | ------------ | ---------------- | ------------- | -------------- |
  | `body fluid` | `UBERON:0006314` | `exactMatch`  | `narrowMatch`  |
  | `body fluid` | `ENVO:02000040`  | `narrowMatch` | `relatedMatch` |
  | `blood`      | `ENVO:02000020`  | `exactMatch`  | `closeMatch`   |
  | `urine`      | `ENVO:00002047`  | `exactMatch`  | `closeMatch`   |
  | `feces`      | `ENVO:00002003`  | `exactMatch`  | `closeMatch`   |
  | `wound`      | `MONDO:0021178`  | `broadMatch`  | `relatedMatch` |
  | `wastewater` | `ENVO:00002001`  | `exactMatch`  | `narrowMatch`  |
  | `sludge`     | `ENVO:00002044`  | `exactMatch`  | `broadMatch`   |
