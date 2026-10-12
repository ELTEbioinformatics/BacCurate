# The BacCurate vocabulary

BacCurate uses a controlled vocabulary to map the values of `isolation_source` and its related attributes 
found in NCBI BioSample. It is designed for bacterial isolates in antimicrobial resistance (AMR) surveillance, with a 
focus on the ESKAPEE pathogens. 

The files in this directory define this vocabulary and its mappings to external ontologies.

| File                 | Content                                                                         |
| -------------------- | ------------------------------------------------------------------------------- |
| `terms.tsv`          | One row per term: identifier, label, facet, parent, definition, and scope note. |
| `synonyms.tsv`       | One row per synonym: term, synonym, and synonym type.                           |
| `relations.tsv`      | One row per relation: subject term, relation type, and object term.             |
| `obsolete_terms.tsv` | One row per obsolete term: identifier, label, replacements, and a note.         |
| `facets.yaml`        | The facets and the vocabulary version.                                          |
| `mappings.sssom.tsv` | The mappings from terms to classes of external ontologies.                      |
| `mappings.sssom.yml` | The metadata and the prefixes of the mapping set.                               |
| `baccurate.ttl`      | The SKOS release file, generated from the other files.                          |

## Model

The vocabulary is one [SKOS](https://www.w3.org/TR/skos-reference/) concept scheme.

This table gives the SKOS counterpart of each element.

| Element               | SKOS counterpart                                              |
| --------------------- | ------------------------------------------------------------- |
| Term (`term_id`)      | `skos:Concept`                                                |
| Label                 | `skos:prefLabel`                                              |
| `exact` synonym       | `skos:altLabel` and `oboInOwl:hasExactSynonym`                |
| `narrow` synonym      | `oboInOwl:hasNarrowSynonym`                                   |
| `broad` synonym       | `oboInOwl:hasBroadSynonym`                                    |
| `related` synonym     | `oboInOwl:hasRelatedSynonym`                                  |
| `definition`          | `skos:definition`                                             |
| `scope_note`          | `skos:scopeNote`                                              |
| Facet                 | `skos:Collection` in the scheme                               |
| Term without a parent | Top concept of the scheme (`skos:topConceptOf`)               |
| Parent with `kind_of` | `iso-thes:broaderGeneric`, a sub-property of `skos:broader`   |
| Parent with `part_of` | `iso-thes:broaderPartitive`, a sub-property of `skos:broader` |
| Relation              | A sub-property of `skos:related`                              |

The two `iso-thes` properties come from the [SKOS extension of ISO 25964](http://purl.org/iso25964/skos-thes). 
In ISO 25964, they represent generic (BTG) and partitive (BTP) hierarchical relationships.

The two `mappings.sssom.*` files combine to form a single [SSSOM](https://mapping-commons.github.io/sssom/) mapping set. 
These files link vocabulary terms to classes in external ontologies using standard SKOS mapping predicates. 
External classes are only used as the targets (objects) of these mappings; the vocabulary maintains its own independent 
hierarchy and internal relationships.

## Identifiers

Each term has a CURIE identifier with the prefix `BACC`, for example `BACC:0000001`.
The prefix `BACC` expands to `https://w3id.org/baccurate/BACC_`, so the IRI of `BACC:0000001` is
`https://w3id.org/baccurate/BACC_0000001`. The prefix `baccurate` expands to
`https://w3id.org/baccurate/`.

| Resource       | IRI                                                |
| -------------- | -------------------------------------------------- |
| Term           | `https://w3id.org/baccurate/BACC_0000001`          |
| Concept scheme | `https://w3id.org/baccurate/vocabulary`            |
| Facet          | `https://w3id.org/baccurate/vocabulary/<facet>`    |
| Relation type  | `https://w3id.org/baccurate/vocabulary/<relation>` |
| Mapping set    | `https://w3id.org/baccurate/vocabulary/mappings`   |

**Note:** The w3id redirects will be registered when the vocabulary is
published on [BioPortal](https://bioportal.bioontology.org/), and the term IRIs will then lead to
the BioPortal pages of the terms.

## Obsolete terms

`terms.tsv` contains only active terms, while retired terms are kept in `obsolete_terms.tsv`.

The `replaced_by` column in `obsolete_terms.tsv` contains active term IDs, separated by `;`, 
that collectively replace the obsolete term. If this column is empty, the obsolete term has no replacements. 
The `note` column records the date and reason for retirement.

When a sample value is only an obsolete ID, it resolves to its replacement terms as explicit terms, 
and term derivation is then applied to each replacement. For example, `BACC:0000024` (*caecal content*) resolves to 
the explicit terms *digestive tract content* and *caecum*. 
In contrast, `BACC:0000078` (*biofilm*) has no replacements and does not yield any terms.

An identifier is never reused, either for an active term or for an obsolete term.

## Synonyms

A synonym is another name of a term. Each row of `synonyms.tsv` defines one synonym using the columns `term_id`, 
`synonym`, and `synonym_type`. Rows are sorted by `term_id`, then by `synonym`.

`synonym_type` uses one of the four synonym scopes defined by [oboInOwl](http://www.geneontology.org/formats/oboInOwl):

| `synonym_type` | Meaning                                                                                              | Example                              |
|----------------|------------------------------------------------------------------------------------------------------|--------------------------------------|
| `exact`        | The synonym and the term have the same meaning.                                                      | `csf` → cerebrospinal fluid          |
| `narrow`       | The synonym refers to something more specific than the term, or to a part of it.                     | `human urine` → urine                |
| `broad`        | The synonym refers to something more general than the term, or to a whole that includes it.          | `throat` → pharynx                   |
| `related`      | The synonym refers to something related to the term, but is neither broader nor narrower in meaning. | `bal` → bronchoalveolar lavage fluid |

All four synonym types are used in direct matching.

The match key for a string is its lowercase form, with spaces and hyphens replaced by underscores. 
Match keys have to be unique across all labels and synonyms.

Strings that provide evidence for a term but are not alternative names for it, such as `blood culture` or `fecal`, 
are not recorded as synonyms. Instead, they are defined as direct-match values in `config/isolation_source.yaml`.

## Definitions and scope notes

`definition` explains the meaning of the term in a single sentence. It is required for any term that lacks a 
`skos:exactMatch` mapping, since no external class gives its meaning. For all other terms, it is 
optional. When a term has a parent, the definition starts with the parent's label. 
For example: "Respiratory secretion collected from the trachea by suction."

`scope_note` clarifies when to apply a term or what the term excludes. Following ANSI/NISO Z39.19 (§6.2.2), a scope 
note acts as guidance for the indexer (or in our case, guidance for the model). For example, the term `drainage fluid` 
uses the following definition and scope note:

| Column       | Text                                                                                                                                 |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `definition` | Body fluid drained from a wound, a body cavity, or a fluid collection.                                                               |
| `scope_note` | Use when the metadata does not name the fluid. If the metadata names the fluid, or names a lesion that implies it, use that instead. |

## Kinds of terms

Terms belonging to the `source_type` facet represent a category of samples, such as `animal host`. 
Terms belonging to the other eight facets represent specific entities, with the facet itself describing how the sample 
relates to that entity.

| Facet                    | A term stands for            |
| ------------------------ | ---------------------------- |
| `source_type`            | A category of samples        |
| `body_product`           | A substance from a host body |
| `body_site`              | An anatomical site           |
| `lesion`                 | A lesion                     |
| `environmental_material` | An environmental material    |
| `facility`               | A facility                   |
| `sampled_object`         | An object                    |
| `food_type`              | A food                       |
| `host_disease`           | A disease or a health state  |

## Standards bindings

The `standards_bindings` section in `facets.yaml` lists the corresponding fields from the 
[MIxS](https://genomicsstandardsconsortium.github.io/mixs/) and [PHA4GE](https://pha4ge.org/) 
([GENEPIO](https://genepio.org/)) specifications for each facet. These external fields capture the same type of value. 
Sometimes, a qualifier in parentheses is used to limit how closely the fields correspond. 
These standards bindings are included only in the source files and the table below.

| Facet                    | MIxS                             | PHA4GE/GENEPIO                            |
| ------------------------ | -------------------------------- | ----------------------------------------- |
| `source_type`            | `package/checklist choice`       | None                                      |
| `body_product`           | `host_body_product`              | `body product`                            |
| `body_site`              | `host_body_site`                 | `anatomical part`                         |
| `lesion`                 | None                             | None                                      |
| `environmental_material` | `env_medium`                     | `environmental material`                  |
| `facility`               | `env_local_scale (part)`         | `environmental site`                      |
| `sampled_object`         | `env_local_scale (part)`         | `environmental site`, `collection device` |
| `food_type`              | `food package food_product_type` | None                                      |
| `host_disease`           | `host_disease_stat`              | `host disease`                            |

## Parents

Each term has at most one parent within the same facet. 
The `parent_relation` column explains how the term relates to its parent, and it must be filled out whenever 
a `parent_id` is provided.

| `parent_relation` | Use when                                               | Example                                  |
|-------------------|--------------------------------------------------------|------------------------------------------|
| `kind_of`         | All instances of the term are instances of the parent. | `catheter` is a kind of `medical device` |
| `part_of`         | The term is part of the parent, regardless of context. | `rectum` is part of `intestine`          |

These two relations follow the generic and whole-part tests defined in 
[ANSI/NISO Z39.19](https://www.niso.org/publications/ansiniso-z3919-2005-r2010), §8.3.1 and §8.3.3.

In some cases the parent term has a broader meaning than its usual everyday sense. In these cases, the term's definition 
explains this broader scope. For example, `skin and soft tissue` covers either skin or soft tissue, and `wastewater` 
includes sludge from wastewater treatment. Despite these broader scopes, both still function as `kind_of` parents.

### One parent per term

Restricting each term to a single parent ensures that every term has a unique label path. If a term is also part of a 
second whole, it connects to that second whole using a `part_of` relation in `relations.tsv`. 
For example, `pharynx` has `upper respiratory tract` as its parent, and links to `digestive tract` via a relation.

This approach means `pharynx` and `oropharynx` deviate slightly from ANSI/NISO Z39.19 §8.3.3.2, which recommends using 
an associative link (RT) when a part belongs to multiple wholes. In this vocabulary, however, one whole serves as the 
parent while any additional wholes are linked via a `part_of` relation.

## Relations

A relation is a directed, non-hierarchical link between two terms, running from the term that carries it (the subject)
to its target (the object). Each row in `relations.tsv` represents one relation, defined by the columns `subject_id`, 
`relation`, and `object_id`. Rows are sorted by these three columns.

| `relation`            | Meaning                                                         | Standard counterpart | Example                                                 |
| --------------------- | --------------------------------------------------------------- | -------------------- | ------------------------------------------------------- |
| `located_in`          | The material is inside the site.                                | `RO:0001025`         | `blood` is located in `circulatory system`              |
| `produced_by`         | The site produces the material.                                 | `RO:0003001`         | `feces` is produced by `digestive tract`                |
| `collected_from`      | The specimen is taken from the site.                            | None (local)         | `tracheal aspirate` is collected from `trachea`         |
| `part_of`             | The site is part of a second whole outside its parent path.     | `BFO:0000050`        | `pharynx` is part of `digestive tract`                  |
| `has_part`            | The subject contains the object.                                | `BFO:0000051`        | `abscess` has part `pus`                                |
| `implies_source_type` | Every sample under the subject belongs to the source-type term. | None (local)         | `rhizosphere soil` implies the source type `plant host` |

`RO` refers to the [OBO Relations Ontology](https://oborel.github.io/), and `BFO` refers to the 
[Basic Formal Ontology](https://basic-formal-ontology.org/). The two local relation types do not have counterparts in 
these ontologies.

The vocabulary is considered invalid if `relations.tsv` contains any of the following:
- A row with an unknown term, an unknown relation type, or duplicate values of another row.
- An `implies_source_type` row where the object is not a `source_type` term.
- Any other type of row where the object is a `source_type` term.
- A row where the object is the same as the subject, an ancestor of the subject, or a descendant of the subject. 
  (Following SKOS integrity condition S27, `skos:related` must not overlap with `skos:broaderTransitive`.)

## Term derivation

The explicit terms of a sample are those directly selected by the model or matcher. 
Term derivation then adds inferred (derived) terms based on the following rules:

- Every ancestor of an explicit term, following both `kind_of` and `part_of` relations equally.
- The object of every relation belonging to an explicit term, along with all ancestors of that object.
- The object of every `implies_source_type` relation belonging to an ancestor of an explicit term, along with all ancestors of that object.

Only these specific relations trigger derivation. In addition, certain facets and terms imply specific source types automatically:
- `body_product`, `body_site`, `lesion`, and `host_disease` imply `host-associated`.
- `environmental_material` and `sampled_object` imply `environmental`. However, `medical device` and its descendants 
  do not imply any source type.
- `food_type` implies `food or feed`.

Term derivation is a specific rule of the BacCurate pipeline. 
Standard SKOS reasoning alone cannot generate these derived terms because `skos:broader` is not transitive.

## Prepare a release

Load the Turtle file with [`rdflib`](https://rdflib.readthedocs.io/):

```python
from rdflib import Graph
from rdflib.namespace import SKOS

graph = Graph().parse("data/reference/ontology/baccurate.ttl")
for concept, label in graph.subject_objects(SKOS.prefLabel):
    print(concept, label)
```

Merge the two files of the mapping set with [`sssom`](https://mapping-commons.github.io/sssom-py/):

```bash
uvx --from sssom sssom parse data/reference/ontology/mappings.sssom.tsv   -m data/reference/ontology/mappings.sssom.yml -o mappings.sssom.tsv
```

### Release check

After each change to the source files:

1. Run `uv run python scripts/build_vocabulary_release.py`.
2. Run `uv run pytest`.
3. Run `sssom parse` as above, then run `sssom validate` on the result:

   ```bash
   uvx --from sssom sssom validate mappings.sssom.tsv
   ```

The tests compare `baccurate.ttl` with the source files, and check the SKOS integrity conditions
S13, S14, S27, and S46.

### Versions

`vocabulary_version` in `facets.yaml`, and `mapping_set_version` and `subject_source_version` in
`mappings.sssom.yml`, always have the same value. The loader rejects a difference. Versioning system:

| Part  | Increase when                                                                        |
| ----- | ------------------------------------------------------------------------------------ |
| Major | An ID becomes obsolete, or the parent, a relation, or the meaning of a term changes. |
| Minor | A term, a synonym, or a mapping is added.                                            |
| Patch | A text change keeps every meaning.                                                   |

Each version has a section in [`CHANGELOG.md`](CHANGELOG.md).

### License

The files in this directory, `baccurate.ttl` included, are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [`LICENSE`](LICENSE). The code of BacCurate is MIT-licensed.