# Geographic location standardization

Map location annotations from sample metadata to standardized country and
continent names.

[Source](https://github.com/kadan02/BacCurate/blob/main/src/baccurate/standardizers/location.py)

## Contents

- [Usage](#usage)
- [Configuration](#configuration)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Data usage recommendations](#data-usage-recommendations)
- [Methods](#methods)
  - [Workflow](#workflow)
  - [Reference vocabularies](#reference-vocabularies)
  - [Matching cascade](#matching-cascade)
  - [Coordinate decoding](#coordinate-decoding)
  - [Country-name matching](#country-name-matching)
  - [INSDC normalization](#insdc-normalization)
  - [Cache](#cache)
  - [LLM fallback](#llm-fallback)

## Usage

Run the location pipeline for one or more pathogens with the `loc` attribute:

```bash
uv run baccurate <pathogen> --attribute loc
```
See the [main README](../README.md#usage) for installation and the full set of options.


## Configuration

[`config/location.yaml`](../config/location.yaml) contains:

- `coordinate_attributes`: Parsing hints for values already identified as
  location candidates.
- `insdc_country_map`: Remaps `country_converter` names to their INSDC spelling (see below).
- `llm_system_prompt`: System prompt for the LLM fallback.
- `llm_user_prompt_template`: User-prompt template

LLM connection details are read from environment variables (`.env` at the root):

| Variable    | Purpose                     |
|-------------|-----------------------------|
| `API_KEY`   | API key for the LLM service |
| `SERVER`    | OpenAI-compatible base URL  |
| `LLM_MODEL` | Model identifier            |

If any of these are not set the LLM fallback is disabled and unresolved values return `NA`.

## Inputs

| Column          | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `accession`     | Record ID                                                        |
| `loc_attr_orig` | `\|\|`-separated attribute names                                 |
| `loc_val_orig`  | `\|\|`-separated values, paired by position with `loc_attr_orig` |

## Outputs

| Column          | Description                                        |
|-----------------|----------------------------------------------------|
| `accession`     | Record ID                                          |
| `loc_attr_orig` | Unstandardized input attribute(s)                  |
| `loc_val_orig`  | Unstandardized input value(s)                      |
| `loc_UNregion`  | UN geoscheme region of the resolved country        |
| `loc_country`   | Standardized country name                          |
| `loc_other`     | City or sub-country region if available, else `NA` |

## Data usage recommendations

`loc_country` is always drawn from the INSDC Geographical Location List, so the
country field is safe to join against other INSDC-aligned data.

`loc_other` is not standardized against a controlled vocabulary and should be treated as a free-text hint.


## Methods

## Workflow

TODO: add charts/location.png flowchart

Each `||`-separated value is tried in order. A value is decoded as coordinates
if it looks like one or its attribute is configured as a coordinate field,
otherwise it is resolved by name lookup. Values that name lookup cannot resolve
are pooled and sent to the LLM in a single call. If the record has multiple values
and one produced a deterministic match, it is not included in the LLM call.
Matches carrying a sublocation syntax are preferred over those without.

### Reference vocabularies

Country and continent name lookup uses [`country_converter`](https://github.com/IndEcol/country_converter). 
Coordinate decoding uses [`reverse_geocode`](https://github.com/richardpenman/reverse_geocode). 

Final country names are limited to the INSDC Geographical Location List, found in `data/reference/geo_loc_list.txt`.

### Matching workflow

For each attribute/value pair:

1. Coordinate decoding is attempted if the value matches a coordinate
   pattern or the attribute is listed in `coordinate_attributes`.
2. Otherwise the value goes through country-name matching.
3. Pairs that name matching cannot resolve (and which are not equivalent to `NA`) are
   queued for a configurable **LLM** API as a fallback.

After resolving "candidates" for a record are collected. If one follows the 
`Country:Sublocation` format (usually more accurate), that is returned in preference 
to the one without, otherwise the first match wins.

### Coordinate decoding

Common coordinate formatted strings (e.g. `-31.50, -52.31`) are parsed to
`(latitude, longitude)`, range-checked, and reverse-geocoded to a country and
city. The country is then standardized like any other name, while the city becomes the
value in the `sublocation` column.

### Country-name matching

The value is split on `:`, `;`, and `,`, and each part is tried in order against
`country_converter` (so `"France: Paris"` and `"USA, California"` resolve on
their first token). The continent is derived from the resolved country.

### INSDC normalization

`country_converter` short names are remapped to their INSDC spelling via
`insdc_country_map` (e.g. `United States` → `USA`, `Vietnam` → `Viet Nam`). A
country that is absent from the INSDC vocabulary after remapping is dropped 
from the output.

### Cache

LLM-resolved values are cached in SQLite, keyed by the SHA-256 of the prompt's
attribute/value context, storing the resolved country and continent. To force
reprocessing, delete `data/cache/llm_loc_cache.db`. Deterministic matches
(coordinates, name lookup) are not cached, as they are fast to compute.

### LLM fallback

The unresolved attribute/value pairs are joined into one context string and sent
in a single call (`temperature = 0`, fixed seed). The model returns JSON with
`country` and `continent` keys, or arrays when several countries are named. The
returned country is then re-standardized through `country_converter` and INSDC
normalization. On any API error or unparsable response the value resolves to
`NA`.
