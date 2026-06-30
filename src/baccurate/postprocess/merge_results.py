"""
Merges the per-pathogen, per-attribute standardized TSVs into a single table,
joining on accession and flagging accessions present in the ATB ID lists.
"""

import csv
import logging
from collections import defaultdict
from functools import reduce
from pathlib import Path

import pandas as pd

from baccurate.pathogens import scientific_name

logger = logging.getLogger(__name__)

FINAL_COLUMN_ORDER = [
    "accession", "pathogen", "pathogen_sci_name", "in_ATB", "bioproject",
    "date_attr", "date_val", "date_start", "date_end", "date_score",
    "loc_attr", "loc_val", "loc_continent", "loc_UNregion", "loc_country", "loc_other",
    "iso_attr", "iso_val", "iso_host", "iso_terms", "iso_display_term", "iso_ontology_id",
    "host_attr", "host_val", "host_taxid", "host_sci_name",
    "host_common_names", "host_lineage_names", "host_lineage_taxids",
    "host_score", "host_low_conf",
]


def _atb_ids_by_pathogen(index_path: Path) -> dict[str, set[str]]:
    """Map pathogen key -> set of ATB accessions, from biosample_index.tsv pathogen_ATB column."""
    by_pathogen: dict[str, set[str]] = defaultdict(set)
    with index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("in_ATB") or "").strip() != "True":
                continue
            accession = (row.get("accession") or "").strip()
            pathogen = (row.get("pathogen_ATB") or "").strip()
            if accession and pathogen:
                by_pathogen[pathogen].add(accession)
    return by_pathogen


def merge_results(
    names: list[str],
    output_dir: Path,
    index_path: Path,
    merged_output_path: Path,
    pipelines: list[dict],
    extracted_metadata_path: Path,
) -> None:
    """Merge every pathogen's pipeline outputs into merged_output_path as one TSV."""
    logger.info("Merging output files...")
    all_data = []

    atb_by_pathogen = _atb_ids_by_pathogen(index_path)

    for name in names:
        atb_ids = atb_by_pathogen.get(name, set())
        if not atb_ids:
            logger.info("No ATB accessions found for pathogen %r in %s", name, index_path)

        pathogen_dfs = []
        for pipeline in pipelines:
            file_path = output_dir / name / pipeline["output"]

            if not file_path.exists():
                logger.warning("Output file not found: %s", file_path)
                continue

            read_kwargs = {"sep": "\t"}
            if pipeline["name"] == "host":
                read_kwargs["dtype"] = {"host_taxid": "Int64"}
            df = pd.read_csv(file_path, **read_kwargs)
            pathogen_dfs.append(df)

        if not pathogen_dfs:
            continue

        # Merge dataframes
        merged_pathogen_df = reduce(
            lambda left, right: pd.merge(left, right, on="accession", how="outer"), pathogen_dfs
        )

        merged_pathogen_df["pathogen"] = name
        merged_pathogen_df["pathogen_sci_name"] = scientific_name(name)
        merged_pathogen_df["in_ATB"] = merged_pathogen_df["accession"].isin(atb_ids)

        all_data.append(merged_pathogen_df)

        atb_match_count = merged_pathogen_df["in_ATB"].sum()
        logger.info(
            "Pathogen %s: loaded %d rows, %d with ATB matches.",
            name,
            len(merged_pathogen_df),
            atb_match_count,
        )

    if not all_data:
        logger.warning("No data found to merge.")
        return

    merged_df = pd.concat(all_data, ignore_index=True)

    if "host_taxid" in merged_df.columns:
        merged_df["host_taxid"] = merged_df["host_taxid"].astype("Int64")

    bioproject = pd.read_csv(
        extracted_metadata_path, sep="\t", usecols=["accession", "bioproject"], dtype=str
    )
    merged_df = merged_df.merge(bioproject, on="accession", how="left")

    ordered = [c for c in FINAL_COLUMN_ORDER if c in merged_df.columns]
    merged_df = merged_df[ordered]

    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_output_path, sep="\t", index=False)
    logger.info("Merged table with %d rows saved to %s", len(merged_df), merged_output_path)
