"""
Builds host_common_names, host_lineage_names, and host_lineage_taxids columns
by walking NCBI taxonomy (names.dmp / nodes.dmp) for each host taxid.
"""

import logging
from pathlib import Path

import pandas as pd

from baccurate.paths import DEFAULT_NAMES_DMP, DEFAULT_NODES_DMP

logger = logging.getLogger(__name__)


# --- Constants ---

# Ranks to include in host_lineage_names / host_lineage_taxids, root-to-tip.
RANK_ORDER = [
    "subspecies",
    "species",
    "subgenus",
    "genus",
    "tribe",
    "subfamily",
    "family",
    "superfamily",
    "parvorder",
    "infraorder",
    "suborder",
    "order",
    "superorder",
    "infraclass",
    "subclass",
    "class",
    "clade",
    "phylum",
    "kingdom",
    "no rank",
]

NAMES_DMP = DEFAULT_NAMES_DMP
NODES_DMP = DEFAULT_NODES_DMP


# --- Taxonomy parsing ---


def _load_ncbi_taxonomy(
    names_path: Path,
    nodes_path: Path,
) -> tuple[dict[int, tuple[int, str]], dict[int, str], dict[int, list[str]]]:
    """
    Stream-parse names.dmp and nodes.dmp into three lookups: parent_rank
    (taxid -> (parent_taxid, rank)), sciname (taxid -> scientific name), and
    common (taxid -> genbank/common name list, in NCBI listing order).
    """
    parent_rank: dict[int, tuple[int, str]] = {}
    sciname: dict[int, str] = {}
    common: dict[int, list[str]] = {}

    logger.info("Loading NCBI nodes.dmp from %s", nodes_path)
    with nodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            # Format: taxid | parent_taxid | rank |
            if len(parts) < 3:
                continue
            try:
                taxid = int(parts[0])
                parent = int(parts[1])
            except ValueError:
                continue
            rank = parts[2]
            parent_rank[taxid] = (parent, rank)

    logger.info("Loading NCBI names.dmp from %s", names_path)
    with names_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            # Format: taxid | name | unique_name | name_class
            if len(parts) < 4:
                continue
            try:
                taxid = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            name_class = parts[3]
            if name_class == "scientific name":
                sciname.setdefault(taxid, name)
            elif name_class in ("genbank common name", "common name"):
                common.setdefault(taxid, []).append(name)

    logger.info(
        "Loaded %d nodes, %d scientific names, %d taxa with common names.",
        len(parent_rank),
        len(sciname),
        len(common),
    )
    return parent_rank, sciname, common


def _walk_lineage(
    taxid: int,
    parent_rank: dict[int, tuple[int, str]],
    sciname: dict[int, str],
) -> tuple[list[int], list[str]]:
    """Walk from taxid up to the root and return (taxids, names) in root-to-tip order."""
    chain: list[tuple[int, str, str]] = []  # (taxid, rank, sciname)
    current = taxid
    seen: set[int] = set()
    while current and current != 1 and current not in seen:  # drop root
        seen.add(current)
        info = parent_rank.get(current)
        if info is None:
            break
        parent, rank = info
        chain.append((current, rank, sciname.get(current, "")))
        current = parent

    chain.reverse()  # root-to-tip
    taxids: list[int] = []
    names: list[str] = []
    for tid, rank, name in chain:
        taxids.append(tid)
        names.append(name)
    return taxids, names


# --- Column generation ---


def add_lineage_columns(
    merged_path: Path,
    names_dmp: Path,
    nodes_dmp: Path,
) -> None:
    """Add host_common_names, host_lineage_names, host_lineage_taxids columns to the merged TSV."""
    allowed_ranks = set(RANK_ORDER)

    # bioproject is an integer id with blanks; read as str so pandas doesn't
    # infer float64 and write back trailing ".0" when re-saving the merged TSV.
    df = pd.read_csv(merged_path, sep="\t", dtype={"host_taxid": "Int64", "bioproject": str})

    parent_rank, sciname, common = _load_ncbi_taxonomy(names_dmp, nodes_dmp)

    common_cache: dict[int, str] = {}
    lineage_cache: dict[int, tuple[str, str]] = {}

    def _common_for(taxid: int) -> str:
        if taxid in common_cache:
            return common_cache[taxid]
        result = ",".join(common.get(taxid, []))
        common_cache[taxid] = result
        return result

    def _lineage_for(taxid: int) -> tuple[str, str]:
        if taxid in lineage_cache:
            return lineage_cache[taxid]
        taxids, names = _walk_lineage(taxid, parent_rank, sciname)
        # Filter again to RANK_ORDER membership in case the dmp carries ranks
        # outside the configured set entirely.
        filtered = [
            (t, n) for t, n in zip(taxids, names) if parent_rank.get(t, (0, ""))[1] in allowed_ranks
        ]
        names_str = ",".join(n for _, n in filtered)
        taxids_str = ",".join(str(t) for t, _ in filtered)
        result = (names_str, taxids_str)
        lineage_cache[taxid] = result
        return result

    common_col: list[str] = []
    lineage_names_col: list[str] = []
    lineage_taxids_col: list[str] = []

    for taxid in df["host_taxid"]:
        if pd.isna(taxid):
            common_col.append("")
            lineage_names_col.append("")
            lineage_taxids_col.append("")
            continue
        tid = int(taxid)
        common_col.append(_common_for(tid))
        names_str, taxids_str = _lineage_for(tid)
        lineage_names_col.append(names_str)
        lineage_taxids_col.append(taxids_str)

    df["host_common_names"] = common_col
    df["host_lineage_names"] = lineage_names_col
    df["host_lineage_taxids"] = lineage_taxids_col

    df.to_csv(merged_path, sep="\t", index=False)
    logger.info(
        "Added lineage columns to %s (%d unique taxids walked).",
        merged_path,
        len(lineage_cache),
    )
