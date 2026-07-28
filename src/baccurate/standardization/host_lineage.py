"""
Builds host_common_names, host_lineage_names, and host_lineage_taxids columns
by walking NCBI taxonomy (names.dmp / nodes.dmp) for each host taxid.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

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
    for tid, _rank, name in chain:
        taxids.append(tid)
        names.append(name)
    return taxids, names


# --- Column generation ---


@dataclass(frozen=True, slots=True)
class HostLineage:
    """The common names and lineage strings emitted as output columns for one host."""

    common_names: str
    lineage_names: str
    lineage_taxids: str


class HostLineageEnricher:
    """Load taxonomy and add lineage columns to host outcomes (before rows are written)."""

    def __init__(self, names_dmp: Path, nodes_dmp: Path) -> None:
        self._parent_rank, self._sciname, self._common = _load_ncbi_taxonomy(
            names_dmp,
            nodes_dmp,
        )
        self._allowed_ranks = set(RANK_ORDER)
        self._cache: dict[int, HostLineage] = {}

    def enrich(self, taxid: int) -> HostLineage:
        cached = self._cache.get(taxid)
        if cached is not None:
            return cached
        taxids, names = _walk_lineage(taxid, self._parent_rank, self._sciname)
        filtered = [
            (lineage_taxid, name)
            for lineage_taxid, name in zip(taxids, names, strict=True)
            if self._parent_rank.get(lineage_taxid, (0, ""))[1] in self._allowed_ranks
        ]
        result = HostLineage(
            common_names=",".join(self._common.get(taxid, [])),
            lineage_names=",".join(name for _, name in filtered),
            lineage_taxids=",".join(str(lineage_taxid) for lineage_taxid, _ in filtered),
        )
        self._cache[taxid] = result
        return result
