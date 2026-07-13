"""Parses the full NCBI BioSample dump (`biosample_set.xml.gz`).

Produces:
  1. ``data/raw/id_lists/<pathogen_key>.tsv`` (accession, taxid, organism)
  2. ``data/raw/biosamples.xml.gz`` a filtered subset of the dump
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from lxml import etree

# make the package importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baccurate.atb import NA, build_keyword_maps, sylph_to_keyword
from baccurate.pathogens import load_pathogens
from baccurate.paths import DEFAULT_NODES_DMP, DEFAULT_XML_INPUT, RAW_DIR
from baccurate.utils.compressed_io import open_binary, open_text

log = logging.getLogger("parse_biosample_xml")

DEFAULT_DUMP = RAW_DIR / "biosample_set.xml.gz"
DEFAULT_ATB = RAW_DIR / "atb_2025-05.tsv"
DEFAULT_MERGED_DMP = DEFAULT_NODES_DMP.parent / "merged.dmp"
ID_LISTS_DIR = RAW_DIR / "id_lists"
PROGRESS_EVERY = 2_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_seeds() -> tuple[dict[int, str], list[str], dict[str, str]]:
    """Return (seed_taxid -> pathogen_key), the pathogen-key order, and pathogen_key -> space-joined taxids."""
    seeds: dict[int, str] = {}
    pathogen_keys: list[str] = []
    taxids_of: dict[str, str] = {}
    for p in load_pathogens().values():
        pathogen_keys.append(p.key)
        taxids_of[p.key] = " ".join(str(t) for t in p.taxids)
        for t in p.taxids:
            seeds[t] = p.key
    return seeds, pathogen_keys, taxids_of


def build_closure(nodes_dmp: Path, merged_dmp: Path, seeds: dict[int, str]) -> dict[int, str]:
    """Map every taxid in the subtree of a seed to that seed's pathogen key."""
    log.info("loading %s", nodes_dmp)
    children: dict[int, list[int]] = defaultdict(list)
    with nodes_dmp.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("|", 2)
            if len(parts) < 2:
                continue
            try:
                taxid = int(parts[0])
                parent = int(parts[1])
            except ValueError:
                continue
            if taxid != parent:  # root points at itself; don't self-loop the BFS
                children[parent].append(taxid)

    closure: dict[int, str] = {}
    for seed, pathogen_key in seeds.items():
        stack = deque([seed])
        while stack:
            t = stack.popleft()
            prev = closure.get(t)
            if prev is not None:
                if prev != pathogen_key:
                    log.warning("taxid %d in subtrees of both %s and %s; keeping %s", t, prev, pathogen_key, prev)
                continue
            closure[t] = pathogen_key
            stack.extend(children.get(t, ()))
    del children

    # Retired taxids: a record may still carry an old id that was merged into a live one.
    if merged_dmp.exists():
        added = 0
        with merged_dmp.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("|", 2)
                if len(parts) < 2:
                    continue
                try:
                    old = int(parts[0])
                    new = int(parts[1])
                except ValueError:
                    continue
                pathogen_key = closure.get(new)
                if pathogen_key is not None and old not in closure:
                    closure[old] = pathogen_key
                    added += 1
        log.info("merged.dmp: mapped %d retired taxid(s) into scope", added)
    else:
        log.warning("merged.dmp not found at %s. Retired taxids won't resolve", merged_dmp)

    log.info("closure: %d taxids across %d pathogen keys", len(closure), len(set(closure.values())))
    return closure


def load_atb_scope(atb_path: Path) -> set[str]:
    """Accessions whose sylph_species maps to a target pathogen key."""
    log.info("loading ATB scope from %s", atb_path)
    genus_map, species_map = build_keyword_maps()
    keyword_cache: dict[str, str] = {}
    scope: set[str] = set()
    with atb_path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        acc_i = header.index("accession")
        sp_i = header.index("sylph_species")
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) <= sp_i:
                continue
            sp = row[sp_i]
            kw = keyword_cache.get(sp)
            if kw is None:
                kw = sylph_to_keyword(sp, genus_map, species_map)
                keyword_cache[sp] = kw
            if kw != NA:
                scope.add(row[acc_i])
    log.info("ATB scope: %d accessions map to a target pathogen key", len(scope))
    return scope


def stream(
    dump: Path,
    closure: dict[int, str],
    atb_scope: set[str],
    pathogen_keys: list[str],
    out_dir: Path,
    subset_path: Path | None,
) -> dict[str, int]:
    """Single pass over the dump: write per-pathogen-key id_lists and (optionally) the union-scope subset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {m: (out_dir / f"{m}.tsv").open("w", encoding="utf-8", newline="\n") for m in pathogen_keys}
    for h in handles.values():
        h.write("accession\ttaxid\torganism\n")
    counts = dict.fromkeys(pathogen_keys, 0)

    subset = None
    subset_kept = 0
    if subset_path is not None:
        subset = open_text(subset_path, "w", newline="\n")
        subset.write('<?xml version="1.0" encoding="UTF-8"?>\n<BioSampleSet>\n')

    seen = 0
    try:
        with open_binary(dump) as fh:
            context = etree.iterparse(fh, events=("end",), tag="BioSample", huge_tree=True)
            for _event, elem in context:
                seen += 1
                accession = elem.get("accession")
                pathogen_key = None
                if accession:
                    org = elem.find("Description/Organism")
                    if org is not None:
                        raw = org.get("taxonomy_id")
                        if raw and raw.isdigit():
                            pathogen_key = closure.get(int(raw))
                            if pathogen_key is not None:
                                handles[pathogen_key].write(f"{accession}\t{raw}\t{org.get('taxonomy_name', '')}\n")
                                counts[pathogen_key] += 1
                    if subset is not None and (pathogen_key is not None or accession in atb_scope):
                        subset.write(etree.tostring(elem, encoding="unicode", with_tail=False))
                        subset.write("\n")
                        subset_kept += 1

                elem.clear()
                parent = elem.getparent()
                if parent is not None:
                    while elem.getprevious() is not None:
                        del parent[0]
                if seen % PROGRESS_EVERY == 0:
                    log.info("… %d records scanned, %d matched taxonomy, %d in subset", seen, sum(counts.values()), subset_kept)
            del context
    finally:
        for h in handles.values():
            h.close()
        if subset is not None:
            subset.write("</BioSampleSet>\n")
            subset.close()

    log.info("done: %d records scanned, %d matched taxonomy, %d written to subset", seen, sum(counts.values()), subset_kept)
    return counts


def write_manifest(out_dir: Path, counts: dict[str, int], taxids_of: dict[str, str], dump: Path) -> None:
    manifest = out_dir / "manifest.tsv"
    run = date.today().isoformat()
    src = dump.name
    with manifest.open("w", encoding="utf-8", newline="\n") as f:
        f.write("pathogen_key\trun_date\ttaxids\tsource\trows\tstatus\tsha256\n")
        for pathogen_key, n in counts.items():
            digest = sha256(out_dir / f"{pathogen_key}.tsv")
            f.write(f"{pathogen_key}\t{run}\t{taxids_of[pathogen_key]}\t{src}\t{n}\tok\t{digest}\n")
    log.info("wrote manifest -> %s", manifest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?", type=Path, default=DEFAULT_DUMP, help="BioSample dump (.xml or .xml.gz)")
    ap.add_argument("--atb", type=Path, default=DEFAULT_ATB, help="ATB metadata TSV")
    ap.add_argument("--nodes-dmp", type=Path, default=DEFAULT_NODES_DMP)
    ap.add_argument("--merged-dmp", type=Path, default=DEFAULT_MERGED_DMP)
    ap.add_argument("--id-lists-dir", type=Path, default=ID_LISTS_DIR)
    ap.add_argument("--subset", type=Path, default=DEFAULT_XML_INPUT, help="filtered subset XML for the downstream stage")
    ap.add_argument("--no-subset", action="store_true", help="skip regenerating the filtered subset XML")
    args = ap.parse_args()

    args.id_lists_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.id_lists_dir / "parse_biosample_xml.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    for p in (args.dump, args.nodes_dmp):
        if not p.exists():
            log.error("required input not found: %s", p)
            return 1

    log.info("dump: %s (%d bytes)", args.dump, args.dump.stat().st_size)
    seeds, pathogen_keys, taxids_of = target_seeds()
    closure = build_closure(args.nodes_dmp, args.merged_dmp, seeds)

    subset_path = None if args.no_subset else args.subset
    atb_scope: set[str] = set()
    if subset_path is not None:
        if not args.atb.exists():
            log.error("ATB metadata not found: %s (pass --no-subset to skip)", args.atb)
            return 1
        atb_scope = load_atb_scope(args.atb)

    counts = stream(args.dump, closure, atb_scope, pathogen_keys, args.id_lists_dir, subset_path)
    write_manifest(args.id_lists_dir, counts, taxids_of, args.dump)

    log.info("per-pathogen-key rows: %s", {m: counts[m] for m in pathogen_keys})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
