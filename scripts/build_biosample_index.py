"""Assemble ``data/raw/biosample_index.tsv``.

Output is a union of:
  - accessions in ``data/raw/id_lists/<pathogen_key>.tsv`` (from
    ``parse_biosample_xml.py``). Supplies ``pathogen_biosample`` and ``organism_value``.
  - accessions in the ATB metadata whose ``sylph_species`` maps to a target
    pathogen key. Supplies ``pathogen_ATB``, ``in_ATB`` and ``osf_tarball_filename``.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# make the package importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baccurate.atb import NA, build_keyword_maps, sylph_to_keyword
from baccurate.paths import DEFAULT_INDEX_TSV, RAW_DIR

log = logging.getLogger("build_biosample_index")

ID_LISTS_DIR = RAW_DIR / "id_lists"
COLUMNS = ["accession", "in_ATB", "pathogen_biosample", "pathogen_ATB", "taxid", "organism_value", "osf_tarball_filename"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_taxonomy_branch(id_lists_dir: Path) -> pd.DataFrame:
    """Concatenate per-pathogen-key TSVs into accession / pathogen_biosample / organism_value."""
    frames = []
    for f in sorted(id_lists_dir.glob("*.tsv")):
        if f.name == "manifest.tsv":
            continue
        pathogen_key = f.stem
        d = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        if d.empty:
            continue
        d = d.rename(columns={"organism": "organism_value"})
        d["pathogen_biosample"] = pathogen_key
        frames.append(d[["accession", "taxid", "pathogen_biosample", "organism_value"]])

    if not frames:
        return pd.DataFrame(columns=["accession", "taxid", "pathogen_biosample", "organism_value"])

    tax = pd.concat(frames, ignore_index=True)
    conflicts = tax[tax.duplicated("accession", keep=False)]
    if not conflicts.empty:
        n = conflicts["accession"].nunique()
        log.warning("%d accession(s) matched more than one pathogen-key query. Keeping first by file order", n)
    return tax.drop_duplicates("accession", keep="first")


def setup_logging(out_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "build_biosample_index.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("atb_metadata", type=Path, help="ATB metadata TSV (accession, osf_tarball_filename, sylph_species)")
    ap.add_argument("--id-lists-dir", type=Path, default=ID_LISTS_DIR, help="directory of per-pathogen-key TSVs")
    ap.add_argument("--output", type=Path, default=DEFAULT_INDEX_TSV, help="output index path")
    args = ap.parse_args()

    out_path: Path = args.output
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    log.info("run date: %s", date.today().isoformat())
    log.info("id_lists dir:  %s", args.id_lists_dir)
    log.info("atb_metadata:  %s", args.atb_metadata)
    log.info("  sha256: %s", sha256(args.atb_metadata))

    # taxonomy branch
    tax = load_taxonomy_branch(args.id_lists_dir)
    tax_acc = set(tax["accession"])
    pathogen_key_files = [f for f in args.id_lists_dir.glob("*.tsv") if f.name != "manifest.tsv"]
    log.info("NCBI taxonomy: %d accessions across %d pathogen-key files", len(tax_acc), len(pathogen_key_files))

    manifest = args.id_lists_dir / "manifest.tsv"
    if manifest.exists():
        m = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False)
        if "status" in m.columns:
            bad = m.loc[m["status"] != "ok", "pathogen_key"].tolist()
            if bad:
                log.warning("Manifest flags incomplete pathogen-key fetch(es). re-fetch recommended: %s", ", ".join(bad))

    # ATB branch
    atb = pd.read_csv(args.atb_metadata, sep="\t", dtype=str, keep_default_na=False)
    genus_map, species_map = build_keyword_maps()
    keyword_of = {s: sylph_to_keyword(s, genus_map, species_map) for s in atb["sylph_species"].drop_duplicates()}
    atb = atb.assign(_kw=atb["sylph_species"].map(keyword_of))
    tarball_by_acc = dict(zip(atb["accession"], atb["osf_tarball_filename"]))
    keyword_by_acc = dict(zip(atb["accession"], atb["_kw"]))
    atb_target_acc = set(atb.loc[atb["_kw"] != NA, "accession"])
    log.info("ATB records: %d | atb accessions mapping to a target pathogen key: %d", len(atb), len(atb_target_acc))

    # union
    scope = sorted(tax_acc | atb_target_acc)
    if not scope:
        log.error("No taxonomy id_lists and no ATB-target accessions found")
        return 1

    org_by_acc = dict(zip(tax["accession"], tax["organism_value"]))
    bio_by_acc = dict(zip(tax["accession"], tax["pathogen_biosample"]))
    taxid_by_acc = dict(zip(tax["accession"], tax["taxid"]))

    df = pd.DataFrame({"accession": scope})
    df["in_ATB"] = df["accession"].map(lambda a: "True" if a in tarball_by_acc else "False")
    df["pathogen_biosample"] = df["accession"].map(bio_by_acc).fillna(NA)
    df["pathogen_ATB"] = df["accession"].map(keyword_by_acc).fillna(NA)
    df["taxid"] = df["accession"].map(taxid_by_acc).fillna(NA).replace("", NA)  # NA for ATB-only records
    df["organism_value"] = df["accession"].map(org_by_acc).fillna(NA)
    df["osf_tarball_filename"] = df["accession"].map(tarball_by_acc).fillna(NA)
    df = df[COLUMNS]

    # sanity counts
    only_tax = len(tax_acc - atb_target_acc)
    only_atb = len(atb_target_acc - tax_acc)
    both = len(tax_acc & atb_target_acc)
    log.info("index rows: %d | taxonomy-only: %d | atb-only: %d | both: %d", len(df), only_tax, only_atb, both)
    log.info("in_ATB True: %d", int((df["in_ATB"] == "True").sum()))
    log.info("pathogen_biosample=NA (atb-only): %d | pathogen_ATB=NA: %d",
             int((df["pathogen_biosample"] == NA).sum()), int((df["pathogen_ATB"] == NA).sum()))
    for col in ("pathogen_biosample", "pathogen_ATB"):
        counts = df[col].value_counts().to_dict()
        log.info("%s counts: %s", col, {k: counts[k] for k in sorted(counts)})

    if out_path.exists():
        backup = out_dir / "biosample_index_old.tsv"
        os.replace(out_path, backup)
        log.info("renamed previous index -> %s", backup)
    df.to_csv(out_path, sep="\t", index=False)
    log.info("wrote index -> %s (%d rows)", out_path, len(df))
    log.info("  sha256: %s", sha256(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
