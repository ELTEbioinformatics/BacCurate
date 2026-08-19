"""Assemble the BioSample index and its build log.

Builds the index from:
  - accessions in ``data/raw/id_lists/<pathogen_key>.tsv`` (from
    ``parse_biosample_xml.py``). Supplies ``pathogen_biosample`` and ``organism_value``.
  - accessions in the ATB metadata whose ``sylph_species`` maps to a target
    pathogen key. Supplies ``pathogen_ATB``, ``in_ATB`` and ``osf_tarball_filename``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# make the package importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baccurate.pathogen_registry.registry import PathogenRegistry, load_pathogen_registry
from baccurate.pathogen_registry.species_label_matching import (
    NA,
    build_keyword_maps,
    sylph_to_keyword,
)
from baccurate.paths import (
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_GENBANK_ASSEMBLY_SNAPSHOT_MANIFEST,
    DEFAULT_INDEX_TSV,
    DEFAULT_REFSEQ_ASSEMBLY_SNAPSHOT_MANIFEST,
    DEFAULT_SRA_SNAPSHOT_MANIFEST,
    RAW_DIR,
)
from baccurate.provenance.source_snapshot import SourceSnapshotManifest, sha256_file

log = logging.getLogger("build_biosample_index")

ID_LISTS_DIR = RAW_DIR / "id_lists"
ATB_METADATA = RAW_DIR / "atb_2025-05.tsv"
SEQUENCE_ACCESSIONS_DIR = RAW_DIR / "sequence_accessions"
SEQUENCE_ACCESSION_INTERMEDIATES = (
    ("sra_runs.tsv.gz", "sra_run_accessions"),
    ("genbank_assemblies.tsv.gz", "genbank_assembly_accessions"),
    ("refseq_assemblies.tsv.gz", "refseq_assembly_accessions"),
)
COLUMNS = [
    "accession",
    "in_ATB",
    "pathogen_biosample",
    "pathogen_ATB",
    "taxid",
    "organism_value",
    "osf_tarball_filename",
    "sra_run_accessions",
    "genbank_assembly_accessions",
    "refseq_assembly_accessions",
]


def load_taxonomy_branch(
    id_lists_dir: Path,
    pathogen_registry: PathogenRegistry,
) -> pd.DataFrame:
    """Load registered pathogen-key TSVs, resolving overlaps in registry order.

    Missing registered files are skipped, and unregistered TSVs are excluded.
    """
    frames = []
    for pathogen_key in pathogen_registry.pathogen_keys:
        f = id_lists_dir / f"{pathogen_key}.tsv"
        if not f.exists():
            continue
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
        log.warning(
            "%d accession(s) matched more than one pathogen-key query. "
            "Keeping first by registry order",
            n,
        )
    return tax.drop_duplicates("accession", keep="first")


def load_biosample_linked_accession_column(
    directory: Path,
    filename: str,
    sequence_accession_column: str,
) -> pd.Series:
    """Load one filtered SRA run or genome assembly accession column.

    Require the expected ordered columns, non-empty values, and one row per BioSample accession.
    Return the accession column indexed by BioSample accession.
    """
    path = directory / filename
    intermediate_table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected_columns = ["accession", sequence_accession_column]
    if list(intermediate_table.columns) != expected_columns:
        raise ValueError(f"{path}: expected ordered columns {expected_columns}")
    if intermediate_table[expected_columns].eq("").any(axis=None):
        raise ValueError(f"{path}: missing values are not allowed")
    if intermediate_table["accession"].duplicated().any():
        raise ValueError(f"{path}: duplicate BioSample accession")
    log.info("sequence intermediate: %s | sha256: %s", path, sha256_file(path))
    return intermediate_table.set_index("accession")[sequence_accession_column]


def setup_logging(out_dir: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)
    for handler in (
        logging.FileHandler(out_dir / "build_biosample_index.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--id-lists-dir", type=Path, default=ID_LISTS_DIR, help="directory of per-pathogen-key TSVs"
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_INDEX_TSV, help="output index path")
    ap.add_argument(
        "--biosample-manifest",
        type=Path,
        default=DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
        help="BioSample source snapshot manifest",
    )
    ap.add_argument(
        "--sra-manifest",
        type=Path,
        default=DEFAULT_SRA_SNAPSHOT_MANIFEST,
        help="SRA source snapshot manifest",
    )
    ap.add_argument(
        "--genbank-manifest",
        type=Path,
        default=DEFAULT_GENBANK_ASSEMBLY_SNAPSHOT_MANIFEST,
        help="GenBank Assembly source snapshot manifest",
    )
    ap.add_argument(
        "--refseq-manifest",
        type=Path,
        default=DEFAULT_REFSEQ_ASSEMBLY_SNAPSHOT_MANIFEST,
        help="RefSeq Assembly source snapshot manifest",
    )
    args = ap.parse_args()

    out_path: Path = args.output
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    log.info("run date: %s", date.today().isoformat())
    log.info("id_lists dir:  %s", args.id_lists_dir)
    log.info("atb_metadata:  %s", ATB_METADATA)
    log.info("  sha256: %s", sha256_file(ATB_METADATA))

    manifests = {
        "BioSample": SourceSnapshotManifest.load(args.biosample_manifest),
        "SRA": SourceSnapshotManifest.load(args.sra_manifest),
        "GenBank Assembly": SourceSnapshotManifest.load(args.genbank_manifest),
        "RefSeq Assembly": SourceSnapshotManifest.load(args.refseq_manifest),
    }
    for label, snapshot_manifest in manifests.items():
        log.info(
            "%s manifest: snapshot_id=%s | metadata_reference_date=%s",
            label,
            snapshot_manifest.snapshot_id,
            snapshot_manifest.metadata_reference_date,
        )
    biosample_reference_date = manifests["BioSample"].metadata_reference_date
    for label in ("SRA", "GenBank Assembly", "RefSeq Assembly"):
        if manifests[label].metadata_reference_date < biosample_reference_date:
            raise ValueError(
                f"{label} metadata reference date "
                f"{manifests[label].metadata_reference_date} predates BioSample "
                f"metadata reference date {biosample_reference_date}"
            )

    sequence_accessions_by_column = {
        column: load_biosample_linked_accession_column(
            SEQUENCE_ACCESSIONS_DIR,
            filename,
            column,
        )
        for filename, column in SEQUENCE_ACCESSION_INTERMEDIATES
    }

    pathogen_registry = load_pathogen_registry()

    # taxonomy branch
    tax = load_taxonomy_branch(args.id_lists_dir, pathogen_registry)
    tax_acc = set(tax["accession"])
    pathogen_key_files = [
        args.id_lists_dir / f"{pathogen_key}.tsv"
        for pathogen_key in pathogen_registry.pathogen_keys
        if (args.id_lists_dir / f"{pathogen_key}.tsv").exists()
    ]
    log.info(
        "NCBI taxonomy: %d accessions across %d pathogen-key files",
        len(tax_acc),
        len(pathogen_key_files),
    )

    manifest = args.id_lists_dir / "manifest.tsv"
    if manifest.exists():
        m = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False)
        if "status" in m.columns:
            bad = m.loc[
                (m["status"] != "ok") & m["pathogen_key"].isin(pathogen_registry.pathogen_keys),
                "pathogen_key",
            ].tolist()
            if bad:
                log.warning(
                    "Manifest flags incomplete pathogen-key fetch(es). re-fetch recommended: %s",
                    ", ".join(bad),
                )

    # ATB branch
    atb = pd.read_csv(ATB_METADATA, sep="\t", dtype=str, keep_default_na=False)
    genus_map, species_map = build_keyword_maps(pathogen_registry)
    keyword_of = {
        s: sylph_to_keyword(s, genus_map, species_map)
        for s in atb["sylph_species"].drop_duplicates()
    }
    atb = atb.assign(_kw=atb["sylph_species"].map(keyword_of))
    tarball_by_acc = dict(zip(atb["accession"], atb["osf_tarball_filename"]))
    keyword_by_acc = dict(zip(atb["accession"], atb["_kw"]))
    atb_target_acc = set(atb.loc[atb["_kw"] != NA, "accession"])
    log.info(
        "ATB records: %d | atb accessions mapping to a target pathogen key: %d",
        len(atb),
        len(atb_target_acc),
    )

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
    df["taxid"] = (
        df["accession"].map(taxid_by_acc).fillna(NA).replace("", NA)
    )  # NA for ATB-only records
    df["organism_value"] = df["accession"].map(org_by_acc).fillna(NA)
    df["osf_tarball_filename"] = df["accession"].map(tarball_by_acc).fillna(NA)
    for column, accessions_by_biosample in sequence_accessions_by_column.items():
        df[column] = df["accession"].map(accessions_by_biosample).fillna(NA)
    df = df[COLUMNS]

    # sanity counts
    only_tax = len(tax_acc - atb_target_acc)
    only_atb = len(atb_target_acc - tax_acc)
    both = len(tax_acc & atb_target_acc)
    log.info(
        "index rows: %d | taxonomy-only: %d | atb-only: %d | both: %d",
        len(df),
        only_tax,
        only_atb,
        both,
    )
    log.info("in_ATB True: %d", int((df["in_ATB"] == "True").sum()))
    log.info(
        "pathogen_biosample=NA (atb-only): %d | pathogen_ATB=NA: %d",
        int((df["pathogen_biosample"] == NA).sum()),
        int((df["pathogen_ATB"] == NA).sum()),
    )
    for col in ("pathogen_biosample", "pathogen_ATB"):
        counts = df[col].value_counts().to_dict()
        log.info("%s counts: %s", col, {k: counts[k] for k in sorted(counts)})

    if out_path.exists():
        suffixes = "".join(out_path.suffixes)
        stem = out_path.name.removesuffix(suffixes)
        backup = out_dir / f"{stem}_old{suffixes}"
        os.replace(out_path, backup)
        log.info("renamed previous index -> %s", backup)
    df.to_csv(out_path, sep="\t", index=False)
    log.info("wrote index -> %s (%d rows)", out_path, len(df))
    log.info("  sha256: %s", sha256_file(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
