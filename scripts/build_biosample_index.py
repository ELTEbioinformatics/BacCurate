"""Assemble the BioSample index and its build log.

Builds the index from:
  - accessions in data/raw/id_lists/<taxon_key>.tsv (created by
    parse_biosample_xml.py). Supplies taxon_biosample and organism_value.
  - accessions in the ATB metadata. Supplies sylph_species and osf_tarball_filename.
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
from baccurate.paths import (
    DEFAULT_BIOSAMPLE_SNAPSHOT_MANIFEST,
    DEFAULT_GENBANK_ASSEMBLY_SNAPSHOT_MANIFEST,
    DEFAULT_INDEX_TSV,
    DEFAULT_REFSEQ_ASSEMBLY_SNAPSHOT_MANIFEST,
    DEFAULT_SRA_SNAPSHOT_MANIFEST,
    RAW_DIR,
)
from baccurate.provenance.source_snapshot import SourceSnapshotManifest, sha256_file
from baccurate.taxon_registry.species_label_matching import NA

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
    "taxon_biosample",
    "sylph_species",
    "taxid",
    "organism_value",
    "osf_tarball_filename",
    "sra_run_accessions",
    "genbank_assembly_accessions",
    "refseq_assembly_accessions",
]


def load_prepared_taxon_keys(manifest_path: Path) -> list[str]:
    """Return the taxon keys of the prepared id_lists, in manifest order."""
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
    if "taxon_key" not in manifest.columns:
        raise ValueError(f"{manifest_path}: expected a taxon_key column")
    if "status" in manifest.columns:
        incomplete = manifest.loc[manifest["status"] != "ok", "taxon_key"].tolist()
        if incomplete:
            log.warning(
                "Manifest flags incomplete taxon-key fetch(es). re-fetch: %s",
                ", ".join(incomplete),
            )
    return manifest["taxon_key"].tolist()


def load_taxonomy_branch(
    id_lists_dir: Path,
    taxon_keys: list[str],
) -> pd.DataFrame:
    """Load the prepared taxon-key TSVs. Resolve overlaps in manifest order.

    The function skips missing files.
    """
    frames = []
    for taxon_key in taxon_keys:
        f = id_lists_dir / f"{taxon_key}.tsv"
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        if d.empty:
            continue
        d = d.rename(columns={"organism": "organism_value"})
        d["taxon_biosample"] = taxon_key
        frames.append(d[["accession", "taxid", "taxon_biosample", "organism_value"]])

    if not frames:
        return pd.DataFrame(columns=["accession", "taxid", "taxon_biosample", "organism_value"])

    tax = pd.concat(frames, ignore_index=True)
    conflicts = tax[tax.duplicated("accession", keep=False)]
    if not conflicts.empty:
        n = conflicts["accession"].nunique()
        log.warning(
            "%d accession(s) matched more than one taxon-key query. "
            "Keeping first by manifest order",
            n,
        )
    tax = tax.drop_duplicates("accession", keep="first")
    log.info("NCBI taxonomy: %d accessions across %d taxon-key files", len(tax), len(frames))
    return tax


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
        "--id-lists-dir", type=Path, default=ID_LISTS_DIR, help="directory of per-taxon-key TSVs"
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

    taxon_keys = load_prepared_taxon_keys(args.id_lists_dir / "manifest.tsv")

    # taxonomy branch
    tax = load_taxonomy_branch(args.id_lists_dir, taxon_keys)
    tax_acc = set(tax["accession"])

    # ATB branch
    atb = pd.read_csv(ATB_METADATA, sep="\t", dtype=str, keep_default_na=False)
    tarball_by_acc = dict(zip(atb["accession"], atb["osf_tarball_filename"]))
    sylph_by_acc = dict(zip(atb["accession"], atb["sylph_species"]))
    atb_acc = set(atb["accession"])
    log.info("ATB records: %d | distinct ATB accessions: %d", len(atb), len(atb_acc))

    # union
    scope = sorted(tax_acc | atb_acc)
    if not scope:
        log.error("No taxonomy id_lists and no ATB accessions found")
        return 1

    org_by_acc = dict(zip(tax["accession"], tax["organism_value"]))
    bio_by_acc = dict(zip(tax["accession"], tax["taxon_biosample"]))
    taxid_by_acc = dict(zip(tax["accession"], tax["taxid"]))

    df = pd.DataFrame({"accession": scope})
    df["taxon_biosample"] = df["accession"].map(bio_by_acc).fillna(NA)
    # Raw GTDB label, kept unchanged so that a reader can recover the AllTheBacteria call itself.
    df["sylph_species"] = df["accession"].map(sylph_by_acc).fillna(NA)
    df["taxid"] = (
        df["accession"].map(taxid_by_acc).fillna(NA).replace("", NA)
    )  # NA for ATB-only records
    df["organism_value"] = df["accession"].map(org_by_acc).fillna(NA)
    df["osf_tarball_filename"] = df["accession"].map(tarball_by_acc).fillna(NA)
    for column, accessions_by_biosample in sequence_accessions_by_column.items():
        df[column] = df["accession"].map(accessions_by_biosample).fillna(NA)
    df = df[COLUMNS]

    # sanity counts
    only_tax = len(tax_acc - atb_acc)
    only_atb = len(atb_acc - tax_acc)
    both = len(tax_acc & atb_acc)
    log.info(
        "index rows: %d | taxonomy-only: %d | atb-only: %d | both: %d",
        len(df),
        only_tax,
        only_atb,
        both,
    )
    log.info(
        "sylph_species present: %d | taxon_biosample=NA (atb-only): %d",
        int((df["sylph_species"] != NA).sum()),
        int((df["taxon_biosample"] == NA).sum()),
    )
    counts = df["taxon_biosample"].value_counts().to_dict()
    log.info("taxon_biosample counts: %s", {k: counts[k] for k in sorted(counts)})

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
