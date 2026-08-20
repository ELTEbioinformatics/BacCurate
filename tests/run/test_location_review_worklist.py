"""Verify the review worklist for geographic locations."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from baccurate.run.dataset_builder import DatasetBuilder, DatasetBuildRequest
from baccurate.run.location_review_worklist import (
    COLUMNS,
    LOCATION_REVIEW_WORKLIST_FILENAME,
    LocationReviewWorklist,
)
from baccurate.run.statistics import DatasetBuildStatistics
from baccurate.standardization.location import LocationPolicy, UnresolvedLocationInput
from baccurate.standardization_target.specifications import StandardizationTarget

UNREVIEWED_ALPHA = "unreviewed alpha 4471"
UNREVIEWED_BETA = "unreviewed beta 8890"


def _write_worklist(
    tmp_path: Path,
    observations: Sequence[tuple[str, str, str, str]],
) -> tuple[Path, list[dict[str, str]]]:
    """Observe (accession, pathogen_key, attribute, value) tuples and write the table."""
    worklist = LocationReviewWorklist()
    for accession, pathogen_key, attribute, value in observations:
        worklist.observe(
            (UnresolvedLocationInput(attribute, value),),
            accession=accession,
            pathogen_key=pathogen_key,
        )
    tmp_path.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / LOCATION_REVIEW_WORKLIST_FILENAME
    worklist.write(destination)
    with destination.open(encoding="utf-8", newline="") as stream:
        return destination, list(csv.DictReader(stream, delimiter="\t"))


def test_worklist_row_summarizes_one_normalized_value_across_the_complete_run(
    tmp_path: Path,
) -> None:
    destination, rows = _write_worklist(
        tmp_path,
        [
            ("SAMN1", "ecoli", "geo_loc_name", "Unreviewed Alpha 4471"),
            ("SAMN1", "ecoli", "collection_site", "unreviewed  alpha 4471"),
            ("SAMN2", "abaumannii", "geo_loc_name", "unreviewed alpha 4471"),
        ],
    )

    assert destination.read_bytes().startswith("\t".join(COLUMNS).encode("utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["normalized_submitted_value"] == UNREVIEWED_ALPHA
    assert row["biosample_record_count"] == "2"
    assert row["occurrence_count"] == "3"
    assert json.loads(row["pathogen_counts"]) == {"abaumannii": 1, "ecoli": 2}
    assert json.loads(row["submitted_attribute_counts"]) == {
        "collection_site": 1,
        "geo_loc_name": 2,
    }
    assert sum(json.loads(row["pathogen_counts"]).values()) == 3
    assert sum(json.loads(row["submitted_attribute_counts"]).values()) == 3


def test_worklist_rows_sort_by_record_count_then_occurrences_then_value(
    tmp_path: Path,
) -> None:
    _, rows = _write_worklist(
        tmp_path,
        [
            ("SAMN1", "ecoli", "geo_loc_name", "one record twice"),
            ("SAMN1", "ecoli", "collection_site", "one record twice"),
            ("SAMN2", "ecoli", "geo_loc_name", "two records"),
            ("SAMN3", "ecoli", "geo_loc_name", "two records"),
            ("SAMN4", "ecoli", "geo_loc_name", "one record once"),
        ],
    )

    assert [row["normalized_submitted_value"] for row in rows] == [
        "two records",
        "one record twice",
        "one record once",
    ]


def test_worklist_keeps_three_examples_chosen_independently_of_input_order(
    tmp_path: Path,
) -> None:
    observations = [
        ("SAMN4", "ecoli", "geo_loc_name", UNREVIEWED_ALPHA),
        ("SAMN1", "ecoli", "geo_loc_name", UNREVIEWED_ALPHA),
        ("SAMN3", "ecoli", "geo_loc_name", UNREVIEWED_ALPHA),
        ("SAMN2", "ecoli", "geo_loc_name", UNREVIEWED_ALPHA),
    ]

    _, rows = _write_worklist(tmp_path / "forward", observations)
    _, reversed_rows = _write_worklist(tmp_path / "reversed", list(reversed(observations)))

    assert json.loads(rows[0]["representative_examples"]) == [
        {
            "biosample_accession": accession,
            "pathogen_key": "ecoli",
            "submitted_attribute": "geo_loc_name",
            "submitted_value": UNREVIEWED_ALPHA,
        }
        for accession in ("SAMN1", "SAMN2", "SAMN3")
    ]
    assert rows == reversed_rows


def test_worklist_json_columns_keep_non_ascii_text_unescaped(tmp_path: Path) -> None:
    _, rows = _write_worklist(tmp_path, [("SAMN1", "ecoli", "geo_loc_name", "Tübingen 7734")])

    assert "Tübingen 7734" in rows[0]["representative_examples"]
    assert "\\u" not in rows[0]["representative_examples"]


# =============================================================================
# Worklist production during a run
# =============================================================================


@pytest.fixture
def reviewed_location_policy(
    tmp_path: Path,
    fixture_location_policy: LocationPolicy,
) -> LocationPolicy:
    policy_path = tmp_path / "reviewed-location.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "coordinate_attributes": [],
                "insdc_country_map": {},
                "reviewed_mappings": {"uae": "United Arab Emirates"},
                "reviewed_unmapped": ["ncbs"],
                "geo_loc_list_path": fixture_location_policy.geo_loc_list_path.as_posix(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return LocationPolicy.load(policy_path)


def _located_record(accession: str, submitted_value: str, **overrides: str) -> dict[str, str]:
    return {
        "accession": accession,
        "pathogen": "ecoli",
        "bioproject_accession": "",
        "biosample_last_update": "2020-02-29T00:00:00",
        "date_attr_orig": "collection_date",
        "date_val_orig": "2020-02",
        "date_category": "c",
        "loc_attr_orig": "geo_loc_name",
        "loc_val_orig": submitted_value,
        **overrides,
    }


def _build(
    tmp_path: Path,
    bundle,
    *,
    name: str,
    location_policy: LocationPolicy | None,
    targets: tuple[StandardizationTarget, ...],
    standardization_fixture_resources,
    fixture_pathogen_registry,
) -> tuple[Path, DatasetBuildStatistics]:
    destination = tmp_path / name / "run.tsv"
    statistics = DatasetBuilder().build(
        DatasetBuildRequest(
            extracted_metadata=bundle.extracted_metadata,
            biosample_snapshot_manifest=bundle.biosample_snapshot_manifest,
            bioproject_snapshot_manifest=bundle.bioproject_snapshot_manifest,
            requested_pathogens=("ecoli",),
            requested_targets=targets,
            final_destination=destination,
            pathogen_registry=fixture_pathogen_registry,
            location_policy=location_policy,
            disable_progress=True,
        )
    )
    return destination, statistics


@pytest.fixture
def worklist_bundle(extracted_metadata_bundle_factory):
    return extracted_metadata_bundle_factory(
        "worklist",
        [
            _located_record("SAMN_STANDARDIZED", "Germany"),
            _located_record("SAMN_REVIEWED_MAPPING", "UAE"),
            _located_record("SAMN_REVIEWED_UNMAPPED", "NCBS"),
            _located_record(
                "SAMN_UNRESOLVED",
                f"Germany||{UNREVIEWED_ALPHA}",
                loc_attr_orig="geo_loc_name||collection_site",
            ),
            _located_record("SAMN_UNRESOLVED_ONLY", UNREVIEWED_BETA),
        ],
    )


def test_location_run_writes_the_worklist_and_reports_its_totals(
    tmp_path: Path,
    worklist_bundle,
    reviewed_location_policy: LocationPolicy,
    standardization_fixture_resources,
    fixture_pathogen_registry,
) -> None:
    destination, statistics = _build(
        tmp_path,
        worklist_bundle,
        name="with-worklist",
        location_policy=reviewed_location_policy,
        targets=(StandardizationTarget.LOCATION,),
        standardization_fixture_resources=standardization_fixture_resources,
        fixture_pathogen_registry=fixture_pathogen_registry,
    )

    worklist_path = destination.parent / LOCATION_REVIEW_WORKLIST_FILENAME
    with worklist_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    assert statistics.location is not None
    assert statistics.location.review_worklist is not None
    assert statistics.location.review_worklist.path == worklist_path
    assert statistics.location.review_worklist.row_count == 2
    assert statistics.location.review_worklist.occurrence_count == 2
    assert statistics.location.review_worklist.biosample_record_count == 2
    # Reviewed values stay out; a standardized record still contributes its unresolved value.
    assert [row["normalized_submitted_value"] for row in rows] == [
        UNREVIEWED_ALPHA,
        UNREVIEWED_BETA,
    ]
    assert statistics.location.aggregate.standardized == 3
    assert statistics.location.aggregate.reviewed_mapping_matches == 1


def test_location_run_without_unresolved_inputs_writes_the_header_only_worklist(
    tmp_path: Path,
    extracted_metadata_bundle_factory,
    reviewed_location_policy: LocationPolicy,
    standardization_fixture_resources,
    fixture_pathogen_registry,
) -> None:
    bundle = extracted_metadata_bundle_factory(
        "resolved", [_located_record("SAMN_STANDARDIZED", "Germany")]
    )

    destination, statistics = _build(
        tmp_path,
        bundle,
        name="header-only",
        location_policy=reviewed_location_policy,
        targets=(StandardizationTarget.LOCATION,),
        standardization_fixture_resources=standardization_fixture_resources,
        fixture_pathogen_registry=fixture_pathogen_registry,
    )

    worklist_path = destination.parent / LOCATION_REVIEW_WORKLIST_FILENAME
    assert worklist_path.read_text(encoding="utf-8") == "\t".join(COLUMNS) + "\n"
    assert statistics.location is not None
    assert statistics.location.review_worklist is not None
    assert statistics.location.review_worklist.row_count == 0


def test_run_without_the_location_target_writes_no_worklist(
    tmp_path: Path,
    worklist_bundle,
    standardization_fixture_resources,
    fixture_pathogen_registry,
) -> None:
    destination, statistics = _build(
        tmp_path,
        worklist_bundle,
        name="date-only",
        location_policy=None,
        targets=(StandardizationTarget.DATE,),
        standardization_fixture_resources=standardization_fixture_resources,
        fixture_pathogen_registry=fixture_pathogen_registry,
    )

    assert statistics.location is None
    assert not (destination.parent / LOCATION_REVIEW_WORKLIST_FILENAME).exists()


def test_repeated_location_runs_produce_identical_datasets_and_worklists(
    tmp_path: Path,
    worklist_bundle,
    reviewed_location_policy: LocationPolicy,
    standardization_fixture_resources,
    fixture_pathogen_registry,
) -> None:
    outputs = [
        _build(
            tmp_path,
            worklist_bundle,
            name=name,
            location_policy=reviewed_location_policy,
            targets=(StandardizationTarget.LOCATION,),
            standardization_fixture_resources=standardization_fixture_resources,
            fixture_pathogen_registry=fixture_pathogen_registry,
        )[0]
        for name in ("first", "second")
    ]

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert (outputs[0].parent / LOCATION_REVIEW_WORKLIST_FILENAME).read_bytes() == (
        outputs[1].parent / LOCATION_REVIEW_WORKLIST_FILENAME
    ).read_bytes()
