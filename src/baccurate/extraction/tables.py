"""Per-accession record accumulator and per-pathogen distribution counter."""

from collections import Counter, defaultdict
from collections.abc import Iterator

from baccurate.extraction.classifiers import ATTRIBUTES, AttributeMatch

# accession is the row key, not a stored field, so it leads the column list.
COLUMNS = ["accession", "bioproject", "pathogen", "package", "date_category"] + [
    f"{attribute}_{kind}" for attribute in ATTRIBUTES for kind in ("attr", "val")
]

# Columns holding multiple values, joined with '||' on write.
LIST_COLUMNS = {"date_category"} | {
    f"{attribute}_{kind}" for attribute in ATTRIBUTES for kind in ("attr", "val")
}

REPORT_LABELS = {
    "host": "host",
    "iso": "isolation source",
    "loc": "location",
    "date": "date",
}


def _new_record() -> dict[str, str | list[str]]:
    return {col: [] if col in LIST_COLUMNS else "" for col in COLUMNS if col != "accession"}


class RecordTable:
    """Accumulates classified attributes into one record per BioSample accession."""

    def __init__(self) -> None:
        self._records: defaultdict[str, dict] = defaultdict(_new_record)

    def add(
        self,
        accession: str,
        pathogen: str,
        package: str,
        bioproject: str,
        attr: dict,
        matches: tuple[AttributeMatch, ...],
    ) -> None:
        rec = self._records[accession]
        rec["pathogen"] = pathogen
        rec["package"] = package
        rec["bioproject"] = bioproject
        for match in matches:
            rec[f"{match.attribute}_attr"].append(attr["attribute"] or "")
            rec[f"{match.attribute}_val"].append(attr["value"])
            if match.attribute == "date":
                rec["date_category"].append(match.category)

    def rows(self) -> Iterator[list[str]]:
        for accession in sorted(self._records):
            rec = self._records[accession]
            row = []
            for col in COLUMNS:
                if col == "accession":
                    row.append(accession)
                elif col in LIST_COLUMNS:
                    row.append("||".join(rec[col]))
                else:
                    row.append(rec[col])
            yield row


class DistributionTable:
    """Counts attribute-value occurrences per pathogen and attribute type for the HTML reports."""

    def __init__(self) -> None:
        # Structure: pathogen -> attribute type -> attribute name -> Counter
        self._counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

    def add(self, pathogen: str, attr: dict, matches: tuple[AttributeMatch, ...]) -> None:
        for match in matches:
            self._counts[pathogen][match.attribute][attr["attribute"] or ""][attr["value"]] += 1

    def reports(self) -> Iterator[tuple[str, str, dict]]:
        for pathogen, by_attribute in self._counts.items():
            for attribute, data in by_attribute.items():
                yield pathogen, attribute, data
