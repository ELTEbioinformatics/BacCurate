from collections import Counter
from pathlib import Path


def write_tsv(
    output_dir: str | Path,
    filename: str,
    data: dict[str, dict[str, list[str]]],
    headers: list[str],
) -> None:
    """Write a per-accession TSV with optional '||'-joined list columns."""
    output_dir = Path(output_dir)
    out_path = output_dir / filename
    key_map = {"category": "categories", "attribute_name": "attribute_names", "value": "values"}

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for accession, record in data.items():
            row_parts = [accession]
            for col in headers[1:]:
                dict_key = key_map.get(col, col)
                val = record.get(dict_key, "")

                if isinstance(val, list):
                    row_parts.append("||".join(map(str, val)))
                else:
                    row_parts.append(str(val))

            f.write("\t".join(row_parts) + "\n")


def count_extracted_results(results: list[dict], output_dir: str | Path) -> None:
    """Count attribute and value frequencies and emit counts_attributes.tsv / counts_values.tsv."""
    output_dir = Path(output_dir)
    stats = {"attr_values": Counter(), "attr_types": Counter(), "attribute_distribution": Counter()}
    for record in results:
        for attr in record["attributes"]:
            stats["attr_values"][attr["value"]] += 1
            if attr["attribute_name"]:
                stats["attr_types"][attr["attribute_name"]] += 1
                stats["attribute_distribution"][attr["attribute_name"]] += 1

    attr_names_path = output_dir / "counts_attributes.tsv"
    with open(attr_names_path, "w", encoding="utf-8") as f:
        f.write("attribute\tcount\n")
        for name, count in stats["attribute_distribution"].most_common():
            f.write(f"{name}\t{count}\n")

    values_path = output_dir / "counts_values.tsv"
    with open(values_path, "w", encoding="utf-8") as f:
        f.write("value\tcount\n")
        for value, count in stats["attr_values"].most_common():
            f.write(f"{value}\t{count}\n")


def html_report(
    attr_data: dict,
    output_path: str | Path,
    title: str,
) -> None:
    """Write an HTML report showing the distribution of values for each attribute.

    attr_data: mapping of attribute name -> Counter of value -> count.
    output_path: where to write the HTML file (caller chooses the location).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #333; }",
        "  input[type='text'] { width: 100%; padding: 12px; margin-bottom: 20px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }",
        "  details { margin-bottom: 10px; border: 1px solid #ddd; border-radius: 6px; padding: 10px; background-color: #fafafa; }",
        "  summary { font-weight: bold; cursor: pointer; padding: 5px; outline: none; font-size: 15px; }",
        "  summary:hover { color: #0056b3; }",
        "  .attr-section { margin-left: 20px; margin-top: 15px; margin-bottom: 15px; }",
        "  .attr-title { color: #444; margin-bottom: 5px; }",
        "  .table-container { max-height: 250px; overflow-y: auto; display: inline-block; border: 1px solid #eee; border-radius: 4px; background: #fff; width: 100%; max-width: 600px; }",
        "  table { border-collapse: collapse; width: 100%; }",
        "  th, td { border-bottom: 1px solid #eee; padding: 6px 12px; text-align: left; font-size: 14px; }",
        "  th { background-color: #f4f4f4; position: sticky; top: 0; box-shadow: 0 2px 2px -1px rgba(0,0,0,0.1); }",
        "  .count-cell { width: 80px; text-align: right; font-family: monospace; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h2>{title}</h2>",
        "<input type='text' id='searchInput' placeholder='Search' onkeyup='filterPackages()'>",
        "<div id='package-list'>",
    ]

    sorted_attrs = sorted(attr_data.items(), key=lambda x: sum(x[1].values()), reverse=True)

    for attr_name, counts in sorted_attrs:
        total = sum(counts.values())

        html_content.append("<details class='package-item' open>")
        html_content.append(f"<summary>Attribute: {attr_name} (Total Count: {total})</summary>")

        html_content.append("<div class='attr-section'>")
        html_content.append("<div class='table-container'><table>")
        html_content.append("<tr><th>Value</th><th class='count-cell'>Count</th></tr>")

        for val, count in counts.most_common():
            val_safe = val.replace("<", "&lt;").replace(">", "&gt;")
            html_content.append(f"<tr><td>{val_safe}</td><td class='count-cell'>{count}</td></tr>")

        html_content.append("</table></div></div>")
        html_content.append("</details>")

    html_content.append("</div>")

    js_code = """
    <script>
    function filterPackages() {
        var input = document.getElementById('searchInput').value.toLowerCase();
        var details = document.getElementsByClassName('package-item');

        for (var i = 0; i < details.length; i++) {
            var text = details[i].innerText.toLowerCase();
            if (text.includes(input)) {
                details[i].style.display = "";
            } else {
                details[i].style.display = "none";
            }
        }
    }
    </script>
    </body>
    </html>
    """
    html_content.append(js_code)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))
