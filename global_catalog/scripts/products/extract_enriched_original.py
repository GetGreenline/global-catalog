import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        first = ""
        while True:
            ch = fh.read(1)
            if not ch:
                return []
            if not ch.isspace():
                first = ch
                break
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
            if data is None:
                return []
            return data if isinstance(data, list) else [data]

        records: List[Dict[str, Any]] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


def _clean_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, str):
        text = val.strip()
        if text == "" or text.lower() in {"null", "none", "nan"}:
            return None
        return text
    return val


def _normalize_id(val: Any) -> Any:
    val = _clean_value(val)
    if val is None:
        return None
    return str(val).strip()


def _extract_rows(records: List[Dict[str, Any]], source_hint: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    enriched_rows: List[Dict[str, Any]] = []
    original_rows: List[Dict[str, Any]] = []

    for rec in records:
        me = rec.get("matching_enrichment") or {}
        orig = rec.get("original_data") or {}

        source = _clean_value(me.get("source") or orig.get("source") or source_hint)
        if isinstance(source, str):
            source = source.strip().lower()

        product_id = _normalize_id(me.get("product_id") or orig.get("product_id") or rec.get("product_id"))

        enriched_product_name = _clean_value(
            me.get("product_name") or orig.get("product_name") or rec.get("raw_item_name")
        )
        original_product_name = _clean_value(
            orig.get("product_name") or rec.get("raw_item_name") or me.get("product_name")
        )
        brand = _clean_value(me.get("brand") or orig.get("brand_name"))
        package_size = _clean_value(me.get("package_size"))
        variant_weight = _clean_value(me.get("variant_weight"))
        measure = _clean_value(me.get("measure"))
        uom = _clean_value(me.get("uom"))
        if source == "weedmaps" and measure is None:
            measure = variant_weight

        enriched_rows.append(
            {
                "product_id": product_id,
                "source": source,
                "brand": brand,
                "product_name": enriched_product_name,
                "original_product_name": original_product_name,
                "package_size": package_size,
                "variant_weight": variant_weight,
                "measure": measure,
                "uom": uom,
            }
        )

        original_rows.append(
            {
                "product_id": product_id,
                "source": source,
                "brand_name": _clean_value(orig.get("brand_name") or me.get("brand")),
                "product_name": _clean_value(
                    orig.get("product_name") or rec.get("raw_item_name") or me.get("product_name")
                ),
                "measure": _clean_value(orig.get("measure")),
                "uom": _clean_value(orig.get("uom")),
                "product_variant_weight_converted_mg": _clean_value(
                    orig.get("product_variant_weight_converted_mg")
                ),
            }
        )

    return enriched_rows, original_rows


def _write_csv(rows: List[Dict[str, Any]], columns: List[str], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.reindex(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract original vs enriched rows from combined JSON snapshots.")
    parser.add_argument("--weedmaps", help="Path to weedmaps combined JSON/JSONL.")
    parser.add_argument("--hoodie", help="Path to hoodie combined JSON/JSONL.")
    parser.add_argument("--out-dir", default="artifacts/products/extracts", help="Output directory for CSVs.")
    parser.add_argument("--enriched-out", default=None, help="Override enriched CSV output path.")
    parser.add_argument("--original-out", default=None, help="Override original CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weedmaps and not args.hoodie:
        raise SystemExit("Provide at least one input file via --weedmaps or --hoodie.")

    enriched_rows: List[Dict[str, Any]] = []
    original_rows: List[Dict[str, Any]] = []

    if args.weedmaps:
        records = _read_json_records(Path(args.weedmaps))
        e_rows, o_rows = _extract_rows(records, "weedmaps")
        enriched_rows.extend(e_rows)
        original_rows.extend(o_rows)

    if args.hoodie:
        records = _read_json_records(Path(args.hoodie))
        e_rows, o_rows = _extract_rows(records, "hoodie")
        enriched_rows.extend(e_rows)
        original_rows.extend(o_rows)

    out_dir = Path(args.out_dir)
    enriched_path = Path(args.enriched_out) if args.enriched_out else out_dir / "enriched_products.csv"
    original_path = Path(args.original_out) if args.original_out else out_dir / "original_products.csv"

    enriched_cols = [
        "product_id",
        "source",
        "brand",
        "product_name",
        "original_product_name",
        "package_size",
        "variant_weight",
        "measure",
        "uom",
    ]
    original_cols = [
        "product_id",
        "source",
        "brand_name",
        "product_name",
        "measure",
        "uom",
        "product_variant_weight_converted_mg",
    ]

    enriched_df = _write_csv(enriched_rows, enriched_cols, enriched_path)
    original_df = _write_csv(original_rows, original_cols, original_path)

    print(f"Wrote enriched rows: {len(enriched_df)} -> {enriched_path}")
    print(f"Wrote original rows: {len(original_df)} -> {original_path}")


if __name__ == "__main__":
    main()
