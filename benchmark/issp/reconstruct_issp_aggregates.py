#!/usr/bin/env python3
"""Reconstruct the ISSP benchmark from the official cumulation files."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

MIN_RESPONDENTS = 500

SOURCES = {
    "Environment": {
        "study": "ZA8793",
        "file": "ZA8793_v1-0-0.sav",
        "sha256": "1b4ef1d1b0785d56d9ade7e078cde7589837fc369c303ac940a2188860b660d7",
    },
    "Family": {
        "study": "ZA8795",
        "file": "ZA8795_v1-0-0.sav",
        "sha256": "7adb60c290518df662db0c01481ac1d262f43e53348422938626f4e50bca14c1",
    },
    "Health": {
        "study": "ZA8794",
        "file": "ZA8794_v1-0-0.sav",
        "sha256": "30c2e21fb40a9f229c36968802f743e87d8c25f4e4b61ceca6b4a21fcc65c774",
    },
    "National Identity": {
        "study": "ZA5960",
        "file": "ZA5960_v1-0-0.sav",
        "sha256": "4a1435428d07de90cc714f9a3d963d9b226a2529a74fa29aaf87f76c394c183f",
    },
    "Religion": {
        "study": "ZA8792",
        "file": "ZA8792_v2-0-0.sav",
        "sha256": "3df5bc9cec3b11368e3a6a438ac97192746b31c39fd6fae4b937cceed419ef74",
    },
    "Role of Government": {
        "study": "ZA4747",
        "file": "ZA4747_v2-1-0.sav",
        "sha256": "b8d3a296bb658f1cc3de886b0bb82940f3d83e8513860053e0b7c9fcd46ee5f2",
    },
    "Social Inequality": {
        "study": "ZA8790",
        "file": "ZA8790_v1-0-0.sav",
        "sha256": "a74d91ecf9697ef3bc5d5a6084af9dfe130a53d18c604879a87b01abb2eba125",
    },
    "Work Orientations": {
        "study": "ZA8797",
        "file": "ZA8797_v1-0-0.sav",
        "sha256": "af92665262b858e21b3d69b9edab9f06235dd803501a90016dfbd9c12914ebb6",
    },
}

YEAR_COLUMNS = ("year", "survey", "year_sdno", "isspyear", "ispyear", "wave", "module")
COUNTRY_COLUMNS = ("country", "cntry", "nation", "v3")


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_file(root, name):
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} under {root}; found {len(matches)}")
    return matches[0]


def find_column(columns, candidates, label):
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    raise KeyError(f"Could not find {label} column")


def load_manifest(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def weighted_distribution(values, weights, options):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    numeric_options = [float(x) for x in options]

    valid = values.isin(numeric_options)
    n_valid = int(valid.sum())
    usable = valid & np.isfinite(weights) & (weights > 0)
    n_weight_excluded = int((valid & ~usable).sum())

    if not usable.any():
        return None, n_valid, 0.0, n_weight_excluded

    mass = [float(weights[usable & (values == x)].sum()) for x in numeric_options]
    total = float(sum(mass))
    distribution = {str(k): m / total for k, m in zip(options, mass)}
    return distribution, n_valid, total, n_weight_excluded


def normalized_w1(a, b, options):
    x = np.array([float(v) for v in options], dtype=float)
    p = np.array([a[str(v)] for v in options], dtype=float)
    q = np.array([b[str(v)] for v in options], dtype=float)
    return float(np.sum(np.abs(np.cumsum(p - q)[:-1]) * np.diff(x)) / (x[-1] - x[0]))


def shift_category(value):
    if value <= 0.03:
        return "stable"
    if value < 0.10:
        return "moderate"
    if value < 0.20:
        return "strong"
    return "very_strong"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("issp_release_manifest.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_topic = defaultdict(list)
    for item in items:
        by_topic[item["topic"]].append(item)

    gold = {}
    wording = {}
    years_by_item = defaultdict(list)
    all_year_rows = []
    source_rows = []

    for topic, topic_items in by_topic.items():
        source = SOURCES[topic]
        path = find_file(args.raw_dir, source["file"])
        digest = file_sha256(path)
        if digest != source["sha256"] and not args.skip_hash_check:
            raise ValueError(f"SHA-256 mismatch for {source['file']}")

        _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
        year_col = find_column(meta.column_names, YEAR_COLUMNS, "year")
        country_col = find_column(meta.column_names, COUNTRY_COLUMNS, "country")
        weight_col = find_column(meta.column_names, ("weight",), "weight")

        names = {c.lower(): c for c in meta.column_names}
        raw_vars = sorted({item["raw_variable"].lower() for item in topic_items})
        missing = [v for v in raw_vars if v not in names]
        if missing:
            raise KeyError(f"Missing variables in {source['file']}: {missing[:10]}")

        for item in topic_items:
            raw_name = names[item["raw_variable"].lower()]
            label = meta.column_names_to_labels.get(raw_name)
            labels = meta.variable_value_labels.get(raw_name, {})
            if not label:
                raise ValueError(f"Missing question label: {topic}/{raw_name}")
            options = item["valid_options"]
            response_options = {}
            for code in options:
                response_label = labels.get(float(code))
                if not response_label:
                    raise ValueError(f"Missing response label: {topic}/{raw_name}/{code}")
                response_options[str(code)] = str(response_label)
            wording[item["variable"]] = (str(label), response_options)

        usecols = [year_col, country_col, weight_col] + [names[v] for v in raw_vars]
        df, _ = pyreadstat.read_sav(str(path), usecols=usecols, apply_value_formats=False)
        df.columns = df.columns.str.lower()
        year_col = year_col.lower()
        country_col = country_col.lower()
        weight_col = weight_col.lower()

        source_rows.append({
            "topic": topic,
            "study": source["study"],
            "file": source["file"],
            "sha256": digest,
            "rows": len(df),
        })

        items_by_country = defaultdict(list)
        for item in topic_items:
            items_by_country[item["raw_country"]].append(item)

        for raw_country, country_items in items_by_country.items():
            country_df = df[df[country_col] == raw_country]
            if country_df.empty:
                continue

            for year, year_df in country_df.groupby(year_col):
                if pd.isna(year) or not float(year).is_integer():
                    continue
                year = int(year)

                for item in country_items:
                    raw_var = item["raw_variable"].lower()
                    dist, n_valid, weight_sum, n_weight_excluded = weighted_distribution(
                        year_df[raw_var], year_df[weight_col], item["valid_options"]
                    )
                    if dist is None:
                        continue

                    key = (item["variable"], item["country_code_iso3"], year)
                    gold[key] = {
                        "distribution": dist,
                        "n_valid": n_valid,
                        "weight_sum": weight_sum,
                        "n_weight_excluded": n_weight_excluded,
                    }
                    years_by_item[(item["variable"], item["country_code_iso3"])].append(year)
                    all_year_rows.append({
                        "variable": item["variable"],
                        "topic": topic,
                        "source_study": item["source_study"],
                        "raw_variable": item["raw_variable"],
                        "country_code_iso3": item["country_code_iso3"],
                        "country_name": item["country_name"],
                        "year": year,
                        "n_valid": n_valid,
                        "weight_sum": weight_sum,
                        "n_weight_excluded": n_weight_excluded,
                        "eligible": n_valid >= MIN_RESPONDENTS,
                        "gold_distribution_json": json.dumps(dist, separators=(",", ":")),
                    })

    pair_rows = []
    audit_rows = []
    endpoint_rows = []
    benchmark_rows = []

    for item in items:
        key = (item["variable"], item["country_code_iso3"])
        eligible_years = sorted({
            year
            for year in years_by_item[key]
            if gold[(item["variable"], item["country_code_iso3"], year)]["n_valid"] >= MIN_RESPONDENTS
        })

        older_year = int(item["original_older_year"])
        current_year = int(item["original_current_year"])
        older_key = (item["variable"], item["country_code_iso3"], older_year)
        current_key = (item["variable"], item["country_code_iso3"], current_year)
        if older_key not in gold or current_key not in gold:
            raise ValueError(f"Missing original endpoint for {item['source_pair_id']}")
        retained = (
            gold[older_key]["n_valid"] >= MIN_RESPONDENTS
            and gold[current_key]["n_valid"] >= MIN_RESPONDENTS
        )
        pair_id = item["source_pair_id"] if retained else None

        audit_rows.append({
            "source_pair_id": item["source_pair_id"],
            "pair_id": pair_id,
            "variable": item["variable"],
            "topic": item["topic"],
            "source_study": item["source_study"],
            "raw_variable": item["raw_variable"],
            "country_code_iso3": item["country_code_iso3"],
            "country_name": item["country_name"],
            "original_older_year": item["original_older_year"],
            "original_current_year": item["original_current_year"],
            "eligible_years": json.dumps(eligible_years),
            "retained": retained,
            "endpoint_changed": False,
            "reason": "" if retained else "original_endpoint_below_500_valid_responses",
        })

        if not retained:
            continue

        older = gold[(item["variable"], item["country_code_iso3"], older_year)]
        current = gold[(item["variable"], item["country_code_iso3"], current_year)]
        shift = normalized_w1(older["distribution"], current["distribution"], item["valid_options"])
        category = shift_category(shift)

        pair_rows.append({
            "pair_id": pair_id,
            "source_pair_id": item["source_pair_id"],
            "variable": item["variable"],
            "topic": item["topic"],
            "source_study": item["source_study"],
            "raw_variable": item["raw_variable"],
            "country_code_iso3": item["country_code_iso3"],
            "country_name": item["country_name"],
            "older_year": older_year,
            "current_year": current_year,
            "older_n": older["n_valid"],
            "current_n": current["n_valid"],
            "older_gold_distribution_json": json.dumps(older["distribution"], separators=(",", ":")),
            "current_gold_distribution_json": json.dumps(current["distribution"], separators=(",", ":")),
            "gold_shift_w1": shift,
            "gold_shift_category": category,
        })

        for side, year, endpoint in (
            ("older", older_year, older),
            ("current", current_year, current),
        ):
            endpoint_rows.append({
                "dataset": "issp",
                "pair_id": pair_id,
                "source_pair_id": item["source_pair_id"],
                "endpoint": side,
                "variable": item["variable"],
                "topic": item["topic"],
                "source_study": item["source_study"],
                "raw_variable": item["raw_variable"],
                "country_code_iso3": item["country_code_iso3"],
                "country_name": item["country_name"],
                "year": year,
                "n_valid": endpoint["n_valid"],
                "valid_options": item["valid_options"],
                "gold_distribution": endpoint["distribution"],
                "gold_shift_w1": shift,
                "gold_shift_category": category,
            })

    for endpoint in endpoint_rows:
        options = endpoint["valid_options"]
        numeric_options = [float(code) for code in options]
        if any(not value.is_integer() for value in numeric_options):
            raise ValueError("Benchmark requires integer response codes")
        lo, hi = int(min(numeric_options)), int(max(numeric_options))
        if options != [str(code) for code in range(lo, hi + 1)]:
            raise ValueError("Benchmark requires consecutive ordered response codes")
        question, response_options = wording[endpoint["variable"]]
        side = endpoint["endpoint"]
        scale = ", ".join(f"{code} means {response_options[code]!r}" for code in options)
        template = "{" + ", ".join(f'"{code}": <percentage>' for code in options) + "}"
        prompt = (
            f"In {endpoint['country_name']} in {endpoint['year']}, consider this survey item: "
            f"{question}. Use this response scale: {scale}.\n\n"
            "Instead of choosing one answer, estimate the distribution of responses among people in this country and time period.\n"
            f"Return percentages for every valid response option: {', '.join(options)}. "
            "The percentages must be non-negative and should sum to 100.\n"
            "Return only a JSON object in exactly this key format:\n" + template
        )
        record = dict(endpoint)
        record.update(
            prompt_id=endpoint["pair_id"] + "__" + side,
            wave_type=side, time_period=str(endpoint["year"]),
            scale_min=lo, scale_max=hi, scale_points=len(options),
            variable_title=question, question_text=question,
            response_options=response_options, prompt_text=prompt,
            question_source="SAV variable label",
        )
        benchmark_rows.append(record)

    pd.DataFrame(all_year_rows).sort_values(
        ["topic", "variable", "country_code_iso3", "year"]
    ).to_csv(args.out_dir / "issp_all_year_gold.csv", index=False)

    pd.DataFrame(pair_rows).sort_values("pair_id").to_csv(
        args.out_dir / "issp_pairs.csv", index=False
    )

    pd.DataFrame(audit_rows).sort_values("source_pair_id").to_csv(
        args.out_dir / "issp_pair_audit.csv", index=False
    )

    with (args.out_dir / "issp_endpoints.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(endpoint_rows, key=lambda x: (x["pair_id"], x["endpoint"])):
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    with (args.out_dir / "issp_benchmark.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(benchmark_rows, key=lambda x: (x["pair_id"], x["wave_type"])):
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "min_respondents": MIN_RESPONDENTS,
        "benchmark_rows": len(benchmark_rows),
        "prompt_wording": "SAV variable and value labels; not archived cleaned prompts",
        "weighting": "ISSP survey WEIGHT",
        "manifest_items": len(items),
        "retained_pairs": len(pair_rows),
        "excluded_items": sum(not row["retained"] for row in audit_rows),
        "changed_endpoints": sum(bool(row["endpoint_changed"]) for row in audit_rows),
        "endpoint_rows": len(endpoint_rows),
        "weight_exclusions": int(sum(row["n_weight_excluded"] for row in all_year_rows)),
        "sources": source_rows,
    }
    (args.out_dir / "reproduction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
