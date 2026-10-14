#!/usr/bin/env python3
"""Compute the model-specific cultural plausibility gate."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

KEEP_FRACTION = 0.75
MIN_RESPONDENTS = 500


def normalized_w1(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(p) != len(q) or len(p) < 2:
        raise ValueError("Distributions must have the same length and at least two options")
    p = p / p.sum()
    q = q / q.sum()
    return float(np.abs(np.cumsum(p - q)[:-1]).mean())


def parse_json_distribution(value, keys):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None

    row = {str(k): v for k, v in value.items()}
    values = []
    for key in keys:
        if key not in row:
            return None
        try:
            values.append(float(row[key]))
        except (TypeError, ValueError):
            return None

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        return None
    return values / values.sum()



def canonical_option(value):
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text

def option_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def load_wvs_benchmark(path):
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "wvs":
                continue
            key = (str(row["variable"]), str(row["country_code_iso3"]).upper())
            items[key] = {
                "pair_id": str(row["pair_id"]),
                "source_pair_id": str(row["pair_id"]),
                "shift_category": str(row.get("gold_shift_category", "")),
            }
    if not items:
        raise ValueError("No WVS records found in benchmark")
    return items


def load_issp_pairs(path):
    df = pd.read_csv(path)
    required = {"pair_id", "variable", "country_code_iso3", "gold_shift_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ISSP pairs missing columns: {sorted(missing)}")

    items = {}
    for row in df.to_dict("records"):
        key = (str(row["variable"]), str(row["country_code_iso3"]).upper())
        if key in items:
            raise ValueError(f"Duplicate ISSP item: {key}")
        items[key] = {
            "pair_id": str(row["pair_id"]),
            "source_pair_id": str(row.get("source_pair_id", row["pair_id"])),
            "shift_category": str(row.get("gold_shift_category", "")),
        }
    if not items:
        raise ValueError("No ISSP pairs found")
    return items


def load_wvs_gold(path):
    df = pd.read_csv(path)
    country_col = "country_code_iso3" if "country_code_iso3" in df.columns else "country_alpha"
    time_col = "wave" if "wave" in df.columns else "year" if "year" in df.columns else None
    required = {"variable", country_col, "response_value", "weighted_pct"}
    missing = required - set(df.columns)
    if missing or time_col is None:
        if time_col is None:
            missing.add("wave/year")
        raise ValueError(f"WVS gold missing columns: {sorted(missing)}")

    gold = defaultdict(dict)
    for (variable, country, period), group in df.groupby(["variable", country_col, time_col]):
        group = group.copy()
        group["_option"] = group["response_value"].map(canonical_option)
        group = group.sort_values("_option", key=lambda s: s.map(option_sort_key))
        values = pd.to_numeric(group["weighted_pct"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            continue
        keys = group["_option"].tolist()
        gold[(str(variable), str(country).upper())][int(period)] = (keys, values / values.sum())
    return gold


def load_issp_gold(path):
    df = pd.read_csv(path)
    required = {
        "variable",
        "country_code_iso3",
        "year",
        "n_valid",
        "gold_distribution_json",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ISSP all-year gold missing columns: {sorted(missing)}")

    df["n_valid"] = pd.to_numeric(df["n_valid"], errors="coerce")
    df = df[df["n_valid"] >= MIN_RESPONDENTS].copy()

    gold = defaultdict(dict)
    for row in df.to_dict("records"):
        dist = json.loads(row["gold_distribution_json"])
        dist = {canonical_option(k): v for k, v in dist.items()}
        keys = sorted(dist, key=option_sort_key)
        values = np.asarray([float(dist[k]) for k in keys], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            continue
        key = (str(row["variable"]), str(row["country_code_iso3"]).upper())
        year = int(row["year"])
        if year in gold[key]:
            raise ValueError(f"Duplicate ISSP distribution for {key}, {year}")
        gold[key][year] = (keys, values / values.sum())
    return gold


def row_key(row, dataset):
    pair_id = str(row.get("pair_id", ""))
    if pair_id.startswith(dataset + "__"):
        parts = pair_id.split("__")
        if len(parts) >= 3:
            return parts[1], parts[2].upper()

    variable = str(row.get("variable", "")).strip()
    country = str(row.get("country_code_iso3", row.get("country_alpha", ""))).strip().upper()
    if variable and country:
        return variable, country
    return None


def load_predictions(path, dataset):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "predicted_distribution_json" not in df.columns:
        raise ValueError(f"{path}: missing predicted_distribution_json")

    predictions = {}
    for row in df.to_dict("records"):
        row_dataset = str(row.get("dataset", "")).strip().lower()
        if row_dataset and row_dataset != dataset:
            continue

        status = str(row.get("status", "")).strip().lower()
        if status and status not in {"ok", "ok_with_parse_failures", "success"}:
            continue

        key = row_key(row, dataset)
        if key is None:
            continue
        if key in predictions:
            raise ValueError(f"{path}: duplicate no-time prediction for {key}")
        predictions[key] = row["predicted_distribution_json"]
    return predictions


def apply_retention(df, keep_fraction=KEEP_FRACTION):
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    out = df.copy()
    out["pct_rank"] = out.groupby("n_waves")["k_hist"].rank(
        pct=True, method="average", ascending=True
    )
    out["passes"] = out["pct_rank"] <= keep_fraction
    out["keep_fraction"] = keep_fraction
    return out


def score_dataset(gold, predictions, items, dataset, keep_fraction):
    rows = []
    skipped = defaultdict(int)

    for key, meta in items.items():
        years = gold.get(key)
        if not years:
            skipped["no_gold"] += 1
            continue
        if key not in predictions:
            skipped["missing_prediction"] += 1
            continue

        ordered_years = sorted(years)
        option_keys = years[ordered_years[0]][0]
        pred = parse_json_distribution(predictions[key], option_keys)
        if pred is None:
            skipped["unparseable_prediction"] += 1
            continue

        distances = []
        for period in ordered_years:
            keys, distribution = years[period]
            if keys != option_keys:
                skipped["inconsistent_options"] += 1
                distances = []
                break
            distances.append((normalized_w1(pred, distribution), period))
        if not distances:
            continue

        k_hist, t_star = min(distances, key=lambda x: (x[0], x[1]))
        if t_star == ordered_years[0]:
            position = "oldest"
        elif t_star == ordered_years[-1]:
            position = "newest"
        else:
            position = "intermediate"

        rows.append({
            "dataset": dataset,
            "variable": key[0],
            "country_code_iso3": key[1],
            "source_pair_id": meta["source_pair_id"],
            "pair_id": meta["pair_id"],
            "shift_category": meta["shift_category"],
            "k_hist": k_hist,
            "t_star": t_star,
            "t_star_position": position,
            "available_years_json": json.dumps(ordered_years),
            "n_waves": len(ordered_years),
            "n_options": len(option_keys),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = apply_retention(result, keep_fraction)
        result = result.sort_values(["n_waves", "k_hist", "variable", "country_code_iso3"]).reset_index(drop=True)
    return result, dict(skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wvs-benchmark", required=True)
    parser.add_argument("--wvs-gold", required=True)
    parser.add_argument("--issp-dir", required=True, help="Output directory from reconstruct_issp_aggregates.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--keep-frac", type=float, default=KEEP_FRACTION)
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "WVS_NOTIME", "ISSP_NOTIME"),
        required=True,
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wvs_items = load_wvs_benchmark(args.wvs_benchmark)
    issp_dir = Path(args.issp_dir)
    issp_items = load_issp_pairs(issp_dir / "issp_pairs.csv")
    wvs_gold = load_wvs_gold(args.wvs_gold)
    issp_gold = load_issp_gold(issp_dir / "issp_all_year_gold.csv")

    summaries = []
    for name, wvs_path, issp_path in args.model:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")

        wvs, wvs_skipped = score_dataset(
            wvs_gold,
            load_predictions(wvs_path, "wvs"),
            wvs_items,
            "wvs",
            args.keep_frac,
        )
        issp, issp_skipped = score_dataset(
            issp_gold,
            load_predictions(issp_path, "issp"),
            issp_items,
            "issp",
            args.keep_frac,
        )

        wvs.insert(0, "model", name)
        issp.insert(0, "model", name)
        combined = pd.concat([wvs, issp], ignore_index=True, sort=False)

        wvs.to_csv(out_dir / f"gate_results_{slug}_wvs.csv", index=False)
        issp.to_csv(out_dir / f"gate_results_{slug}_issp.csv", index=False)
        combined.to_csv(out_dir / f"gate_results_{slug}_combined.csv", index=False)

        for dataset, frame, skipped in (
            ("wvs", wvs, wvs_skipped),
            ("issp", issp, issp_skipped),
        ):
            summaries.append({
                "model": name,
                "dataset": dataset,
                "scored": len(frame),
                "passing": int(frame["passes"].sum()) if len(frame) else 0,
                "retention_pct": 100 * float(frame["passes"].mean()) if len(frame) else np.nan,
                "median_k_hist": float(frame["k_hist"].median()) if len(frame) else np.nan,
                "keep_fraction": args.keep_frac,
                "skipped": json.dumps(skipped, sort_keys=True),
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "gate_summary.csv", index=False)
    (out_dir / "gate_manifest.json").write_text(
        json.dumps({
            "keep_fraction": args.keep_frac,
            "issp_min_respondents": MIN_RESPONDENTS,
            "grouping": "dataset and number of available survey years",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
