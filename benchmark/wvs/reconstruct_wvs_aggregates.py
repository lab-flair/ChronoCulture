#!/usr/bin/env python3
"""Reconstruct the WVS part of ChronoCulture from the official WVS Time Series file.

No WVS data is distributed with this repository. Users download the WVS Time
Series 1981-2022 file themselves (see benchmark/wvs/README.md) and run this
script, which rebuilds the survey-weighted gold distributions for every pair
listed in wvs_release_manifest.jsonl.

Construction rules (same as used for the paper):
  * wave column S002VS, country column COUNTRY_ALPHA, weight column S017;
  * a response is valid if its code is positive; negative codes (don't know,
    no answer, not asked, missing) and code 0 are excluded;
  * C001 and C002 are recoded from the WVS order (1 agree, 2 disagree,
    3 neither) to an ordinal order (1 agree, 2 neither, 3 disagree);
  * gold distribution = share of the summed S017 weight of valid respondents
    choosing each option (nonpositive or missing weights are excluded and
    counted in the outputs);
  * each pair compares an older wave (3, 4 or 5) with Wave 7, and is kept only
    if both endpoints have at least 500 valid respondents (unweighted);
  * human shift = normalized Wasserstein-1 distance between the two gold
    distributions; categories stable <= 0.03 < moderate < 0.10 <= strong
    < 0.20 <= very strong.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MIN_RESPONDENTS = 500
WAVES = (3, 4, 5, 7)          # waves used to build the benchmark and its all-wave gold
EXPECTED_FILE = "F00011931-WVS_Time_Series_1981-2022_csv_v5_0.csv"
EXPECTED_SHA256 = None        # fill in with the hash printed on first run, if desired
WAVE_COL, COUNTRY_COL, WEIGHT_COL = "S002VS", "COUNTRY_ALPHA", "S017"


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def accumulate(csv_path, variables, countries, chunksize):
    """Return per (variable, country, wave, raw_code) counts and weight sums."""
    header = pd.read_csv(csv_path, nrows=0).columns
    lookup = {c.upper(): c for c in header}
    for col in (WAVE_COL, COUNTRY_COL, WEIGHT_COL):
        if col not in lookup:
            raise KeyError(f"Column {col} not found in {csv_path.name}")
    missing = [v for v in variables if v.upper() not in lookup]
    if missing:
        raise KeyError(f"Variables missing from {csv_path.name}: {missing[:10]}")

    usecols = [lookup[c] for c in (WAVE_COL, COUNTRY_COL, WEIGHT_COL)] + [lookup[v.upper()] for v in variables]
    count = defaultdict(int)            # (var, country, wave, code) -> respondents
    wsum = defaultdict(float)           # (var, country, wave, code) -> summed weight
    wexcl = defaultdict(int)            # (var, country, wave) -> valid answers with unusable weight
    rows_read = 0

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk.columns = [c.upper() for c in chunk.columns]
        rows_read += len(chunk)
        wave = pd.to_numeric(chunk[WAVE_COL], errors="coerce")
        chunk = chunk[wave.isin(WAVES) & chunk[COUNTRY_COL].isin(countries)]
        if chunk.empty:
            continue
        wave = pd.to_numeric(chunk[WAVE_COL], errors="coerce").astype(int)
        country = chunk[COUNTRY_COL].astype(str)
        weight = pd.to_numeric(chunk[WEIGHT_COL], errors="coerce")
        good_w = np.isfinite(weight) & (weight > 0)

        for var in variables:
            vals = pd.to_numeric(chunk[var.upper()], errors="coerce")
            valid = vals > 0
            if not valid.any():
                continue
            frame = pd.DataFrame({
                "country": country[valid], "wave": wave[valid],
                "code": vals[valid].astype(int), "w": weight[valid], "good": good_w[valid],
            })
            for (c, wv, code), g in frame.groupby(["country", "wave", "code"]):
                count[(var, c, wv, code)] += len(g)
                wsum[(var, c, wv, code)] += float(g.loc[g["good"], "w"].sum())
                n_bad = int((~g["good"]).sum())
                if n_bad:
                    wexcl[(var, c, wv)] += n_bad
    return count, wsum, wexcl, rows_read


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wvs-csv", type=Path, required=True,
                        help=f"Path to the unzipped WVS Time Series CSV ({EXPECTED_FILE}).")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("wvs_release_manifest.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--reference", type=Path, default=None,
                        help="Optional existing benchmark JSONL to compare reconstructed gold against.")
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    digest = file_sha256(args.wvs_csv)
    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        raise ValueError(f"SHA-256 mismatch for {args.wvs_csv.name}: {digest}")

    variables = sorted({it["variable"] for it in items})
    countries = sorted({it["country_code_iso3"] for it in items})
    meta = {}
    for it in items:
        meta.setdefault((it["variable"], it["country_code_iso3"]), it)

    count, wsum, wexcl, rows_read = accumulate(args.wvs_csv, variables, set(countries), args.chunksize)

    # ---- gold for every (variable, country, wave) present ------------------------
    gold = {}
    all_rows = []
    codes_by_key = defaultdict(list)
    for (var, c, wv, code), n in count.items():
        codes_by_key[(var, c, wv)].append((code, n, wsum[(var, c, wv, code)]))
    for var, c, wv in sorted(codes_by_key):
        it = meta.get((var, c))
        if it is None:
            continue
        options = it["valid_options"]
        recode = it.get("raw_to_option") or {}
        n_valid = 0
        n_out = 0
        mass = {o: 0.0 for o in options}
        cnt = {o: 0 for o in options}
        for code, n, w in codes_by_key[(var, c, wv)]:
            n_valid += n
            opt = recode.get(str(code), str(code))
            if opt not in mass:
                n_out += n
                continue
            mass[opt] += w
            cnt[opt] += n
        total = sum(mass.values())
        if total <= 0:
            continue
        dist = {o: mass[o] / total for o in options}
        gold[(var, c, wv)] = {"distribution": dist, "n_valid": n_valid,
                              "weight_sum": total, "n_out_of_scale": n_out,
                              "n_weight_excluded": wexcl.get((var, c, wv), 0)}
        for o in options:
            all_rows.append({
                "variable": var, "variable_title": it["variable_title"],
                "country_alpha": c, "country_name": it["country_name"], "wave": wv,
                "response_value": int(o), "count": cnt[o], "weighted_count": mass[o],
                "weighted_pct": 100.0 * dist[o], "total_valid": n_valid,
                "total_weighted_valid": total, "eligible": n_valid >= MIN_RESPONDENTS,
            })

    # ---- pairs ------------------------------------------------------------------
    pair_rows, audit_rows, bench_rows = [], [], []
    for it in items:
        var, c = it["variable"], it["country_code_iso3"]
        ow, cw = int(it["original_older_wave"]), int(it["original_current_wave"])
        older, current = gold.get((var, c, ow)), gold.get((var, c, cw))
        reason = ""
        if older is None or current is None:
            reason = "endpoint_missing_in_source_file"
        elif older["n_valid"] < MIN_RESPONDENTS or current["n_valid"] < MIN_RESPONDENTS:
            reason = "endpoint_below_500_valid_responses"
        audit_rows.append({"source_pair_id": it["source_pair_id"], "variable": var,
                           "country_code_iso3": c, "older_wave": ow, "current_wave": cw,
                           "older_n": older["n_valid"] if older else None,
                           "current_n": current["n_valid"] if current else None,
                           "retained": reason == "", "reason": reason})
        if reason:
            continue
        options = it["valid_options"]
        shift = normalized_w1(older["distribution"], current["distribution"], options)
        cat = shift_category(shift)
        pair_rows.append({"pair_id": it["source_pair_id"], "variable": var, "country_code_iso3": c,
                          "country_name": it["country_name"], "older_wave": ow, "current_wave": cw,
                          "older_n": older["n_valid"], "current_n": current["n_valid"],
                          "gold_shift_w1": shift, "gold_shift_category": cat})
        for side, period, prompt, g in (("older", it["older_period"], it["older_prompt_text"], older),
                                        ("current", it["current_period"], it["current_prompt_text"], current)):
            bench_rows.append({
                "dataset": "wvs", "pair_id": it["source_pair_id"],
                "prompt_id": f"{it['source_pair_id']}__{side}",
                "variable": var, "variable_title": it["variable_title"],
                "country_code_iso3": c, "country_name": it["country_name"],
                "wave_type": side, "time_period": period,
                "scale_min": int(options[0]), "scale_max": int(options[-1]), "scale_points": len(options),
                "question_text": it["question_text"], "valid_options": options,
                "response_options": it["response_options"], "prompt_text": prompt,
                "gold_distribution": g["distribution"],
                "gold_shift_category": cat, "gold_shift_w1": shift,
            })

    # ---- write ------------------------------------------------------------------
    pd.DataFrame(all_rows).sort_values(["variable", "country_alpha", "wave", "response_value"]).to_csv(
        args.out_dir / "wvs_all_year_gold.csv", index=False)
    pd.DataFrame(pair_rows).sort_values("pair_id").to_csv(args.out_dir / "wvs_pairs.csv", index=False)
    pd.DataFrame(audit_rows).sort_values("source_pair_id").to_csv(args.out_dir / "wvs_pair_audit.csv", index=False)
    with (args.out_dir / "wvs_benchmark.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(bench_rows, key=lambda x: (x["pair_id"], x["wave_type"] != "older")):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "source_file": args.wvs_csv.name, "sha256": digest, "rows_read": rows_read,
        "waves": list(WAVES), "min_respondents": MIN_RESPONDENTS, "weighting": "WVS S017",
        "manifest_pairs": len(items), "retained_pairs": len(pair_rows),
        "excluded_pairs": sum(not r["retained"] for r in audit_rows),
        "benchmark_rows": len(bench_rows),
        "shift_categories": pd.Series([r["gold_shift_category"] for r in pair_rows]).value_counts().to_dict(),
        "out_of_scale_responses": int(sum(g["n_out_of_scale"] for g in gold.values())),
        "weight_exclusions": int(sum(g["n_weight_excluded"] for g in gold.values())),
    }

    if args.reference:
        ref = {}
        with args.reference.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("dataset", "wvs") == "wvs":
                        ref[r["prompt_id"]] = r
        diffs, cat_mismatch, missing = [], 0, 0
        for r in bench_rows:
            o = ref.get(r["prompt_id"])
            if o is None:
                missing += 1
                continue
            diffs.append(max(abs(r["gold_distribution"][k] - float(o["gold_distribution"][k]))
                             for k in r["valid_options"]))
            cat_mismatch += r["gold_shift_category"] != o.get("gold_shift_category")
        report["reference_check"] = {
            "compared_rows": len(diffs), "missing_in_reference": missing,
            "max_abs_gold_difference": max(diffs) if diffs else None,
            "rows_differing_by_more_than_1e-6": int(sum(d > 1e-6 for d in diffs)),
            "shift_category_mismatches": int(cat_mismatch),
        }

    (args.out_dir / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
