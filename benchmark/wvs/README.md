# WVS reconstruction

This repository does not distribute any World Values Survey data. To rebuild the
WVS part of ChronoCulture, download the WVS Time Series file yourself and run
the script below.

## 1. Download the WVS Time Series

1. Go to the World Values Survey website (https://www.worldvaluessurvey.org),
   *Data & Documentation → WVS Time-Series (1981–2022)*.
2. Accept the WVS terms of use and download the **CSV** version 5.0:
   `F00011931-WVS_Time_Series_1981-2022_csv_v5_0.csv` (zipped; unzip it).

Use of the data is subject to the WVS terms of use. Please cite the WVS
Time Series data set as requested by the WVS Association.

## 2. Rebuild the benchmark

Install `numpy` and `pandas`. From the repository root, run:

```bash
python benchmark/wvs/reconstruct_wvs_aggregates.py \
  --wvs-csv /path/to/F00011931-WVS_Time_Series_1981-2022_csv_v5_0.csv \
  --out-dir reproduced/wvs
```

The script reads the file in chunks and needs no other input.
`wvs_release_manifest.jsonl` lists the 3,228 temporal pairs (variable,
country, older wave, Wave 7, response options, question wording, and the exact
prompts). It contains no survey responses or distributions.

## Construction rules

- Waves 3, 4, 5, and 7 (column `S002VS`); countries from `COUNTRY_ALPHA`.
- Valid responses are positive codes. Negative codes (don't know, no answer,
  not asked, missing) and code 0 are excluded.
- C001 and C002 are recoded from the WVS order (1 agree, 2 disagree,
  3 neither) to an ordinal order (1 agree, 2 neither, 3 disagree).
- Gold distributions use the survey weight `S017`: each option's share of the
  summed weight of valid respondents. Missing or nonpositive weights are
  excluded and counted in the report.
- Each pair compares an older wave with Wave 7 and is kept only if both
  endpoints have at least 500 valid (unweighted) responses.
- Human shift is the normalized Wasserstein-1 distance between the two gold
  distributions: stable ≤ 0.03 < moderate < 0.10 ≤ strong < 0.20 ≤ very strong.

## Outputs

- `wvs_benchmark.jsonl`: dated prompts with gold distributions and shift
  categories, in the same format as the ISSP benchmark file.
- `wvs_all_year_gold.csv`: weighted gold for every available wave of each
  question and country (long format). Pass this file to the knowledge filter
  as `--wvs-gold`.
- `wvs_pairs.csv`: retained pairs.
- `wvs_pair_audit.csv`: retention decision for every manifest pair.
- `reproduction_report.json`: counts, file hash, and source information.

To check a rebuild against an existing copy of the benchmark, add
`--reference /path/to/benchmark.jsonl`; the report then lists the largest
difference in gold distributions and any shift-category mismatches.

Build undated or MCQ prompts with the existing repository builders:

```bash
python benchmark/build_undated_prompts.py \
  --benchmark reproduced/wvs/wvs_benchmark.jsonl \
  --output-jsonl reproduced/wvs/wvs_undated.jsonl

python benchmark/build_mcq_prompts.py \
  --benchmark reproduced/wvs/wvs_benchmark.jsonl \
  --output-jsonl reproduced/wvs/wvs_mcq.jsonl \
  --review-csv reproduced/wvs/wvs_mcq_review.csv
```
