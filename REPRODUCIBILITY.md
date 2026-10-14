# Reproducibility

## Setup

```bash
pip install -r requirements.txt
```

## WVS reconstruction

WVS data are not redistributed. Download the WVS Time Series 1981-2022 CSV (v5.0) as described in `benchmark/wvs/README.md` and run:

```bash
python benchmark/wvs/reconstruct_wvs_aggregates.py \
  --wvs-csv /path/to/F00011931-WVS_Time_Series_1981-2022_csv_v5_0.csv \
  --out-dir reproduced/wvs
```

The reconstruction uses the WVS survey weight `S017` and keeps pairs whose older wave and Wave 7 both have at least 500 valid responses. The main outputs used by evaluation are:

- `reproduced/wvs/wvs_benchmark.jsonl`
- `reproduced/wvs/wvs_all_year_gold.csv`
- `reproduced/wvs/wvs_pairs.csv`

## ISSP reconstruction

ISSP respondent-level data are not redistributed. Download the eight cumulation files listed in `benchmark/issp/README.md` from GESIS, place them under one directory, and run:

```bash
python benchmark/issp/reconstruct_issp_aggregates.py \
  --raw-dir /path/to/issp \
  --out-dir reproduced/issp
```

The reconstruction uses the ISSP survey `WEIGHT` variable and keeps question-country-year observations with at least 500 valid responses. 

The main outputs used by evaluation are:

- `reproduced/issp/issp_benchmark.jsonl`
- `reproduced/issp/issp_all_year_gold.csv`
- `reproduced/issp/issp_pairs.csv`
- `reproduced/issp/issp_pair_audit.csv`

## Combined benchmark

The pair-metric scripts read one benchmark file with both surveys:

```bash
cat reproduced/wvs/wvs_benchmark.jsonl reproduced/issp/issp_benchmark.jsonl \
  > reproduced/chronoculture.jsonl
```

## Cultural plausibility gate

The gate compares each undated model prediction with every available survey year for the same question-country item. Items are ranked separately within each survey by the number of available survey years, and the lowest 75% in each group are retained.

```bash
python evaluation/gate/compute_knowledge_gate.py \
  --wvs-benchmark reproduced/wvs/wvs_benchmark.jsonl \
  --wvs-gold reproduced/wvs/wvs_all_year_gold.csv \
  --issp-dir reproduced/issp \
  --out-dir results/gates \
  --model MODEL_LABEL /path/to/wvs_notime.csv /path/to/issp_notime.csv
```

`--issp-dir` is the output directory from the ISSP reconstruction step; no ISSP metadata workbook is required.

## Pair metrics

```bash
python evaluation/compute_pair_metrics.py compute \
  --benchmark reproduced/chronoculture.jsonl \
  --predictions /path/to/dated_predictions.csv \
  --no-time /path/to/notime_predictions.csv \
  --model-name MODEL_LABEL \
  --tau 0.10 \
  --out-dir results/MODEL/metrics

python evaluation/compute_pair_metrics.py summarize \
  --metrics results/MODEL/metrics/pair_metrics_all.csv \
  --gate-csv results/gates/gate_results_MODEL_combined.csv \
  --gate-pair-column pair_id \
  --gate-pass-column passes \
  --gate-model-column model \
  --gate-model-value MODEL_LABEL \
  --out-dir results/MODEL/summary
```

Run the remaining analysis scripts on the retained pair metrics.

