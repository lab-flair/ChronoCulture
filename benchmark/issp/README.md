# ISSP reconstruction

Download these official GESIS cumulations and place them under one directory:

- ZA8790_v1-0-0.sav
- ZA8792_v2-0-0.sav
- ZA8793_v1-0-0.sav
- ZA8794_v1-0-0.sav
- ZA8795_v1-0-0.sav
- ZA8797_v1-0-0.sav
- ZA4747_v2-1-0.sav
- ZA5960_v1-0-0.sav

Install `numpy`, `pandas`, and `pyreadstat`. From the repository root, run:

```bash
python benchmark/issp/reconstruct_issp_aggregates.py \
  --raw-dir /path/to/issp \
  --out-dir reproduced/issp
```

The script uses survey `WEIGHT`, keeps original pair IDs and years, and drops pairs if either original endpoint has fewer than 500 valid responses. Missing or nonpositive weights are excluded from weighted distributions and counted in the outputs. The 500 cutoff counts valid answers before weight exclusion.


Outputs:

- `issp_benchmark.jsonl`: dated prompts, weighted gold, and evaluation fields.
- `issp_all_year_gold.csv`: weighted gold and counts for available years, including ineligible years.
- `issp_pairs.csv`: retained pairs.
- `issp_pair_audit.csv`: retention decisions.
- `issp_endpoints.jsonl`: retained gold endpoints without wording.
- `reproduction_report.json`: counts and source information.

Prompts are built from the variable and response labels in the GESIS files. For our experiments, question wording was checked against the official ISSP questionnaires, so generated prompts may differ slightly in wording from those used in the paper. Gold distributions and temporal pairs are unaffected.

Build undated or MCQ prompts with the existing repository builders:

```bash
python benchmark/build_undated_prompts.py \
  --benchmark reproduced/issp/issp_benchmark.jsonl \
  --output-jsonl reproduced/issp/issp_undated.jsonl

python benchmark/build_mcq_prompts.py \
  --benchmark reproduced/issp/issp_benchmark.jsonl \
  --output-jsonl reproduced/issp/issp_mcq.jsonl \
  --review-csv reproduced/issp/issp_mcq_review.csv
```

Use `issp_benchmark.jsonl` with the inference runner and pair-metrics script. The gate reads `issp_pairs.csv` and `issp_all_year_gold.csv` through `--issp-dir reproduced/issp`.

Existing predictions correspond to the original cleaned prompts. Do not describe new generations from SAV-label prompts as the original experiment.
