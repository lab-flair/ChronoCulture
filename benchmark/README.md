# ChronoCulture benchmark

No survey data is distributed in this repository. Both parts of the benchmark
are rebuilt from the original survey files, which users obtain themselves:

| Survey | Script | Instructions |
|---|---|---|
| WVS | `wvs/reconstruct_wvs_aggregates.py` | `wvs/README.md` (WVS Time Series 1981-2022, CSV v5.0) |
| ISSP | `issp/reconstruct_issp_aggregates.py` | `issp/README.md` (eight GESIS cumulation files) |

Each script reads a release manifest (`wvs/wvs_release_manifest.jsonl`,
`issp/issp_release_manifest.jsonl`) that lists the temporal pairs: question,
country, survey years or waves, and response options. The manifests contain no
survey responses or distributions.

## Record format

Each rebuilt benchmark file (`wvs_benchmark.jsonl`, `issp_benchmark.jsonl`) has
one JSON object per dated prompt, two per temporal pair. Fields follow
`schema.json`:

| Field | Meaning |
|---|---|
| `dataset` | `"wvs"` or `"issp"` |
| `pair_id` | Temporal pair identifier |
| `prompt_id` | `{pair_id}__{wave_type}` |
| `variable`, `variable_title` | Survey variable and title |
| `country_code_iso3`, `country_name` | Country |
| `wave_type` | `older` or `current` (newer period of the pair) |
| `time_period` | Survey period referenced in the prompt |
| `scale_min`, `scale_max`, `scale_points`, `valid_options`, `response_options` | Ordered response scale and labels |
| `question_text`, `prompt_text` | Question wording and the dated prompt given to models |
| `gold_distribution` | Survey-weighted share of respondents choosing each option (sums to 1) |
| `gold_shift_w1`, `gold_shift_category` | Human shift between the two periods (normalized Wasserstein) and its category |

Other scripts in this folder build undated prompts (`build_undated_prompts.py`)
and the supplementary MCQ prompts (`build_mcq_prompts.py`) from a rebuilt
benchmark file.

## Terms of use

Use of the rebuilt data is subject to the WVS and ISSP (GESIS) terms of use: it
is limited to non-commercial purposes, and publications must cite the original
surveys as specified by the WVS Association and GESIS, in addition to this
benchmark.
