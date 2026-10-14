# ChronoCulture

ChronoCulture evaluates whether language models capture cultural attitudes as they change over time. This repository contains the cleaned inference and evaluation code used for the paper. Due to the WVS and ISSP licensing terms, no survey data is distributed in this repository. Instead, `benchmark/` provides scripts that rebuild both parts of the benchmark from the original survey files, which users download themselves (see `REPRODUCIBILITY.md`).

## Repository layout

- `benchmark/`: schema, WVS and ISSP reconstruction scripts (`benchmark/wvs/`, `benchmark/issp/`), undated-prompt builder, data-interface notes, and supplementary MCQ prompt builder.
- `inference/`: one verbalized-distribution runner for both OpenAI and OpenAI-compatible vLLM endpoints, plus the MCQ runner.
- `evaluation/gate/compute_knowledge_gate.py`: model-specific undated cultural-knowledge filter.
- `evaluation/compute_pair_metrics.py`: pair-level Wasserstein, magnitude, direction, temporal-alignment, stability, and date-cue metrics. It computes metrics **before** filtering, then applies a supplied gate in a separate `summarize` step.
- `evaluation/analyze_period_accuracy.py`, `analyze_default_period.py`, `analyze_date_cues.py`, `analyze_topics.py`: paper-facing summaries.
- `evaluation/threshold_sensitivity.py`: tighter/default/looser human-shift robustness analysis.
- `configs/model_settings.json`: paper inference settings without machine-specific model paths.


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference


```bash
python inference/run_distribution_inference.py \
  --input-jsonl DATA.jsonl --output-csv predictions.csv \
  --backend vllm --base-url http://localhost:8000/v1 --model SERVED_MODEL_ID \
  --temperature 0 --top-p 1 --max-completion-tokens 120
```

For OpenAI, use `--backend openai --model MODEL_ID` and set `OPENAI_API_KEY`. Runs in the paper use one generation per prompt.

## Evaluation order

1. Build one undated prompt per pair with `benchmark/build_undated_prompts.py`, then run undated inference for the knowledge filter and dated inference for the temporal benchmark.
2. Compute gates with `evaluation/gate/compute_knowledge_gate.py`.
3. Compute all pair metrics with `evaluation/compute_pair_metrics.py compute`.
4. Apply the relevant model gate with `evaluation/compute_pair_metrics.py summarize --gate-pass-column passes`.
5. Generate period-accuracy, default-period, date-cue, topic, and threshold-sensitivity summaries as needed.

See `REPRODUCIBILITY.md` for command templates and the mapping from analyses to scripts.


## License

- **Code**: MIT License, see [`LICENSE`](LICENSE).
- **Pair manifests and prompts**: CC BY-NC-SA 4.0, see [`LICENSE-DATA`](LICENSE-DATA). Survey question wording remains the property of the original survey programmes.

No survey data is distributed in this repository: no respondent-level data and no
response distributions. The WVS and ISSP parts of the benchmark are rebuilt from
files that users obtain from the WVS Association and GESIS. Use of the rebuilt
data is restricted to non-commercial purposes, and users must comply with the
WVS and ISSP terms of use and cite the original surveys.

## Use of AI Assistants

The initial manuscript was drafted by the authors, with AI assistance used for subsequent language revision and refinement. The benchmark design and core implementation were developed by the authors. AI assistants were used for auxiliary coding tasks, including visualization, debugging, and helper scripts. The literature review and verification of references were conducted by the authors. The LLMs evaluated in this study were research subjects, and their use is documented separately in the experimental methodology. All AI-assisted writing and code, as well as the analyses and reported results, were reviewed and verified by the authors.
