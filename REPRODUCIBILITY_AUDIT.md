# Reproducibility audit

Date: 2026-07-20 KST

This document records the canonical public reproducibility workflow for CASTS.

## Verification status

| Item                         | Status   | Evidence                                                                 |
| ---------------------------- | -------- | ------------------------------------------------------------------------ |
| Dataset inventory            | verified | 150 included labeled time series across NAB, SKAB, SMD, SMAP, and MSL    |
| Fixed split construction     | verified | `scripts/generate_splits.py`                                           |
| Split evaluation             | verified | `scripts/evaluate_splits.py`                                           |
| Continuity-weight experiment | verified | `scripts/run_lambda_sweep.py`                                          |
| Detector evaluation          | verified | 77 common time series, four splitting methods, and three detector models |
| Table generation             | verified | `scripts/build_tables.py`                                              |
| Release verification         | passed   | `make verify`                                                          |

## Canonical experiment workflow

```bash

python scripts/prepare_datasets.py \
  --data-root <DATA_ROOT>

python scripts/generate_splits.py \
  --data-root <DATA_ROOT> \
  --output splits/fixed_split_intervals.csv.gz \
  --jobs 8

python scripts/evaluate_splits.py \
  --data-root <DATA_ROOT> \
  --assignments splits/fixed_split_intervals.csv.gz

python scripts/run_lambda_sweep.py \
  --config configs/lambda_sweep.yaml \
  --data-root <DATA_ROOT> \
  --jobs 8

python scripts/run_detectors.py \
  --data-root <DATA_ROOT> \
  --splits splits/fixed_split_intervals.csv.gz \
  --dataset-list results/summaries/common_77_datasets.csv \
  --models CNN1D GRU LSTM \
  --methods \
    "Chronological 60/20/20" \
    "Bin-level stratified random" \
    "CASTS-05" \
    "CASTS-35" \
  --window 32 \
  --stride 4 \
  --epochs 5 \
  --batch-size 128 \
  --output results/raw/detector_results_by_run.csv \
  --summary-output results/raw/detector_model_metrics_summary.csv

make verify
```



## Canonical generated artifacts

| Artifact                                                    | Description                                            |
| ----------------------------------------------------------- | ------------------------------------------------------ |
| `results/raw/dataset_inventory.csv`                       | Metadata for the 150 included time series              |
| `splits/fixed_split_intervals.csv.gz`                     | Fixed train, validation, and test interval assignments |
| `results/raw/generated_split_audit_by_split.csv`          | Split-level evaluation records                         |
| `results/summaries/generated_split_method_summary.csv`    | Overall split-level summary                            |
| `splits/lambda_sweep_intervals.csv.gz`                    | Continuity-weight split assignments                    |
| `results/raw/generated_lambda_audit_by_split.csv`         | Continuity-weight evaluation records                   |
| `results/summaries/generated_lambda_tradeoff_summary.csv` | Overall continuity-weight summary                      |
| `results/raw/detector_results_by_run.csv`                 | Detector evaluation records                            |
| `results/raw/detector_model_metrics_summary.csv`          | Dataset--method--model detector summaries              |
| `results/summaries/common_77_datasets.csv`                | Common detector evaluation set                         |
| `results/tables/`                                         | Generated Tables 1--5 and 9--13                        |
| `results/expected/`                                       | Rounded expected values used by verification           |

## Detector evaluation scope

The detector evaluation uses CNN1D, GRU, and LSTM models on the common set of 77 time series. It evaluates:

<pre class="overflow-visible! px-0!" data-start="8338" data-end="8418"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>Chronological 60/20/20
Bin-level stratified random
CASTS-05
CASTS-35</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

The completed run contains:

<pre class="overflow-visible! px-0!" data-start="8449" data-end="8527"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>27,342 raw evaluation rows
924 dataset--method--model summary rows</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

AP and AUROC are computed from continuous anomaly scores. Balanced accuracy and F1 use binary predictions at the fixed score threshold of 0.5.

## Generated paper tables

| Paper table | Generated file                                                  |
| ----------- | --------------------------------------------------------------- |
| Table 1     | `results/tables/table1_benchmark_summary.csv`                 |
| Table 2     | `results/tables/table2_split_method_summary.csv`              |
| Table 3     | `results/tables/table3_coverage_crossing_summary.csv`         |
| Table 4     | `results/tables/table4_lambda_tradeoff_summary.csv`           |
| Table 5     | `results/tables/table5_detector_method_summary.csv`           |
| Table 9     | `results/tables/table9_family_casts_summary.csv`              |
| Table 10    | `results/tables/table10_family_lambda_anomaly_difference.csv` |
| Table 11    | `results/tables/table11_family_lambda_contiguous_count.csv`   |
| Table 12    | `results/tables/table12_family_coverage_crossing.csv`         |
| Table 13    | `results/tables/table13_detector_summary_by_model.csv`        |

## Release verification

`make verify` regenerates the tables and checks the consistency of the current artifacts. A successful verification confirms:

* the benchmark-family counts;
* fixed assignment coverage;
* nonempty train, validation, and test splits;
* bin-boundary alignment for bin-based methods;
* continuity-weight coverage;
* the common 77 detector evaluation set;
* detector summary completeness;
* raw-to-summary consistency;
* expected table values;
* the absence of private absolute paths.

The completed canonical run ended with:

<pre class="overflow-visible! px-0!" data-start="9970" data-end="10009"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>release verification passed</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></pre>

## Data handling

The repository does not redistribute NAB, SKAB, SMD, SMAP, or MSL data. Users should obtain these datasets from their upstream sources and follow their respective licenses and terms. `DATASETS.md` specifies the expected local directory layout.
