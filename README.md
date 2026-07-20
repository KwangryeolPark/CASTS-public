This repository provides the code, fixed split assignments, and generated results for CASTS: Contiguity-Aware Stratified Temporal Splits for time-series anomaly detection benchmarks.

CASTS constructs train, validation, and test splits for fully collected labeled time series. It jointly considers target split ratios, anomaly-label distribution, and temporal continuity between adjacent bins.

The original NAB, SKAB, SMD, SMAP, and MSL benchmark data are not redistributed in this repository.

## Repository structure

```text
configs/            experiment configurations and the common-77 evaluation set
src/casts/          reusable data, splitting, metrics, detector, and reporting code
scripts/            command-line entry points
manifests/          dataset and artifact manifests
splits/             generated split assignment artifacts
results/raw/        generated split, lambda, and detector result CSV files
results/summaries/  derived summary CSV files
results/expected/   rounded expected table values used by verification
results/tables/     generated paper-table CSV and LaTeX outputs
```

# Installation

```Shell
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

# Dataset preparation

Prepare the upstream datasets according to DATASETS.md, then provide their parent directory as <DATA_ROOT>.

```python
python scripts/prepare_datasets.py \
  --data-root <DATA_ROOT>
```

This command creates:

```Shell
results/raw/dataset_inventory.csv
```

# Split construction and evaluation

Generate the fixed train, validation, and test assignments.

```Shell
python scripts/generate_splits.py
  --data-root <DATA_ROOT>
  --output splits/fixed_split_intervals.csv.gz
  --jobs 8
```

Evaluate the resulting assignments.

```Shell
python scripts/evaluate_splits.py
  --data-root <DATA_ROOT>
  --assignments splits/fixed_split_intervals.csv.gz
```

These commands create:

```Shell
splits/fixed_split_intervals.csv.gz
results/raw/generated_split_audit_by_split.csv
results/summaries/generated_split_method_summary.csv
```

The split summary includes split-size error, anomaly-proportion difference, contiguous count, both-class rate, and anomaly-segment crossing rate.

# Continuity-weight experiment

Evaluate the CASTS objective across continuity weights.

```Shell
python scripts/run_lambda_sweep.py
  --config configs/lambda_sweep.yaml
  --data-root <DATA_ROOT>
  --jobs 8
```

This command creates:

```Shell
splits/lambda_sweep_intervals.csv.gz
results/raw/generated_lambda_audit_by_split.csv
results/summaries/generated_lambda_tradeoff_summary.csv
```


# Detector evaluation

The detector evaluation uses CNN1D, GRU, and LSTM models on the common set of 77 time series for which AP and AUROC are defined for all four evaluated splitting methods. The repository includes the fixed common-set list at:

```Shell
results/summaries/common_77_datasets.csv
```

Run the full detector evaluation with:

```Shell
python scripts/run_detectors.py
  --data-root <DATA_ROOT>
  --splits splits/fixed_split_intervals.csv.gz
  --dataset-list results/summaries/common_77_datasets.csv
  --models CNN1D GRU LSTM
  --methods
    "Chronological 60/20/20"
    "Bin-level stratified random"
    "CASTS-05"
    "CASTS-35"
  --window 32
  --stride 4
  --epochs 5
  --batch-size 128
  --output results/raw/detector_results_by_run.csv
  --summary-output results/raw/detector_model_metrics_summary.csv
```

The canonical full run produces:

```Shell
results/raw/detector_results_by_run.csv
results/raw/detector_model_metrics_summary.csv
```

The completed common-77 evaluation contains 27,342 raw evaluation rows and 924 dataset--method--model summary rows.

AP and AUROC are computed from continuous anomaly scores. Balanced accuracy and F1 use the detector's binary predictions at the fixed score threshold of 0.5.

# Table generation and verification

Generate Tables 1--5 and 9--13.

```Shell
make reproduce-tables
```

Run the complete release verification.

```Shell
make verify
```

The verification checks:

* the 150-series benchmark inventory and family counts;
* method and continuity-weight coverage;
* fixed split interval coverage and nonempty train, validation, and test splits;
* bin-boundary alignment for bin-based methods;
* the common 77 detector evaluation set and all 12 method--model combinations;
* raw-to-summary consistency;
* generated table values and rounded expected values;
* the absence of private absolute paths.

A successful run ends with:

```Shell
release verification passed
```

# Fixed split assignment artifact

splits/fixed_split_intervals.csv.gz records the train, validation, and test assignments for all included time series, split methods, seeds, and replicates.

The included methods are:

```Shell
Chronological 60/20/20
Point-level stratified random
Bin-level stratified random
CASTS-05
CASTS-35
CASTS-lambda-0
CASTS-lambda-0.01
CASTS-lambda-0.1
CASTS-lambda-0.2
CASTS-lambda-0.5
```

The experiments use seeds 3, 17, 29, 41, 53 and replicates 0, 1, 2.

# Citation

If you use this repository, please cite the CASTS paper and the metadata in CITATION.cff.

# License

The source code license is described in LICENSE_PENDING.md. The upstream benchmark datasets remain governed by their respective licenses and terms.

Split construction and evaluation
