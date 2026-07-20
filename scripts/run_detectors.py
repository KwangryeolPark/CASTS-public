#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from casts.data.loaders import load_public_series
from casts.detectors.torch_models import apply_standardizer, build_model, fit_standardizer, make_windows
from casts.splitting.methods import build_specs_for_labels

DETECTOR_METHODS = ['Chronological 60/20/20', 'Bin-level stratified random', 'CASTS-05', 'CASTS-35']
MODELS = ['CNN1D', 'GRU', 'LSTM']
SPLITS = ['train', 'validation', 'test']


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_assignments(path: Path) -> pd.DataFrame:
    compression = 'gzip' if path.suffix == '.gz' else None
    df = pd.read_csv(path, compression=compression)
    return df[df['method'].isin(DETECTOR_METHODS)].copy()


def read_dataset_list(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if 'dataset' in df.columns:
        values = df['dataset']
    elif 'dataset_id' in df.columns:
        values = df['dataset_id']
    else:
        raise ValueError(f'{path} must contain a dataset or dataset_id column')
    return {str(value) for value in values.dropna()}


def intervals_for(group: pd.DataFrame, split: str) -> tuple[tuple[int, int], ...]:
    d = group[group['split'] == split].sort_values(['interval_start','interval_end'])
    return tuple((int(row.interval_start), int(row.interval_end)) for row in d.itertuples(index=False))


def build_bin_level_interval_overrides(series_items: list, assignments: pd.DataFrame) -> dict[tuple[str, int, int], tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    subset = assignments[assignments['method'] == 'Bin-level stratified random'].copy()
    if subset.empty:
        return {}
    seeds = sorted(int(value) for value in subset['seed'].dropna().unique().tolist())
    replicates = int(subset['replicate'].max()) + 1
    wanted = {(str(row.dataset), int(row.seed), int(row.replicate)) for row in subset[['dataset', 'seed', 'replicate']].drop_duplicates().itertuples(index=False)}
    overrides: dict[tuple[str, int, int], tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {}
    for item in series_items:
        specs = build_specs_for_labels(item.dataset_id, item.benchmark, item.y, seeds, replicates, lambda_values=())
        for spec in specs:
            if spec.method != 'Bin-level stratified random':
                continue
            key = (str(spec.dataset), int(spec.seed), int(spec.replicate))
            if key in wanted:
                overrides[key] = spec.intervals_by_split
    return overrides


def predict_scores(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def fit_single_model(model_name: str, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, seed: int, epochs: int, batch_size: int, device: torch.device) -> nn.Module:
    set_seed(seed)
    model = build_model(model_name, train_x.shape[-1]).to(device)
    pos = float(train_y.sum())
    neg = float(len(train_y) - train_y.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))), batch_size=batch_size, shuffle=True)
    best_state = None
    best_ap = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        val_score = predict_scores(model, val_x, batch_size, device)
        ap = average_precision_score(val_y, val_score) if len(np.unique(val_y)) == 2 else 0.0
        if ap >= best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(np.int64)
    two_class = len(np.unique(y_true)) == 2
    return {
        'balanced_accuracy': balanced_accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred, zero_division=0),
        'auroc': roc_auc_score(y_true, score) if two_class else np.nan,
        'average_precision': average_precision_score(y_true, score) if two_class else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Run CNN1D/GRU/LSTM detector evaluation on local benchmark data and split assignments.')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs' / 'detector.yaml')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--splits', type=Path, default=ROOT / 'splits' / 'fixed_split_intervals.csv.gz')
    ap.add_argument('--output', type=Path, default=ROOT / 'results' / 'raw' / 'fresh_detector_results_by_run.csv')
    ap.add_argument('--summary-output', type=Path, default=ROOT / 'results' / 'summaries' / 'fresh_detector_model_metrics_summary.csv')
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--methods', nargs='+', default=DETECTOR_METHODS)
    ap.add_argument('--window', type=int, default=32)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--limit-datasets', type=int, default=None, help='optional smoke-test limit')
    ap.add_argument('--dataset-list', type=Path, default=None, help='optional CSV with dataset or dataset_id column')
    args = ap.parse_args()

    assignments = read_assignments(args.splits)
    assignments = assignments[assignments['method'].isin(args.methods)].copy()
    series = load_public_series(args.data_root)
    if args.dataset_list is not None:
        keep = read_dataset_list(args.dataset_list)
        series = [item for item in series if item.dataset_id in keep]
        assignments = assignments[assignments['dataset'].isin(keep)].copy()
    if args.limit_datasets is not None:
        keep = {item.dataset_id for item in series[:args.limit_datasets]}
        series = [item for item in series if item.dataset_id in keep]
        assignments = assignments[assignments['dataset'].isin(keep)].copy()
    series_map = {item.dataset_id: item for item in series}
    bin_level_overrides = build_bin_level_interval_overrides(series, assignments)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    key_cols = ['dataset','benchmark','method','lambda_cont','seed','replicate']
    for key, group in assignments.groupby(key_cols, dropna=False):
        dataset, benchmark, method, lambda_cont, seed, replicate = key
        if dataset not in series_map:
            continue
        item = series_map[dataset]
        if method == 'Bin-level stratified random':
            split_intervals = bin_level_overrides.get((str(dataset), int(seed), int(replicate)))
            if split_intervals is None:
                continue
            train_intervals, val_intervals, test_intervals = split_intervals
        else:
            train_intervals = intervals_for(group, 'train')
            val_intervals = intervals_for(group, 'validation')
            test_intervals = intervals_for(group, 'test')
        train_x, train_y = make_windows(item.X, item.y, train_intervals, args.window, args.stride)
        val_x, val_y = make_windows(item.X, item.y, val_intervals, args.window, args.stride)
        test_x, test_y = make_windows(item.X, item.y, test_intervals, args.window, args.stride)
        if min(len(train_y), len(val_y), len(test_y)) == 0:
            continue
        mean, std = fit_standardizer(train_x)
        train_x = apply_standardizer(train_x, mean, std)
        val_x = apply_standardizer(val_x, mean, std)
        test_x = apply_standardizer(test_x, mean, std)
        for model_name in args.models:
            model = fit_single_model(model_name, train_x, train_y, val_x, val_y, int(seed) + int(replicate) + 1000, args.epochs, args.batch_size, device)
            for split_name, X_eval, y_eval in [('validation', val_x, val_y), ('test', test_x, test_y)]:
                scores = predict_scores(model, X_eval, args.batch_size, device)
                row = {
                    'dataset': dataset,
                    'benchmark': benchmark,
                    'method': method,
                    'lambda_cont': lambda_cont,
                    'seed': int(seed),
                    'replicate': int(replicate),
                    'model': model_name,
                    'split': split_name,
                    'train_windows': int(len(train_y)),
                    'eval_windows': int(len(y_eval)),
                    'train_positive_rate': float(train_y.mean()),
                    'eval_positive_rate': float(y_eval.mean()),
                }
                row.update(score_metrics(y_eval, scores))
                rows.append(row)
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    summary = out[out['split'] == 'test'].groupby(['dataset','method','model'], as_index=False).agg(
        mean_balanced_accuracy=('balanced_accuracy','mean'),
        std_balanced_accuracy=('balanced_accuracy','std'),
        mean_f1=('f1','mean'),
        mean_auroc=('auroc','mean'),
        mean_average_precision=('average_precision','mean'),
        replicates=('replicate','nunique'),
        seeds=('seed','nunique'),
    )
    summary.to_csv(args.summary_output, index=False)
    print(f'wrote {args.output} ({len(out)} rows)')
    print(f'wrote {args.summary_output} ({len(summary)} rows)')


if __name__ == '__main__':
    main()
