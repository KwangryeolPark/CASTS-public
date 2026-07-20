#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from casts.data.loaders import load_public_series
from casts.metrics.split_metrics import anomaly_segment_crossing_rate, intervals_to_indices

SPLIT_ORDER = ['train', 'validation', 'test']
SPLIT_ID = {name: idx for idx, name in enumerate(SPLIT_ORDER)}
TARGET_RATIO = {'train': 0.6, 'validation': 0.2, 'test': 0.2}


def normalize_method(method: str) -> str:
    return {
        'CASTS-05': 'CASTS-05',
        'CASTS-35': 'CASTS-35',
        'Point-level stratified random': 'Point-level stratified random',
        'Bin-level stratified random': 'Bin-level stratified random',
    }.get(method, method)


def load_assignments(path: Path) -> pd.DataFrame:
    compression = 'gzip' if path.suffix == '.gz' else None
    df = pd.read_csv(path, compression=compression)
    required = {'dataset','benchmark','method','lambda_cont','seed','replicate','split','interval_start','interval_end','bin_size'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'assignment file is missing columns: {sorted(missing)}')
    return df


def intervals_for(group: pd.DataFrame, split: str) -> tuple[tuple[int, int], ...]:
    d = group[group['split'] == split].sort_values(['interval_start','interval_end'])
    return tuple((int(row.interval_start), int(row.interval_end)) for row in d.itertuples(index=False))


def validate_assignment(series_length: int, group: pd.DataFrame, method: str, bin_size: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    intervals_by_split = tuple(intervals_for(group, split) for split in SPLIT_ORDER)
    assign = np.full(series_length, -1, dtype=np.int8)
    for split_id, intervals in enumerate(intervals_by_split):
        if not intervals:
            raise ValueError('empty split')
        for lo, hi in intervals:
            if lo < 0 or hi >= series_length or hi < lo:
                raise ValueError(f'invalid interval {lo}-{hi}')
            if np.any(assign[lo:hi + 1] != -1):
                raise ValueError(f'overlapping interval {lo}-{hi}')
            assign[lo:hi + 1] = split_id
            if method != 'Point-level stratified random' and bin_size > 1:
                if lo % bin_size != 0:
                    raise ValueError(f'bin interval starts inside a bin: {lo}')
                # Last interval may end at series_length - 1; other interval ends should align to bin boundary.
                if hi != series_length - 1 and (hi + 1) % bin_size != 0:
                    raise ValueError(f'bin interval ends inside a bin: {hi}')
    if np.any(assign < 0):
        raise ValueError(f'{int(np.sum(assign < 0))} observations are unassigned')
    return intervals_by_split


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate generated split assignments.')
    ap.add_argument('--assignments', type=Path, default=ROOT / 'splits' / 'fixed_split_intervals.csv.gz')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output-raw', type=Path, default=ROOT / 'results' / 'raw' / 'generated_split_audit_by_split.csv')
    ap.add_argument('--output-summary', type=Path, default=ROOT / 'results' / 'summaries' / 'generated_split_method_summary.csv')
    args = ap.parse_args()
    if not args.assignments.exists():
        raise SystemExit(f'assignment file not found: {args.assignments}')
    series_map = {item.dataset_id: item for item in load_public_series(args.data_root)}
    df = load_assignments(args.assignments)
    rows = []
    spec_rows = []
    key_cols = ['dataset','benchmark','method','lambda_cont','seed','replicate']
    for key, group in df.groupby(key_cols, dropna=False):
        dataset, benchmark, method, lambda_cont, seed, replicate = key
        if dataset not in series_map:
            raise SystemExit(f'assignment references unknown dataset: {dataset}')
        series = series_map[dataset]
        bin_size = int(group['bin_size'].iloc[0])
        intervals_by_split = validate_assignment(series.length, group, method, bin_size)
        crossing = anomaly_segment_crossing_rate(series.y, intervals_by_split)
        spec_rows.append({
            'dataset': dataset,
            'benchmark': benchmark,
            'method': method,
            'lambda_cont': lambda_cont,
            'seed': seed,
            'replicate': replicate,
            'crossing_rate': crossing,
            'has_any_episode_crossing': bool(crossing > 0),
        })
        global_rate = float(series.y.mean())
        for split in SPLIT_ORDER:
            intervals = intervals_by_split[SPLIT_ID[split]]
            idx = intervals_to_indices(intervals)
            y_split = series.y[idx]
            rows.append({
                'dataset': dataset,
                'benchmark': benchmark,
                'method': method,
                'lambda_cont': lambda_cont,
                'seed': seed,
                'replicate': replicate,
                'split': split,
                'n': int(idx.size),
                'size_error': abs(idx.size / series.length - TARGET_RATIO[split]),
                'positive_count': int(y_split.sum()),
                'global_positive_rate': global_rate,
                'positive_rate': float(y_split.mean()),
                'anomaly_proportion_difference': abs(float(y_split.mean()) - global_rate),
                'contiguous_count': len(intervals),
                'both_class': bool(np.unique(y_split).size == 2),
                'episode_crossing_rate': crossing,
                'has_any_episode_crossing': bool(crossing > 0),
            })
    raw = pd.DataFrame(rows)
    specs = pd.DataFrame(spec_rows)
    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_raw, index=False)
    split_summary = raw.groupby('method', as_index=False).agg(
        mean_size_error=('size_error','mean'),
        mean_anomaly_difference=('anomaly_proportion_difference','mean'),
        mean_contiguous_count=('contiguous_count','mean'),
        both_class_rate=('both_class','mean'),
        split_rows=('dataset','size'),
    )
    crossing = specs.groupby('method', as_index=False).agg(crossing_rate=('has_any_episode_crossing','mean'), specs=('dataset','size'))
    summary = split_summary.merge(crossing, on='method', how='left').sort_values('method')
    summary.to_csv(args.output_summary, index=False)
    print(f'wrote {args.output_raw} ({len(raw)} split rows)')
    print(f'wrote {args.output_summary} ({len(summary)} methods)')


if __name__ == '__main__':
    main()
