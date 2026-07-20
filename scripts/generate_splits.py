#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from casts.data.loaders import load_public_series
from casts.splitting.methods import SPLIT_NAMES, build_specs_for_labels

DEFAULT_SEEDS = [3, 17, 29, 41, 53]
DEFAULT_LAMBDAS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50]


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(',') if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item) for item in text.split(',') if item.strip()]


def split_allowed(method: str, requested: set[str]) -> bool:
    if 'all' in requested:
        return True
    if method in requested:
        return True
    if method.startswith('CASTS-lambda-') and 'lambda-sweep' in requested:
        return True
    if method in {'CASTS-05', 'CASTS-35'} and ('main' in requested or 'lambda-sweep' in requested):
        return True
    return False


def validate_spec(series_length: int, spec) -> None:
    seen = 0
    for split_id, intervals in enumerate(spec.intervals_by_split):
        if not intervals:
            raise ValueError(f'{spec.dataset}/{spec.method}/seed{spec.seed}/rep{spec.replicate}: empty split {split_id}')
        for lo, hi in intervals:
            if lo < 0 or hi >= series_length or hi < lo:
                raise ValueError(f'{spec.dataset}/{spec.method}: invalid interval {lo}-{hi}')
            seen += hi - lo + 1
    if seen != series_length:
        raise ValueError(f'{spec.dataset}/{spec.method}/seed{spec.seed}/rep{spec.replicate}: intervals cover {seen}, expected {series_length}')


def worker(payload):
    dataset_id, benchmark, y, seeds, replicates, lambdas, methods = payload
    rows = []
    specs = build_specs_for_labels(dataset_id, benchmark, y, seeds, replicates, lambdas)
    for spec in specs:
        if not split_allowed(spec.method, methods):
            continue
        validate_spec(len(y), spec)
        for split_id, intervals in enumerate(spec.intervals_by_split):
            for interval_id, (lo, hi) in enumerate(intervals):
                rows.append({
                    'dataset': dataset_id,
                    'benchmark': benchmark,
                    'method': spec.method,
                    'lambda_cont': spec.lambda_cont,
                    'seed': spec.seed,
                    'replicate': spec.replicate,
                    'split': SPLIT_NAMES[split_id],
                    'interval_id': interval_id,
                    'interval_start': int(lo),
                    'interval_end': int(hi),
                    'bin_size': int(spec.bin_size),
                })
    return dataset_id, rows


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate fixed split interval assignments from local benchmark data.')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT / 'splits' / 'fixed_split_intervals.csv.gz')
    ap.add_argument('--seeds', default=','.join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument('--replicates', type=int, default=3)
    ap.add_argument('--lambdas', default=','.join(str(x) for x in DEFAULT_LAMBDAS))
    ap.add_argument('--methods', nargs='+', default=['all'], help='all, main, lambda-sweep, or explicit method names')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()

    series = load_public_series(args.data_root)
    if len(series) != 150:
        raise SystemExit(f'loaded {len(series)} included series, expected 150; check DATASETS.md layout')

    seeds = parse_ints(args.seeds)
    lambdas = parse_floats(args.lambdas)
    methods = set(args.methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset','benchmark','method','lambda_cont','seed','replicate','split','interval_id','interval_start','interval_end','bin_size']
    total_rows = 0
    total_series = 0
    payloads = [(item.dataset_id, item.benchmark, item.y, seeds, args.replicates, lambdas, methods) for item in series]
    with gzip.open(args.output, 'wt', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(worker, payload) for payload in payloads]
                for fut in as_completed(futures):
                    dataset_id, rows = fut.result()
                    rows.sort(key=lambda row: (row['dataset'], row['method'], row['seed'], row['replicate'], row['split'], row['interval_id']))
                    writer.writerows(rows)
                    total_rows += len(rows)
                    total_series += 1
                    print(f'processed {total_series}/{len(payloads)} series: {dataset_id}, rows={len(rows)}', flush=True)
        else:
            for payload in payloads:
                dataset_id, rows = worker(payload)
                writer.writerows(rows)
                total_rows += len(rows)
                total_series += 1
                print(f'processed {total_series}/{len(payloads)} series: {dataset_id}, rows={len(rows)}', flush=True)
    print(f'wrote {args.output} with {total_rows} interval rows from {total_series} series')


if __name__ == '__main__':
    main()
