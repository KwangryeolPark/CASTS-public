#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
import re
import sys
import numpy as np
import pandas as pd

from casts.reporting.tables import (
    table1, split_method_summaries, lambda_summary, family_tables,
    detector_summaries, common_77, dataset_manifest,
)

EXPECTED_COUNTS = {'NAB': 7, 'SKAB': 34, 'SMD': 28, 'SMAP': 54, 'MSL': 27}
EXPECTED_LAMBDAS = {0.00, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50}
TARGET_METHODS = {'Chronological 60/20/20','Point-level stratified random','Bin-level stratified random','CASTS-05','CASTS-35'}
TABLES = [
    'table1_benchmark_summary','table2_split_method_summary','table3_coverage_crossing_summary',
    'table4_lambda_tradeoff_summary','table5_detector_method_summary','table9_family_casts_summary',
    'table10_family_lambda_anomaly_difference','table11_family_lambda_contiguous_count',
    'table12_family_coverage_crossing','table13_detector_summary_by_model',
]


def fail(msg: str) -> None:
    raise AssertionError(msg)


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(3)
    return out.astype(str)


def assert_frame_equal_rounded(actual_path: Path, expected_path: Path) -> None:
    actual = rounded(pd.read_csv(actual_path))
    expected = pd.read_csv(expected_path).astype(str)
    if actual.shape != expected.shape or list(actual.columns) != list(expected.columns) or not actual.equals(expected):
        fail(f'rounded table mismatch: {actual_path} vs {expected_path}')



def assert_frame_close(actual: pd.DataFrame, expected: pd.DataFrame, name: str, tol: float = 1e-12) -> None:
    if actual.shape != expected.shape or list(actual.columns) != list(expected.columns):
        fail(f'{name} shape or columns differ')
    for col in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[col]) and pd.api.types.is_numeric_dtype(expected[col]):
            if not np.allclose(actual[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float), rtol=tol, atol=tol, equal_nan=True):
                fail(f'{name} numeric column differs: {col}')
        else:
            a = actual[col].fillna('').astype(str).reset_index(drop=True)
            e = expected[col].fillna('').astype(str).reset_index(drop=True)
            if not a.equals(e):
                fail(f'{name} text column differs: {col}')


def validate_split_assignment_artifact(path: Path, datasets: pd.DataFrame) -> None:
    required = {
        'dataset', 'benchmark', 'method', 'lambda_cont', 'seed', 'replicate',
        'split', 'interval_start', 'interval_end', 'bin_size'
    }
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        fail(f'split assignment artifact is missing columns: {sorted(missing)}')
    length_by_dataset = dict(zip(datasets['dataset_id'], datasets['length']))
    expected_splits = {'train', 'validation', 'test'}
    key_cols = ['dataset', 'benchmark', 'method', 'lambda_cont', 'seed', 'replicate']
    if df.empty:
        fail('split assignment artifact is empty')
    for key, group in df.groupby(key_cols, dropna=False, sort=False):
        dataset, _benchmark, method, _lambda_cont, _seed, _replicate = key
        if dataset not in length_by_dataset:
            fail(f'split assignment references unknown dataset: {dataset}')
        length = int(length_by_dataset[dataset])
        observed_splits = set(group['split'])
        if observed_splits != expected_splits:
            fail(f'{key} has splits {sorted(observed_splits)}, expected train/validation/test')
        if group.groupby('split').size().le(0).any():
            fail(f'{key} has an empty split')
        bin_sizes = set(int(x) for x in group['bin_size'].dropna().unique())
        if len(bin_sizes) != 1:
            fail(f'{key} has inconsistent bin_size values: {sorted(bin_sizes)}')
        bin_size = next(iter(bin_sizes))
        intervals = group[['interval_start', 'interval_end']].astype(int).sort_values(['interval_start', 'interval_end'])
        cursor = 0
        total = 0
        for lo, hi in intervals.itertuples(index=False):
            if lo < 0 or hi < lo or hi >= length:
                fail(f'{key} has invalid interval {lo}-{hi} for length {length}')
            if lo != cursor:
                fail(f'{key} coverage error near index {cursor}: next interval is {lo}-{hi}')
            cursor = hi + 1
            total += hi - lo + 1
            is_bin_method = method == 'Bin-level stratified random' or str(method).startswith('CASTS')
            if is_bin_method and bin_size > 1:
                if lo % bin_size != 0:
                    fail(f'{key} interval starts inside a bin: {lo}')
                if hi != length - 1 and (hi + 1) % bin_size != 0:
                    fail(f'{key} interval ends inside a bin: {hi}')
        if cursor != length or total != length:
            fail(f'{key} does not cover exactly {length} observations; covered {total}')


def assert_raw_summary_consistency(results: Path) -> None:
    actual_datasets = pd.read_csv(ROOT / 'manifests' / 'datasets.csv')
    assert_frame_close(actual_datasets, dataset_manifest(results), 'manifests/datasets.csv')
    t2, t3 = split_method_summaries(results)
    t4 = lambda_summary(results)
    t9, t12, _t10, _t11, lambda_family = family_tables(results)
    common = common_77(results)
    t5, t13 = detector_summaries(results)
    checks = [
        (results / 'summaries' / 'split_method_summary.csv', t2, 'split_method_summary.csv'),
        (results / 'summaries' / 'coverage_crossing_summary.csv', t3, 'coverage_crossing_summary.csv'),
        (results / 'summaries' / 'lambda_tradeoff_summary.csv', t4, 'lambda_tradeoff_summary.csv'),
        (results / 'summaries' / 'family_casts_summary.csv', t9, 'family_casts_summary.csv'),
        (results / 'summaries' / 'family_coverage_crossing.csv', t12, 'family_coverage_crossing.csv'),
        (results / 'summaries' / 'lambda_summary_by_family.csv', lambda_family, 'lambda_summary_by_family.csv'),
        (results / 'summaries' / 'common_77_datasets.csv', common, 'common_77_datasets.csv'),
        (results / 'summaries' / 'detector_method_summary.csv', t5, 'detector_method_summary.csv'),
        (results / 'summaries' / 'detector_summary_by_model.csv', t13, 'detector_summary_by_model.csv'),
    ]
    for path, expected, name in checks:
        assert_frame_close(pd.read_csv(path), expected, name)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', type=Path, default=Path('results'))
    args = ap.parse_args()
    results = args.results_root

    datasets = pd.read_csv(ROOT / 'manifests' / 'datasets.csv')
    if len(datasets) != 150:
        fail(f'dataset count is {len(datasets)}, expected 150')
    counts = datasets.groupby('benchmark').size().to_dict()
    for fam, expected in EXPECTED_COUNTS.items():
        if int(counts.get(fam, 0)) != expected:
            fail(f'{fam} count mismatch: {counts.get(fam, 0)} != {expected}')

    split = pd.read_csv(results / 'summaries' / 'split_method_summary.csv')
    if set(split['Method']) != TARGET_METHODS:
        fail(f'target split methods mismatch: {set(split["Method"])}')

    lam = pd.read_csv(results / 'summaries' / 'lambda_tradeoff_summary.csv')
    if {round(float(x), 2) for x in lam['lambda_cont']} != EXPECTED_LAMBDAS:
        fail('lambda sweep values do not match expected seven values')


    split_artifact = ROOT / 'splits' / 'fixed_split_intervals.csv.gz'
    if not split_artifact.exists() or split_artifact.stat().st_size == 0:
        fail('fixed split assignment artifact is missing or empty')
    validate_split_assignment_artifact(split_artifact, datasets)

    generated = pd.read_csv(results / 'summaries' / 'generated_split_method_summary.csv')
    expected_generated_methods = TARGET_METHODS | {'CASTS-lambda-0', 'CASTS-lambda-0.01', 'CASTS-lambda-0.1', 'CASTS-lambda-0.2', 'CASTS-lambda-0.5'}
    if set(generated['method']) != expected_generated_methods:
        fail(f'generated split assignment methods mismatch: {set(generated["method"])}')
    if not (generated['split_rows'] == 6750).all():
        fail('generated split summaries do not have 6750 split rows per method')
    if not (generated['specs'] == 2250).all():
        fail('generated split summaries do not have 2250 specs per method')
    generated_lambdas = {0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50}
    lambda_methods = generated[generated['method'].str.startswith('CASTS')]['method'].tolist()
    if len(lambda_methods) < len(generated_lambdas):
        fail('generated split assignment does not cover all lambda settings')

    common = pd.read_csv(results / 'summaries' / 'common_77_datasets.csv')
    if len(common) != 77:
        fail(f'common evaluation set has {len(common)} datasets, expected 77')

    det_raw = pd.read_csv(results / 'raw' / 'detector_model_metrics_summary.csv')
    common_set = set(common['dataset'])
    det_common = det_raw[det_raw['dataset'].isin(common_set)]
    if det_common[['mean_auroc','mean_average_precision']].isna().any().any():
        fail('NaN detector AP/AUROC found in common 77 model summary rows')
    det_common = det_common.copy()
    public_method = det_common['method'].replace({
        'Random stratified bins': 'Bin-level stratified random',
        'CASTS-lambda-0.05': 'CASTS-05',
        'CASTS-lambda-0.35': 'CASTS-35',
    })
    det_common['method_public'] = public_method
    needed = {'Chronological 60/20/20', 'Bin-level stratified random', 'CASTS-05', 'CASTS-35'}
    combos = (
        det_common[det_common['method_public'].isin(needed)]
        .groupby('dataset')
        .apply(lambda g: len(set(zip(g['method_public'], g['model']))))
    )
    if not (combos == 12).all():
        fail('common 77 detector rows do not contain all 12 method/model combinations')

    assert_raw_summary_consistency(results)

    for name in TABLES:
        assert_frame_equal_rounded(results / 'tables' / f'{name}.csv', results / 'expected' / f'{name}.csv')

    private_patterns = [re.compile('/' + 'root/'), re.compile('/' + 'workspace/')]
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix not in {'.pyc'}:
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            for pat in private_patterns:
                if pat.search(text):
                    fail(f'private absolute path found in {path}')

    print('release verification passed')


if __name__ == '__main__':
    try:
        main()
    except AssertionError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
