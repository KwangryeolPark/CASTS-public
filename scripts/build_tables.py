#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
import hashlib
import pandas as pd
from casts.reporting.tables import (
    table1, split_method_summaries, lambda_summary, family_tables,
    detector_summaries, common_77, dataset_manifest, write_csv_and_tex,
)


def rounded_for_expected(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(3)
    return out


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', type=Path, default=Path('results'))
    ap.add_argument('--output-dir', type=Path, default=Path('results/tables'))
    ap.add_argument('--update-expected', action='store_true', help='overwrite static expected rounded tables')
    args = ap.parse_args()
    results_root = args.results_root
    output_dir = args.output_dir
    summaries = results_root / 'summaries'
    expected = results_root / 'expected'
    summaries.mkdir(parents=True, exist_ok=True)
    expected.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = dataset_manifest(results_root)
    datasets.to_csv(ROOT / 'manifests' / 'datasets.csv', index=False)

    t1 = table1(results_root)
    t2, t3 = split_method_summaries(results_root)
    t4 = lambda_summary(results_root)
    t9, t12, t10, t11, lambda_family = family_tables(results_root)
    common = common_77(results_root)
    common.to_csv(summaries / 'common_77_datasets.csv', index=False)
    t5, t13 = detector_summaries(results_root)

    # Required summary files.
    t2.to_csv(summaries / 'split_method_summary.csv', index=False)
    t3.to_csv(summaries / 'coverage_crossing_summary.csv', index=False)
    t4.to_csv(summaries / 'lambda_tradeoff_summary.csv', index=False)
    t9.to_csv(summaries / 'family_casts_summary.csv', index=False)
    t12.to_csv(summaries / 'family_coverage_crossing.csv', index=False)
    lambda_family.to_csv(summaries / 'lambda_summary_by_family.csv', index=False)
    t5.to_csv(summaries / 'detector_method_summary.csv', index=False)
    t13.to_csv(summaries / 'detector_summary_by_model.csv', index=False)

    # Table outputs.
    tables = {
        'table1_benchmark_summary': t1,
        'table2_split_method_summary': t2[['Method','Mean anomaly difference','Mean contiguous count']],
        'table3_coverage_crossing_summary': t3[['Method','Both-class rate','Crossing rate']],
        'table4_lambda_tradeoff_summary': t4[['lambda_cont','Anomaly difference','Contiguous count']],
        'table5_detector_method_summary': t5[['Method','BAcc','F1','AUROC','AP']],
        'table9_family_casts_summary': t9[['Family','Config','Anomaly difference','Contiguous count','Both-class rate','Crossing rate']],
        'table10_family_lambda_anomaly_difference': t10,
        'table11_family_lambda_contiguous_count': t11,
        'table12_family_coverage_crossing': t12[['Family','Config','Both-class rate','Crossing rate']],
        'table13_detector_summary_by_model': t13[['Model','Method','BAcc','F1','AUROC','AP']],
    }
    for name, df in tables.items():
        write_csv_and_tex(df, output_dir, name)
        expected_path = expected / f'{name}.csv'
        if args.update_expected or not expected_path.exists():
            rounded_for_expected(df).to_csv(expected_path, index=False)

    artifacts = []
    for path in sorted([*Path('splits').glob('*.gz'), *Path('results').glob('raw/*.csv'), *summaries.glob('*.csv'), *output_dir.glob('*.csv'), *expected.glob('*.csv')]):
        artifacts.append({'path': path.as_posix(), 'sha256': checksum_file(path), 'bytes': path.stat().st_size})
    pd.DataFrame(artifacts).to_csv(ROOT / 'manifests' / 'artifacts.csv', index=False)
    with (ROOT / 'manifests' / 'checksums.sha256').open('w', encoding='utf-8') as f:
        for item in artifacts:
            f.write(f"{item['sha256']}  {item['path']}\n")

    print(f'wrote {len(tables)} tables to {output_dir}')


if __name__ == '__main__':
    main()
