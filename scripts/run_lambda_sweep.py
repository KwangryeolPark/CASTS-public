#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_config(path: Path) -> dict[str, str]:
    text = path.read_text(encoding='utf-8')
    out: dict[str, str] = {}
    m = re.search(r'lambda_cont:\s*\[([^\]]+)\]', text)
    if m:
        out['lambdas'] = ','.join(item.strip() for item in m.group(1).split(','))
    m = re.search(r'seeds:\s*\[([^\]]+)\]', text)
    if m:
        out['seeds'] = ','.join(item.strip() for item in m.group(1).split(','))
    m = re.search(r'replicates:\s*(\d+)', text)
    if m:
        out['replicates'] = m.group(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Run the CASTS continuity-weight sweep on local benchmark data.')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs' / 'lambda_sweep.yaml')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT / 'splits' / 'lambda_sweep_intervals.csv.gz')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    cfg = parse_config(args.config)
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'generate_splits.py'),
        '--data-root', str(args.data_root),
        '--output', str(args.output),
        '--methods', 'lambda-sweep',
        '--jobs', str(args.jobs),
    ]
    if 'lambdas' in cfg:
        cmd.extend(['--lambdas', cfg['lambdas']])
    if 'seeds' in cfg:
        cmd.extend(['--seeds', cfg['seeds']])
    if 'replicates' in cfg:
        cmd.extend(['--replicates', cfg['replicates']])
    subprocess.run(cmd, check=True)
    subprocess.run([
        sys.executable, str(ROOT / 'scripts' / 'evaluate_splits.py'),
        '--data-root', str(args.data_root),
        '--assignments', str(args.output),
        '--output-raw', str(ROOT / 'results' / 'raw' / 'generated_lambda_audit_by_split.csv'),
        '--output-summary', str(ROOT / 'results' / 'summaries' / 'generated_lambda_tradeoff_summary.csv'),
    ], check=True)


if __name__ == '__main__':
    main()
