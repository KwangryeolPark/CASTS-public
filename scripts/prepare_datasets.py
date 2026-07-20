#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from casts.data.loaders import dataset_manifest, load_public_series


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare dataset manifest from a local benchmark data root.')
    ap.add_argument('--data-root', type=Path, required=True)
    # ap.add_argument('--output', type=Path, default=ROOT / 'manifests' / 'datasets.csv')
    ap.add_argument('--output', type=Path, default=ROOT / 'results' / 'raw' / "dataset_inventory.csv")
    args = ap.parse_args()
    if not args.data_root.exists():
        raise SystemExit(f'data root does not exist: {args.data_root}')
    series = load_public_series(args.data_root)
    if len(series) != 150:
        raise SystemExit(f'loaded {len(series)} included series, expected 150; check DATASETS.md layout')
    out = dataset_manifest(series)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'wrote {args.output} with {len(out)} included series')


if __name__ == '__main__':
    main()
