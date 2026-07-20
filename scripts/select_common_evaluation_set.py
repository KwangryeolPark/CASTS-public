#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from casts.reporting.tables import common_77


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', type=Path, default=Path('results'))
    args = ap.parse_args()
    out = args.results_root / 'summaries' / 'common_77_datasets.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df = common_77(args.results_root)
    df.to_csv(out, index=False)
    print(f'wrote {out} ({len(df)} datasets)')


if __name__ == '__main__':
    main()
