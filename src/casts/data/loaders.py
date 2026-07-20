from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesData:
    dataset_id: str
    benchmark: str
    source_file: str
    timestamps: np.ndarray | None
    index: np.ndarray
    X: np.ndarray
    y: np.ndarray
    positive_episodes: tuple[tuple[int, int], ...]

    @property
    def length(self) -> int:
        return int(self.y.size)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def anomaly_count(self) -> int:
        return int(self.y.sum())

    @property
    def anomaly_rate(self) -> float:
        return float(self.anomaly_count / self.length) if self.length else 0.0


def contiguous_episodes(y: np.ndarray) -> tuple[tuple[int, int], ...]:
    labels = np.asarray(y).astype(np.int8, copy=False)
    pos = np.flatnonzero(labels == 1)
    if pos.size == 0:
        return tuple()
    starts = [int(pos[0])]
    ends: list[int] = []
    for prev, cur in zip(pos[:-1], pos[1:]):
        if int(cur) != int(prev) + 1:
            ends.append(int(prev))
            starts.append(int(cur))
    ends.append(int(pos[-1]))
    return tuple(zip(starts, ends))


def labels_from_windows(timestamps: pd.Series, windows: Iterable[Iterable[str]]) -> np.ndarray:
    ts = pd.to_datetime(timestamps)
    y = np.zeros(len(ts), dtype=np.int8)
    for item in windows:
        start_raw, end_raw = list(item)
        mask = (ts >= pd.Timestamp(start_raw)) & (ts <= pd.Timestamp(end_raw))
        y[mask.to_numpy()] = 1
    return y


def load_nab(data_root: Path) -> list[SeriesData]:
    root = data_root / 'NAB' / 'raw'
    labels_path = root / 'labels' / 'combined_windows.json'
    if not labels_path.exists():
        return []
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    out: list[SeriesData] = []
    for path in sorted(root.glob('real*/**/*.csv')):
        df = pd.read_csv(path)
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            continue
        rel = str(path.relative_to(root)).replace('\\', '/')
        y = labels_from_windows(df['timestamp'], labels.get(rel, []))
        episodes = contiguous_episodes(y)
        if not episodes:
            continue
        X = df[['value']].to_numpy(dtype=np.float32)
        out.append(SeriesData(
            dataset_id=f'nab_{path.parent.name}_{path.stem}',
            benchmark='NAB',
            source_file=str(path.relative_to(data_root)),
            timestamps=df['timestamp'].to_numpy(),
            index=np.arange(len(y), dtype=np.int64),
            X=X,
            y=y,
            positive_episodes=episodes,
        ))
    return out


def load_skab(data_root: Path) -> list[SeriesData]:
    root = data_root / 'SKAB' / 'raw'
    out: list[SeriesData] = []
    for path in sorted(root.glob('**/*.csv')):
        df = pd.read_csv(path, sep=None, engine='python')
        if 'anomaly' not in df.columns:
            continue
        feature_cols = [
            col for col in df.columns
            if col not in {'datetime', 'anomaly', 'changepoint'} and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not feature_cols:
            continue
        y = df['anomaly'].fillna(0).astype(int).clip(0, 1).to_numpy(dtype=np.int8)
        episodes = contiguous_episodes(y)
        if not episodes:
            continue
        X = df[feature_cols].to_numpy(dtype=np.float32)
        out.append(SeriesData(
            dataset_id=f"skab_{path.relative_to(root).with_suffix('').as_posix().replace('/', '_')}",
            benchmark='SKAB',
            source_file=str(path.relative_to(data_root)),
            timestamps=df['datetime'].to_numpy() if 'datetime' in df.columns else None,
            index=np.arange(len(y), dtype=np.int64),
            X=X,
            y=y,
            positive_episodes=episodes,
        ))
    return out


def load_txt_array(path: Path, dtype: type[np.floating] | type[np.integer] = np.float32) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=',', dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_smd(data_root: Path) -> list[SeriesData]:
    root = data_root / 'SMD' / 'raw' / 'ServerMachineDataset'
    test_root = root / 'test'
    label_root = root / 'test_label'
    out: list[SeriesData] = []
    for test_path in sorted(test_root.glob('machine-*.txt')):
        label_path = label_root / test_path.name
        if not label_path.exists():
            continue
        X = load_txt_array(test_path, dtype=np.float32)
        y = np.loadtxt(label_path, delimiter=',', dtype=np.int8).reshape(-1).astype(np.int8, copy=False)
        if len(y) != X.shape[0]:
            n = min(len(y), X.shape[0])
            X = X[:n]
            y = y[:n]
        y = np.clip(y, 0, 1).astype(np.int8, copy=False)
        episodes = contiguous_episodes(y)
        if not episodes:
            continue
        out.append(SeriesData(
            dataset_id=f'smd_{test_path.stem}',
            benchmark='SMD',
            source_file=str(test_path.relative_to(data_root)),
            timestamps=None,
            index=np.arange(len(y), dtype=np.int64),
            X=X,
            y=y,
            positive_episodes=episodes,
        ))
    return out


def parse_anomaly_sequences(value: object) -> list[tuple[int, int]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parsed = ast.literal_eval(str(value))
    return [(int(start), int(end)) for start, end in parsed]


def union_clamped_intervals(intervals: Iterable[tuple[int, int]], n_rows: int) -> tuple[tuple[int, int], ...]:
    cleaned = []
    for start, end in intervals:
        lo = max(0, int(start))
        hi = min(n_rows - 1, int(end))
        if hi >= lo:
            cleaned.append((lo, hi))
    if not cleaned:
        return tuple()
    cleaned.sort()
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def nasa_labels_for_channel(labels: pd.DataFrame, channel: str, n_rows: int) -> tuple[str, np.ndarray, tuple[tuple[int, int], ...]]:
    channel_rows = labels[labels['chan_id'].astype(str) == channel]
    family = str(channel_rows['spacecraft'].iloc[0]).upper() if not channel_rows.empty and 'spacecraft' in channel_rows else 'NASA'
    intervals: list[tuple[int, int]] = []
    for value in channel_rows.get('anomaly_sequences', []):
        intervals.extend(parse_anomaly_sequences(value))
    episodes = union_clamped_intervals(intervals, n_rows)
    y = np.zeros(n_rows, dtype=np.int8)
    for start, end in episodes:
        y[start:end + 1] = 1
    return family, y, episodes


def load_nasa(data_root: Path) -> list[SeriesData]:
    root = data_root / 'NASA' / 'raw'
    labels_path = root / 'labeled_anomalies.csv'
    test_root = root / 'test'
    if not labels_path.exists():
        return []
    labels = pd.read_csv(labels_path)
    out: list[SeriesData] = []
    for test_path in sorted(test_root.glob('*.npy')):
        X = np.load(test_path).astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        family, y, episodes = nasa_labels_for_channel(labels, test_path.stem, X.shape[0])
        if family not in {'SMAP', 'MSL'} or not episodes:
            continue
        out.append(SeriesData(
            dataset_id=f'{family.lower()}_{test_path.stem}',
            benchmark=family,
            source_file=str(test_path.relative_to(data_root)),
            timestamps=None,
            index=np.arange(len(y), dtype=np.int64),
            X=X,
            y=y,
            positive_episodes=episodes,
        ))
    return out


def load_public_series(data_root: Path, families: Iterable[str] = ('NAB', 'SKAB', 'SMD', 'SMAP', 'MSL')) -> list[SeriesData]:
    data_root = Path(data_root)
    requested = {item.upper() for item in families}
    series: list[SeriesData] = []
    if 'NAB' in requested:
        series.extend(load_nab(data_root))
    if 'SKAB' in requested:
        series.extend(load_skab(data_root))
    if 'SMD' in requested:
        series.extend(load_smd(data_root))
    if {'SMAP', 'MSL'} & requested:
        series.extend([item for item in load_nasa(data_root) if item.benchmark in requested])
    return sorted(series, key=lambda item: (item.benchmark, item.dataset_id))


def dataset_manifest(series: Iterable[SeriesData]) -> pd.DataFrame:
    rows = []
    for item in series:
        rows.append({
            'dataset_id': item.dataset_id,
            'benchmark': item.benchmark,
            'source_file': item.source_file,
            'length': item.length,
            'n_features': item.n_features,
            'anomaly_count': item.anomaly_count,
            'anomaly_rate': item.anomaly_rate,
            'included': True,
            'exclusion_reason': '',
        })
    return pd.DataFrame(rows).sort_values(['benchmark', 'dataset_id']).reset_index(drop=True)
