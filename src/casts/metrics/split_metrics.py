from __future__ import annotations

import numpy as np


def intervals_to_indices(intervals):
    parts = [np.arange(int(start), int(end) + 1, dtype=np.int64) for start, end in intervals if int(end) >= int(start)]
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def contiguous_episodes(y: np.ndarray) -> tuple[tuple[int, int], ...]:
    pos = np.flatnonzero(np.asarray(y).astype(np.int8) == 1)
    if pos.size == 0:
        return tuple()
    starts = [int(pos[0])]
    ends = []
    for prev, cur in zip(pos[:-1], pos[1:]):
        if int(cur) != int(prev) + 1:
            ends.append(int(prev))
            starts.append(int(cur))
    ends.append(int(pos[-1]))
    return tuple(zip(starts, ends))


def split_metrics(y: np.ndarray, intervals_by_split, ratios=(0.6, 0.2, 0.2)):
    total = len(y)
    p = float(np.mean(y)) if total else float('nan')
    rows = []
    for split_id, split_name in enumerate(['train', 'validation', 'test']):
        idx = intervals_to_indices(intervals_by_split[split_id])
        split_y = y[idx] if idx.size else np.array([], dtype=np.int8)
        rate = float(np.mean(split_y)) if idx.size else float('nan')
        rows.append({
            'split': split_name,
            'n': int(idx.size),
            'size_error': abs(idx.size / total - ratios[split_id]) if total else float('nan'),
            'anomaly_proportion_difference': abs(rate - p) if idx.size else float('nan'),
            'contiguous_count': len(intervals_by_split[split_id]),
            'both_class': bool(np.unique(split_y).size == 2),
        })
    return rows


def anomaly_segment_crossing_rate(y: np.ndarray, intervals_by_split) -> float:
    episodes = contiguous_episodes(y)
    if not episodes:
        return float('nan')
    split_indices = [intervals_to_indices(intervals) for intervals in intervals_by_split]
    crossing = 0
    for lo, hi in episodes:
        touched = 0
        for idx in split_indices:
            if idx.size and np.any((idx >= lo) & (idx <= hi)):
                touched += 1
        crossing += int(touched > 1)
    return crossing / len(episodes)
