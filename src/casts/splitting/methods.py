from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

RATIOS = np.array([0.6, 0.2, 0.2], dtype=np.float64)
SPLIT_NAMES = ("train", "validation", "test")

@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    benchmark: str
    method: str
    seed: int
    replicate: int
    assign: np.ndarray
    intervals_by_split: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    bin_size: int
    lambda_cont: float

def bin_random_repair_empty(
    assign: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    fixed = assign.copy()
    for split_id in range(3):
        if np.any(fixed == split_id):
            continue
        donor = int(rng.integers(0, len(fixed)))
        fixed[donor] = split_id
    return fixed


def bin_random_initial_assignment(
    n_bins: int,
    rng: np.random.Generator,
    ratios: np.ndarray = RATIOS,
) -> np.ndarray:
    probabilities = ratios / ratios.sum()
    assign = rng.choice(3, size=n_bins, p=probabilities)
    return bin_random_repair_empty(assign, rng)


def bin_stratified_random_search(
    lengths: np.ndarray,
    positives: np.ndarray,
    seed: int,
    search_runs: int = 8,
    assignment_steps: int = 400,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_bins = len(lengths)
    best = None
    best_score = math.inf

    for _ in range(search_runs):
        current = bin_random_initial_assignment(n_bins, rng)
        current_score = objective(
            current,
            lengths,
            positives,
            RATIOS,
            lambda_cont=0.0,
        )

        tau = 0.25

        for _ in range(assignment_steps):
            bin_index = int(rng.integers(0, n_bins))
            proposal = current.copy()
            proposal[bin_index] = int(rng.integers(0, 3))
            proposal = bin_random_repair_empty(proposal, rng)

            proposal_score = objective(
                proposal,
                lengths,
                positives,
                RATIOS,
                lambda_cont=0.0,
            )

            delta = proposal_score - current_score
            if delta <= 0 or rng.random() < math.exp(
                -delta / max(tau, 1e-6)
            ):
                current = proposal
                current_score = proposal_score

            tau *= 0.999

        if current_score < best_score:
            best = current.copy()
            best_score = current_score

    if best is None:
        raise RuntimeError(
            "Bin-level stratified random search did not return an assignment"
        )

    return best

def choose_bin_size(n_rows: int) -> int:
    if n_rows < 2000:
        return max(4, min(16, n_rows))
    if n_rows < 10000:
        return 32
    return 128


def make_bins(n_rows: int, bin_size: int) -> list[tuple[int, int]]:
    out = []
    start = 0
    while start < n_rows:
        end = min(n_rows - 1, start + bin_size - 1)
        out.append((start, end))
        start = end + 1
    return out


def bin_stats(y: np.ndarray, bins: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([end - start + 1 for start, end in bins], dtype=np.int64)
    positives = np.array([int(y[start:end + 1].sum()) for start, end in bins], dtype=np.int64)
    return lengths, positives


def objective(assign: np.ndarray, lengths: np.ndarray, positives: np.ndarray, ratios: np.ndarray = RATIOS, lambda_cont: float = 0.05) -> float:
    total = float(lengths.sum())
    global_rate = float(positives.sum() / max(total, 1.0))
    counts = np.bincount(assign, weights=lengths, minlength=3)[:3].astype(np.float64, copy=False)
    pos_counts = np.bincount(assign, weights=positives, minlength=3)[:3].astype(np.float64, copy=False)
    l_size = float(np.abs(counts / total - ratios).sum())
    l_label = float(np.abs(pos_counts - global_rate * counts).sum() / total)
    r_cont = float(np.sum(assign[1:] != assign[:-1])) / max(len(assign) - 1, 1)
    return l_size + l_label + lambda_cont * r_cont


def repair_empty(assign: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fixed = assign.copy()
    for split_id in range(3):
        if np.any(fixed == split_id):
            continue
        counts = np.array([np.sum(fixed == item) for item in range(3)])
        donor_split = int(np.argmax(counts))
        donor_positions = np.flatnonzero(fixed == donor_split)
        donor = int(rng.choice(donor_positions)) if donor_positions.size else int(rng.integers(0, len(fixed)))
        fixed[donor] = split_id
    return fixed


def chronological_assignment(n_items: int) -> np.ndarray:
    assign = np.zeros(n_items, dtype=np.int64)
    cut1 = max(1, int(round(0.6 * n_items)))
    cut2 = max(cut1 + 1, int(round(0.8 * n_items)))
    cut2 = min(cut2, n_items)
    assign[cut1:cut2] = 1
    assign[cut2:] = 2
    if n_items >= 3 and not np.any(assign == 2):
        assign[-1] = 2
    return assign


def random_assignment(n_items: int, rng: np.random.Generator, ratios: np.ndarray = RATIOS) -> np.ndarray:
    assign = np.empty(n_items, dtype=np.int64)
    order = rng.permutation(n_items)
    cut1 = int(round(ratios[0] * n_items))
    cut2 = int(round((ratios[0] + ratios[1]) * n_items))
    cut1 = min(max(cut1, 1), max(n_items - 2, 1))
    cut2 = min(max(cut2, cut1 + 1), max(n_items - 1, cut1 + 1))
    assign[order[:cut1]] = 0
    assign[order[cut1:cut2]] = 1
    assign[order[cut2:]] = 2
    return repair_empty(assign, rng)


def local_search(lengths: np.ndarray, positives: np.ndarray, lambda_cont: float, seed: int, search_runs: int, assignment_steps: int, initialization: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_bins = len(lengths)
    best = None
    best_score = math.inf
    for run in range(search_runs):
        if initialization == "chronological" or (initialization == "mixed" and run % 2 == 0):
            current = repair_empty(chronological_assignment(n_bins), rng)
        else:
            current = random_assignment(n_bins, rng)
        current_score = objective(current, lengths, positives, RATIOS, lambda_cont)
        run_best = current.copy()
        run_best_score = current_score
        tau = 0.25
        for _ in range(assignment_steps):
            idx = int(rng.integers(0, n_bins))
            proposal = current.copy()
            proposal[idx] = int(rng.integers(0, 3))
            proposal = repair_empty(proposal, rng)
            score = objective(proposal, lengths, positives, RATIOS, lambda_cont)
            delta = score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / max(tau, 1e-6)):
                current, current_score = proposal, score
                if current_score < run_best_score:
                    run_best, run_best_score = current.copy(), current_score
            tau *= 0.999
        if run_best_score < best_score:
            best, best_score = run_best.copy(), run_best_score
    if best is None:
        raise RuntimeError("CASTS search failed to initialize")
    return best


def indices_to_intervals(indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    if indices.size == 0:
        return tuple()
    idx = np.sort(np.unique(indices.astype(np.int64, copy=False)))
    starts = [int(idx[0])]
    ends = []
    for prev, cur in zip(idx[:-1], idx[1:]):
        if int(cur) != int(prev) + 1:
            ends.append(int(prev))
            starts.append(int(cur))
    ends.append(int(idx[-1]))
    return tuple(zip(starts, ends))


def merge_bins(bins: list[tuple[int, int]], assign: np.ndarray, split_id: int) -> tuple[tuple[int, int], ...]:
    intervals = []
    for (start, end), label in zip(bins, assign):
        if int(label) != split_id:
            continue
        if intervals and start == intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
    return tuple(intervals)


def stratified_point_assignment(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assign = np.empty(len(y), dtype=np.int64)
    for label in sorted(np.unique(y).tolist()):
        idx = np.flatnonzero(y == label)
        if idx.size == 0:
            continue
        local = random_assignment(idx.size, rng) if idx.size >= 3 else rng.choice(3, size=idx.size, p=RATIOS / RATIOS.sum())
        assign[idx] = local
    return repair_empty(assign, rng) if len(assign) >= 3 else assign


def build_specs_for_labels(dataset: str, benchmark: str, y: np.ndarray, seeds: list[int], replicates: int, lambda_values: Iterable[float] = (0.05, 0.35)) -> list[SplitSpec]:
    n_rows = int(len(y))
    bin_size = choose_bin_size(n_rows)
    bins = make_bins(n_rows, bin_size)
    lengths, positives = bin_stats(y, bins)
    specs: list[SplitSpec] = []
    for seed in seeds:
        for replicate in range(replicates):
            chrono = chronological_assignment(len(bins))
            assignments = [
                ("Chronological 60/20/20", 0.0, chrono, bins, bin_size),
                ("Point-level stratified random", 0.0, stratified_point_assignment(y, seed + 7919 * replicate), None, 1),
                ("Bin-level stratified random", 0.0, bin_stratified_random_search(lengths, positives, seed=seed + 11 * replicate, search_runs=8, assignment_steps=400), bins, bin_size),
            ]
            for lam in lambda_values:
                method = "CASTS-05" if float(lam) == 0.05 else ("CASTS-35" if float(lam) == 0.35 else f"CASTS-lambda-{lam:g}")
                assignments.append((method, float(lam), local_search(lengths, positives, float(lam), seed + 211 + 11 * replicate, 10, 400, "mixed"), bins, bin_size))
            for method, lam, assign, method_bins, spec_bin_size in assignments:
                if method_bins is None:
                    intervals = tuple(indices_to_intervals(np.flatnonzero(assign == split_id)) for split_id in range(3))
                else:
                    intervals = tuple(merge_bins(method_bins, assign, split_id) for split_id in range(3))
                specs.append(SplitSpec(dataset, benchmark, method, seed, replicate, assign, intervals, spec_bin_size, float(lam)))
    return specs
