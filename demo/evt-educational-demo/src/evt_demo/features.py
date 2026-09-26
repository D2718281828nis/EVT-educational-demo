"""Node features linking moment detection to source localization."""

from __future__ import annotations
import numpy as np


def node_features(
    values: np.ndarray,
    adjacency: np.ndarray,
    detected_at: int,
    baseline_end: int,
    window: int = 50,
) -> np.ndarray:
    """Build causal node features using data up to a short post-detection window."""
    x = np.asarray(values, float)
    base = x[:baseline_end]
    med = np.median(base, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(base - med), axis=0), 1e-8)
    stop = min(len(x), detected_at + window)
    segment = (x[detected_at:stop] - med) / mad
    peak = np.max(np.abs(segment), axis=0)
    energy = np.mean(segment**2, axis=0)
    slope = np.mean(np.diff(segment, axis=0), axis=0)
    latency = np.full(x.shape[1], window, float)
    for node in range(x.shape[1]):
        hits = np.flatnonzero(np.abs(segment[:, node]) > 3)
        latency[node] = hits[0] if len(hits) else window
    degree = adjacency.sum(1)
    neighbour_peak = adjacency @ peak / np.maximum(degree, 1)
    return np.column_stack(
        [peak, energy, slope, latency / window, degree, neighbour_peak]
    ).astype(np.float32)


def edge_index(adjacency: np.ndarray) -> np.ndarray:
    return np.vstack(np.nonzero(adjacency)).astype(np.int64)
