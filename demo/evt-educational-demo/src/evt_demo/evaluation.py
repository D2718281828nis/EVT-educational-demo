"""Comparable point-level quality metrics for EVT and an AR residual baseline."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .data_generator import GraphTimeSeries
from .detection import DetectionResult


@dataclass(frozen=True)
class DetectionComparison:
    names: tuple[str, str]
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    evt_mask: np.ndarray
    ar_mask: np.ndarray
    truth_mask: np.ndarray


def compare_evt_ar(
    series: GraphTimeSeries, detection: DetectionResult, baseline_end: int = 700
) -> DetectionComparison:
    """Compare point alarms against the known active interval of a synthetic event."""
    truth = np.zeros(len(series.values), dtype=bool)
    truth[
        series.event_onset : min(len(truth), int(series.arrival_times.max()) + 18)
    ] = True
    evt_mask = detection.indicator > detection.threshold
    indicator = detection.indicator
    phi = float(
        np.dot(indicator[1:baseline_end], indicator[: baseline_end - 1])
        / np.dot(indicator[: baseline_end - 1], indicator[: baseline_end - 1])
    )
    residual = indicator[1:] - phi * indicator[:-1]
    center = float(np.median(residual[: baseline_end - 1]))
    scale = float(1.4826 * np.median(np.abs(residual[: baseline_end - 1] - center)))
    ar_mask = np.zeros(len(indicator), dtype=bool)
    ar_mask[1:] = residual > center + 3 * max(scale, 1e-8)
    scores = [
        precision_recall_fscore_support(truth, mask, average="binary", zero_division=0)[
            :3
        ]
        for mask in (evt_mask, ar_mask)
    ]
    precision, recall, f1 = np.asarray(scores).T
    return DetectionComparison(
        ("EVT/POT", "AR(1) residual"), precision, recall, f1, evt_mask, ar_mask, truth
    )
