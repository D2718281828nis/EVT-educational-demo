"""Component 1: EVT detection of the event moment in a multichannel series."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True)
class DetectionResult:
    indicator: np.ndarray
    threshold: float
    detected_at: int | None
    xi: float
    scale: float


class EVTEventDetector:
    """Fit a GPD to baseline exceedances of a robust graph-level indicator."""

    def __init__(
        self,
        baseline_end: int,
        threshold_percentile: float = 90,
        false_alarm_probability: float = 1e-3,
        persistence: int = 2,
    ):
        self.baseline_end = baseline_end
        self.threshold_percentile = threshold_percentile
        self.false_alarm_probability = false_alarm_probability
        self.persistence = persistence

    def fit_detect(self, values: np.ndarray) -> DetectionResult:
        x = np.asarray(values, float)
        if x.ndim != 2 or not 20 <= self.baseline_end < len(x):
            raise ValueError("Invalid series or baseline")
        baseline = x[: self.baseline_end]
        med = np.median(baseline, axis=0)
        mad = 1.4826 * np.median(np.abs(baseline - med), axis=0)
        mad = np.maximum(mad, 1e-8)
        z = np.abs((x - med) / mad)
        k = max(1, x.shape[1] // 5)
        indicator = np.mean(np.partition(z, -k, axis=1)[:, -k:], axis=1)
        base_indicator = indicator[: self.baseline_end]
        u = float(np.percentile(base_indicator, self.threshold_percentile))
        excess = base_indicator[base_indicator > u] - u
        if len(excess) < 10:
            raise ValueError("Too few baseline exceedances")
        xi, _, scale = genpareto.fit(excess, floc=0)
        threshold = float(
            u + genpareto.ppf(1 - self.false_alarm_probability, xi, loc=0, scale=scale)
        )
        above = indicator > threshold
        run = 0
        detected = None
        for time in range(self.baseline_end, len(above)):
            run = run + 1 if above[time] else 0
            if run == self.persistence:
                detected = time - self.persistence + 1
                break
        return DetectionResult(indicator, threshold, detected, float(xi), float(scale))
