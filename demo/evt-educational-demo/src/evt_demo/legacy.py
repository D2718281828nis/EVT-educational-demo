"""Backward-compatible APIs used by the first generation of demo notebooks."""

from __future__ import annotations
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import genextreme, genpareto


def _array(data: ArrayLike) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a non-empty one-dimensional finite array")
    return values


@dataclass(frozen=True)
class GEVFit:
    mu: float
    sigma: float
    xi: float
    block_maxima: np.ndarray
    block_bounds: np.ndarray
    dropped_points: int


@dataclass(frozen=True)
class GPDFit:
    threshold: float
    sigma_u: float
    xi: float
    exceedances: np.ndarray
    indices: np.ndarray
    exceedance_rate: float


class EVTAnalyzer:
    """Compatibility façade for Block Maxima and POT tutorial notebooks."""

    def fit_gev_block_maxima(self, data: ArrayLike, block_size: int = 100) -> GEVFit:
        values = _array(data)
        n_blocks = len(values) // block_size
        if block_size < 2 or n_blocks < 10:
            raise ValueError("Need block_size>=2 and at least ten complete blocks")
        used = n_blocks * block_size
        maxima = values[:used].reshape(n_blocks, block_size).max(axis=1)
        shape, location, scale = genextreme.fit(maxima)
        bounds = np.column_stack(
            (np.arange(n_blocks) * block_size, (np.arange(n_blocks) + 1) * block_size)
        )
        return GEVFit(
            float(location),
            float(scale),
            float(-shape),
            maxima,
            bounds,
            len(values) - used,
        )

    def fit_gpd_peaks_over_threshold(
        self,
        data: ArrayLike,
        threshold: float | None = None,
        threshold_percentile: float = 95.0,
    ) -> GPDFit:
        values = _array(data)
        level = float(
            np.percentile(values, threshold_percentile)
            if threshold is None
            else threshold
        )
        indices = np.flatnonzero(values > level)
        excess = values[indices] - level
        if len(excess) < 20:
            raise ValueError("Need at least twenty exceedances")
        shape, _, scale = genpareto.fit(excess, floc=0)
        return GPDFit(
            level,
            float(scale),
            float(shape),
            excess,
            indices,
            len(indices) / len(values),
        )

    def calculate_gev_return_levels(
        self,
        fit: GEVFit,
        return_periods: Sequence[float] = (10, 50, 100, 500),
        **_: object,
    ) -> pd.DataFrame:
        periods = np.asarray(return_periods, dtype=float)
        q = -np.log1p(-1 / periods)
        levels = (
            fit.mu - fit.sigma * np.log(q)
            if abs(fit.xi) < 1e-6
            else fit.mu + fit.sigma / fit.xi * (q ** (-fit.xi) - 1)
        )
        return pd.DataFrame({"return_period": periods, "return_level": levels})

    calculate_return_levels = calculate_gev_return_levels

    def detect_extremes_via_pot(
        self,
        data: ArrayLike,
        fit: GPDFit,
        tail_probability: float = 0.01,
    ) -> np.ndarray:
        values = _array(data)
        selected = values > fit.threshold
        result = np.zeros(len(values), dtype=bool)
        result[selected] = (
            genpareto.sf(
                values[selected] - fit.threshold, fit.xi, loc=0, scale=fit.sigma_u
            )
            <= tail_probability
        )
        return result

    def detect_extremes_via_evt(
        self,
        data: ArrayLike,
        threshold_percentile: float = 99.0,
    ) -> np.ndarray:
        values = _array(data)
        return values > np.percentile(values, threshold_percentile)

    def diagnostics(self, fit: GEVFit | GPDFit) -> dict[str, np.ndarray]:
        sample = fit.block_maxima if isinstance(fit, GEVFit) else fit.exceedances
        probabilities = (np.arange(1, len(sample) + 1) - 0.5) / len(sample)
        empirical = np.sort(sample)
        theoretical = (
            genextreme.ppf(probabilities, -fit.xi, loc=fit.mu, scale=fit.sigma)
            if isinstance(fit, GEVFit)
            else genpareto.ppf(probabilities, fit.xi, loc=0, scale=fit.sigma_u)
        )
        return {
            "probabilities": probabilities,
            "empirical_quantiles": empirical,
            "theoretical_quantiles": theoretical,
        }

    def diagnostic_plots(
        self, data: ArrayLike, method: str = "gev"
    ) -> dict[str, np.ndarray]:
        fit = (
            self.fit_gev_block_maxima(data)
            if method == "gev"
            else self.fit_gpd_peaks_over_threshold(data)
        )
        return self.diagnostics(fit)


class AutoregressiveModel:
    """Small least-squares AR(p) compatibility baseline."""

    def __init__(self, order: int = 1):
        if order < 1:
            raise ValueError("order must be positive")
        self.order = order
        self.coefficients: np.ndarray | None = None
        self.intercept: float | None = None
        self.residual_std: float | None = None

    def fit(self, data: ArrayLike) -> "AutoregressiveModel":
        values = _array(data)
        design = np.column_stack(
            [np.ones(len(values) - self.order)]
            + [
                values[self.order - lag : len(values) - lag]
                for lag in range(1, self.order + 1)
            ]
        )
        target = values[self.order :]
        params = np.linalg.lstsq(design, target, rcond=None)[0]
        self.intercept = float(params[0])
        self.coefficients = params[1:]
        self.residual_std = float(np.std(target - design @ params, ddof=len(params)))
        return self

    def _check(self) -> None:
        if self.coefficients is None or self.intercept is None:
            raise RuntimeError("Call fit first")

    def calculate_residuals(self, data: ArrayLike) -> np.ndarray:
        self._check()
        values = _array(data)
        residuals = np.full(len(values), np.nan)
        for time in range(self.order, len(values)):
            residuals[time] = (
                values[time]
                - self.intercept
                - np.dot(self.coefficients, values[time - self.order : time][::-1])
            )
        return residuals

    def detect_anomalies_via_residuals(
        self,
        data: ArrayLike,
        threshold_std: float = 3.0,
    ) -> np.ndarray:
        residuals = self.calculate_residuals(data)
        finite = np.isfinite(residuals)
        result = np.zeros(len(residuals), dtype=bool)
        result[finite] = np.abs(residuals[finite]) > threshold_std * np.std(
            residuals[finite]
        )
        return result

    def predict(self, data: ArrayLike, steps: int = 1) -> np.ndarray:
        self._check()
        history = list(_array(data))
        forecast = []
        for _ in range(steps):
            value = self.intercept + np.dot(
                self.coefficients, history[-self.order :][::-1]
            )
            forecast.append(value)
            history.append(float(value))
        return np.asarray(forecast)

    def forecast_with_confidence(
        self,
        data: ArrayLike,
        steps: int = 100,
        confidence_level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forecast = self.predict(data, steps)
        z = NormalDist().inv_cdf((1 + confidence_level) / 2)
        width = z * self.residual_std * np.sqrt(np.arange(1, steps + 1))
        return forecast, forecast - width, forecast + width
