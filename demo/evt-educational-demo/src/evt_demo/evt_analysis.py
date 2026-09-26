"""GEV Block Maxima, GPD POT и диагностические данные."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import genextreme, genpareto

@dataclass(frozen=True)
class GEVFit:
    mu: float; sigma: float; xi: float
    block_maxima: np.ndarray; block_bounds: np.ndarray; dropped_points: int

@dataclass(frozen=True)
class GPDFit:
    threshold: float; sigma_u: float; xi: float
    exceedances: np.ndarray; indices: np.ndarray; exceedance_rate: float


def _array(data: ArrayLike) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("Ожидается непустой одномерный массив конечных чисел")
    return x

class EVTAnalyzer:
    """Оценивание классических одномерных EVT-моделей."""
    def fit_gev_block_maxima(self, data: ArrayLike, block_size: int = 100) -> GEVFit:
        x = _array(data)
        if block_size < 2: raise ValueError("block_size должен быть не меньше 2")
        n_blocks = len(x) // block_size
        if n_blocks < 10: raise ValueError("Для GEV требуется минимум 10 полных блоков")
        used = n_blocks * block_size
        maxima = x[:used].reshape(n_blocks, block_size).max(axis=1)
        c, loc, scale = genextreme.fit(maxima)
        if scale <= 0 or not np.all(np.isfinite([c, loc, scale])): raise RuntimeError("Не удалось оценить GEV")
        bounds = np.column_stack((np.arange(n_blocks) * block_size, (np.arange(n_blocks) + 1) * block_size))
        return GEVFit(float(loc), float(scale), float(-c), maxima, bounds, len(x) - used)

    def fit_gpd_peaks_over_threshold(self, data: ArrayLike, threshold: float | None = None,
                                     threshold_percentile: float = 95.0) -> GPDFit:
        x = _array(data)
        if not 0 < threshold_percentile < 100: raise ValueError("Перцентиль должен лежать между 0 и 100")
        u = float(np.percentile(x, threshold_percentile) if threshold is None else threshold)
        indices = np.flatnonzero(x > u); exceedances = x[indices] - u
        if len(exceedances) < 20: raise ValueError("Нужно минимум 20 превышений: снизьте порог или увеличьте выборку")
        xi, _, scale = genpareto.fit(exceedances, floc=0)
        if scale <= 0 or not np.all(np.isfinite([xi, scale])): raise RuntimeError("Не удалось оценить GPD")
        return GPDFit(u, float(scale), float(xi), exceedances, indices, len(indices) / len(x))

    def calculate_gev_return_levels(self, fit: GEVFit, return_periods: Sequence[float] = (10, 50, 100, 500),
                                    *, confidence_level: float = .95, n_bootstrap: int = 200,
                                    seed: int = 42) -> pd.DataFrame:
        periods = np.asarray(return_periods, dtype=float)
        if np.any(periods <= 1) or not 0 < confidence_level < 1 or n_bootstrap < 1:
            raise ValueError("T > 1, confidence_level в (0,1), n_bootstrap >= 1")
        def levels(mu: float, sigma: float, xi: float) -> np.ndarray:
            q = -np.log1p(-1 / periods)
            return mu - sigma * np.log(q) if abs(xi) < 1e-6 else mu + sigma / xi * (q ** (-xi) - 1)
        estimate = levels(fit.mu, fit.sigma, fit.xi)
        rng = np.random.default_rng(seed); samples = []
        c = -fit.xi
        for _ in range(n_bootstrap):
            sample = genextreme.rvs(c, loc=fit.mu, scale=fit.sigma, size=len(fit.block_maxima), random_state=rng)
            try:
                cb, mb, sb = genextreme.fit(sample)
                if sb > 0 and np.all(np.isfinite([cb, mb, sb])): samples.append(levels(mb, sb, -cb))
            except (ValueError, RuntimeError, FloatingPointError):
                continue
        if len(samples) < .8 * n_bootstrap: raise RuntimeError("Успешно менее 80% bootstrap-реплик")
        alpha = (1 - confidence_level) / 2
        low, high = np.quantile(np.asarray(samples), [alpha, 1 - alpha], axis=0)
        return pd.DataFrame({"return_period": periods, "return_level": estimate,
                             "ci_lower": low, "ci_upper": high})

    def detect_extremes_via_pot(self, data: ArrayLike, fit: GPDFit,
                                tail_probability: float = .01) -> np.ndarray:
        x = _array(data)
        if not 0 < tail_probability < 1: raise ValueError("tail_probability должен лежать в (0,1)")
        mask = x > fit.threshold; result = np.zeros(len(x), dtype=bool)
        result[mask] = genpareto.sf(x[mask] - fit.threshold, fit.xi, loc=0, scale=fit.sigma_u) <= tail_probability
        return result

    def diagnostics(self, fit: GEVFit | GPDFit) -> dict[str, np.ndarray]:
        sample = fit.block_maxima if isinstance(fit, GEVFit) else fit.exceedances
        probs = (np.arange(1, len(sample) + 1) - .5) / len(sample)
        empirical = np.sort(sample)
        grid = np.linspace(float(empirical.min()), float(empirical.max()), 200)
        if isinstance(fit, GEVFit):
            theoretical = genextreme.ppf(probs, -fit.xi, loc=fit.mu, scale=fit.sigma)
            fitted_cdf = genextreme.cdf(empirical, -fit.xi, loc=fit.mu, scale=fit.sigma)
            density = genextreme.pdf(grid, -fit.xi, loc=fit.mu, scale=fit.sigma)
            return_periods = np.geomspace(2, 500, 100)
            q = -np.log1p(-1 / return_periods)
            return_levels = (fit.mu - fit.sigma * np.log(q) if abs(fit.xi) < 1e-6
                             else fit.mu + fit.sigma / fit.xi * (q ** (-fit.xi) - 1))
        else:
            theoretical = genpareto.ppf(probs, fit.xi, loc=0, scale=fit.sigma_u)
            fitted_cdf = genpareto.cdf(empirical, fit.xi, loc=0, scale=fit.sigma_u)
            density = genpareto.pdf(grid, fit.xi, loc=0, scale=fit.sigma_u)
            return_periods = np.array([], dtype=float)
            return_levels = np.array([], dtype=float)
        return {"probabilities": probs, "empirical_quantiles": empirical,
                "theoretical_quantiles": theoretical, "fitted_cdf": fitted_cdf,
                "density_grid": grid, "fitted_density": density,
                "return_periods": return_periods, "return_levels": return_levels}
