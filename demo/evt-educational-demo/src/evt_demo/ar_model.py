"""Авторегрессионная baseline-модель для учебного сравнения."""
from __future__ import annotations
from statistics import NormalDist
from typing import Any
import importlib
import importlib.util
import numpy as np
from numpy.typing import ArrayLike

class AutoregressiveModel:
    def __init__(self, order: int = 1):
        if order < 1: raise ValueError("order должен быть положительным")
        self.order = order; self.result: Any | None = None
        self.coefficients: np.ndarray | None = None; self.intercept: float | None = None
        self.residual_std: float | None = None

    @staticmethod
    def _data(data: ArrayLike) -> np.ndarray:
        x = np.asarray(data, dtype=float)
        if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
            raise ValueError("Ожидается одномерный ряд конечных чисел")
        return x

    def fit(self, data: ArrayLike) -> "AutoregressiveModel":
        x = self._data(data)
        if len(x) <= 2 * self.order + 1: raise ValueError("Ряд слишком короткий для выбранного порядка")
        if importlib.util.find_spec("statsmodels") is not None:
            module = importlib.import_module("statsmodels.tsa.ar_model")
            self.result = module.AutoReg(x, lags=self.order, trend="c", old_names=False).fit()
            params = np.asarray(self.result.params)
            fitted_residuals = np.asarray(self.result.resid)
        else:  # минимальный офлайн fallback; при установленных зависимостях используется AutoReg
            design = np.column_stack([np.ones(len(x) - self.order)] +
                                     [x[self.order-lag:len(x)-lag] for lag in range(1, self.order + 1)])
            target = x[self.order:]
            params = np.linalg.lstsq(design, target, rcond=None)[0]
            fitted_residuals = target - design @ params
            self.result = {"fallback": "numpy-lstsq"}
        self.intercept = float(params[0]); self.coefficients = params[1:]
        self.residual_std = float(np.std(fitted_residuals, ddof=len(params)))
        return self

    def _fitted(self) -> Any:
        if self.result is None: raise RuntimeError("Сначала вызовите fit")
        return self.result

    def predict(self, data: ArrayLike, steps: int = 1) -> np.ndarray:
        x = self._data(data); self._fitted()
        if steps < 1: raise ValueError("steps должен быть положительным")
        history = list(x); out = []
        assert self.coefficients is not None and self.intercept is not None
        for _ in range(steps):
            value = self.intercept + float(np.dot(self.coefficients, history[-self.order:][::-1]))
            out.append(value); history.append(value)
        return np.asarray(out)

    def calculate_residuals(self, data: ArrayLike) -> np.ndarray:
        x = self._data(data); self._fitted(); residuals = np.full(len(x), np.nan)
        assert self.coefficients is not None and self.intercept is not None
        for t in range(self.order, len(x)):
            residuals[t] = x[t] - self.intercept - np.dot(self.coefficients, x[t-self.order:t][::-1])
        return residuals

    def detect_anomalies_via_residuals(self, data: ArrayLike, threshold_std: float = 3.0) -> np.ndarray:
        if threshold_std <= 0: raise ValueError("threshold_std должен быть положительным")
        residuals = self.calculate_residuals(data); result = np.zeros(len(residuals), dtype=bool)
        finite = np.isfinite(residuals); scale = float(np.std(residuals[finite], ddof=1))
        result[finite] = np.abs(residuals[finite] - np.mean(residuals[finite])) > threshold_std * scale
        return result

    def forecast_with_confidence(self, data: ArrayLike, steps: int = 100,
                                 confidence_level: float = .95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 < confidence_level < 1: raise ValueError("confidence_level должен лежать в (0,1)")
        forecast = self.predict(data, steps); assert self.residual_std is not None
        z = NormalDist().inv_cdf((1 + confidence_level) / 2)
        width = z * self.residual_std * np.sqrt(np.arange(1, steps + 1))
        return forecast, forecast - width, forecast + width
