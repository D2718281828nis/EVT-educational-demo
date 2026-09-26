import numpy as np
import pytest
from evt_demo.ar_model import AutoregressiveModel

def test_ar_shapes_and_state():
    x=np.random.default_rng(4).normal(size=500); model=AutoregressiveModel(2)
    with pytest.raises(RuntimeError): model.predict(x)
    model.fit(x); residuals=model.calculate_residuals(x)
    assert residuals.shape==x.shape and np.isnan(residuals[:2]).all()
    assert model.detect_anomalies_via_residuals(x).dtype==bool
    forecast,low,high=model.forecast_with_confidence(x,12); assert forecast.shape==low.shape==high.shape==(12,); assert np.all(low<=forecast) and np.all(forecast<=high)
