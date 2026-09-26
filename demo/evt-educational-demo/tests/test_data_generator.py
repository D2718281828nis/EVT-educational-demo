from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from evt_demo.data_generator import generate_time_series_with_extreme_events, save_to_csv

def test_reproducible_and_mask(tmp_path: Path):
    a,ma=generate_time_series_with_extreme_events(seed=7); b,mb=generate_time_series_with_extreme_events(seed=7)
    assert np.array_equal(a,b) and np.array_equal(ma,mb); assert ma.sum()==9
    path=tmp_path/"x.csv"; save_to_csv(a,ma,path); frame=pd.read_csv(path)
    assert list(frame)==["timestamp","value","is_extreme"]; assert frame.is_extreme.sum()==9

def test_invalid_parameters():
    for kwargs in ({"n_points":99},{"phi":1.0},{"sigma":0},{"extreme_duration":0}):
        with pytest.raises(ValueError): generate_time_series_with_extreme_events(**kwargs)
