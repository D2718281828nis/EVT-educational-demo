"""Tests for reproducible educational dataset generation."""

import numpy as np
import pytest

from evt_demo.data_generator import (
    generate_fractal_extreme_series,
    generate_kuramoto_time_series,
)


def test_fractal_series_is_reproducible_and_has_one_event():
    first, first_mask = generate_fractal_extreme_series(seed=17)
    second, second_mask = generate_fractal_extreme_series(seed=17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first.shape == first_mask.shape == (1500,)
    assert first_mask.dtype == np.bool_
    assert first_mask.sum() == 25
    assert np.ptp(np.flatnonzero(first_mask)) == 24


def test_kuramoto_shapes_and_reproducibility():
    first = generate_kuramoto_time_series(n_points=300, n_channels=5, seed=9)
    second = generate_kuramoto_time_series(n_points=300, n_channels=5, seed=9)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.coherence, second.coherence)
    assert first.values.shape == first.phases.shape == (300, 5)
    assert first.coherence.shape == (300,)
    assert np.all((0 <= first.coherence) & (first.coherence <= 1))


@pytest.mark.parametrize("kwargs", [{"n_points": 100}, {"hurst_exponent": 1.0}])
def test_fractal_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        generate_fractal_extreme_series(**kwargs)
