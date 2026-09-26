import numpy as np
import pytest

from evt_demo.data_generator import generate_fractal_extreme_series, generate_kuramoto_time_series


def test_fractal_series_is_reproducible_and_contains_an_event():
    first, first_mask = generate_fractal_extreme_series(200, 0.75, 7)
    second, second_mask = generate_fractal_extreme_series(200, 0.75, 7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first.shape == first_mask.shape == (200,)
    assert first_mask.sum() == 10


def test_kuramoto_shape_range_and_reproducibility():
    first = generate_kuramoto_time_series(100, 5, 2, 11)
    second = generate_kuramoto_time_series(100, 5, 2, 11)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.values.shape == (100, 5)
    assert np.all((0 <= first.coherence) & (first.coherence <= 1))


@pytest.mark.parametrize("channel", [-1, 3])
def test_kuramoto_rejects_invalid_source_channel(channel):
    with pytest.raises(ValueError, match="source_channel"):
        generate_kuramoto_time_series(n_channels=3, source_channel=channel)
