import numpy as np
import pytest

from evt_demo.data_generator import (
    generate_fractal_extreme_series,
    generate_kuramoto_propagation_recording,
    generate_kuramoto_propagation_series,
    generate_kuramoto_time_series,
)


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


def test_propagation_series_is_reproducible():
    first = generate_kuramoto_propagation_series(300, 6, 2, 11)
    second = generate_kuramoto_propagation_series(300, 6, 2, 11)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.event_mask, second.event_mask)
    assert first.event_onset == second.event_onset


def test_propagation_series_marks_the_source_channel_and_ground_truth_shape():
    n_channels = 6
    series = generate_kuramoto_propagation_series(300, n_channels, 2, 11)
    assert series.source_channel == 2
    assert series.values.shape == (300, n_channels)
    assert series.event_mask.shape == (300,)
    assert series.channel_event_mask.shape == (300, n_channels)
    # The source channel is its own hop-0 neighbor: zero propagation delay.
    assert series.channel_arrival_delay[series.source_channel] == 0
    assert series.channel_arrival_delay.min() == 0
    assert (series.channel_arrival_delay >= 0).all()
    # Every channel carries some part of the injected event.
    assert series.channel_event_mask.any(axis=0).all()
    # The global mask is the union of all per-channel windows.
    np.testing.assert_array_equal(series.event_mask, series.channel_event_mask.any(axis=1))


def test_propagation_series_amplitude_decays_with_distance_from_source():
    series = generate_kuramoto_propagation_series(
        300, 6, source_channel=0, seed=11, event_magnitude_in_sigma=8.0, decay_per_hop=0.5,
    )
    injected = np.zeros(6)
    for channel in range(6):
        start = series.event_onset + int(series.channel_arrival_delay[channel])
        window = slice(start, start + series.event_duration)
        injected[channel] = np.abs(series.values[window, channel]).max()
    # Ring distance from channel 0 in a 6-node ring: 0,1,2,3,2,1 -- so the
    # amplitude at the farthest channel (index 3) should not exceed that of
    # a near neighbor (index 1).
    assert injected[3] <= injected[1]


def test_propagation_series_rejects_invalid_source_channel():
    with pytest.raises(ValueError, match="source_channel"):
        generate_kuramoto_propagation_series(n_channels=3, source_channel=5)


def test_propagation_recording_is_reproducible():
    first = generate_kuramoto_propagation_recording(n_events=3, n_channels=6, seed=5, baseline_size=100, gap=80)
    second = generate_kuramoto_propagation_recording(n_events=3, n_channels=6, seed=5, baseline_size=100, gap=80)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.event_times == second.event_times
    assert first.sources == second.sources


def test_propagation_recording_has_well_separated_events_after_baseline():
    n_events, baseline_size, gap = 4, 100, 80
    recording = generate_kuramoto_propagation_recording(
        n_events=n_events, n_channels=6, seed=5, baseline_size=baseline_size, gap=gap,
    )
    assert len(recording.event_times) == n_events
    assert len(recording.sources) == n_events
    assert all(0 <= source < 6 for source in recording.sources)
    assert all(time > baseline_size for time in recording.event_times)
    # Consecutive onsets are evenly spaced by `gap`, mirroring make_recording.
    spacing = np.diff(recording.event_times)
    assert (spacing == gap).all()
    assert recording.values.shape == (baseline_size + gap * n_events + gap // 2, 6)
    assert len(recording.channel_arrival_delays) == n_events


def test_propagation_recording_as_input_matches_values_and_names():
    recording = generate_kuramoto_propagation_recording(n_events=2, n_channels=4, seed=1, baseline_size=60, gap=60)
    wrapped = recording.as_input()
    np.testing.assert_array_equal(np.asarray(wrapped.values), recording.values)
    assert wrapped.channel_names == recording.channel_names == ("channel_00", "channel_01", "channel_02", "channel_03")


def test_propagation_recording_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="n_events"):
        generate_kuramoto_propagation_recording(n_events=0)
    with pytest.raises(ValueError, match="gap"):
        generate_kuramoto_propagation_recording(gap=10)
    with pytest.raises(ValueError, match="n_channels"):
        generate_kuramoto_propagation_recording(n_channels=1)
