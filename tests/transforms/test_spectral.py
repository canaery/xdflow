import pytest

try:
    from xdflow.transforms.spectral import MultiTaperTransform
except ImportError as exc:
    if exc.name != "spectral_connectivity":
        raise
    pytestmark = pytest.mark.skip(reason="spectral-connectivity is not installed")


FREQ_RANGES = {
    "theta": (4.0, 8.0),
    "beta": (12.0, 30.0),
}


def test_multitaper_avg_within_freq_bands_expected_dims():
    transform = MultiTaperTransform(
        fs=500,
        avg_over_time_windows=True,
        avg_within_freq_bands=True,
        freq_ranges=FREQ_RANGES,
    )

    assert transform.avg_within_freq_bands is True
    assert transform.get_expected_output_dims(("trial", "channel", "time")) == ("trial", "channel", "freq_band")


def test_multitaper_clone_preserves_avg_within_freq_bands():
    transform = MultiTaperTransform(
        fs=500,
        avg_over_time_windows=True,
        avg_within_freq_bands=True,
        freq_ranges=FREQ_RANGES,
    )

    cloned = transform.clone()

    assert cloned.avg_within_freq_bands is True
    assert cloned.freq_ranges == FREQ_RANGES
