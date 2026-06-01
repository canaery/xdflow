import numpy as np
import pytest
import xarray as xr

from xdflow.core import DataContainer

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


def test_multitaper_preserves_existing_history():
    rng = np.random.default_rng(0)
    data = xr.DataArray(
        rng.normal(size=(2, 2, 200)),
        dims=("trial", "channel", "time"),
        coords={
            "trial": [0, 1],
            "channel": ["ch0", "ch1"],
            "time": np.arange(200) / 100,
            "stimulus": ("trial", ["A", "B"]),
        },
        attrs={
            "data_history": [{"class": "PreviousTransform", "params": {}}],
            "sampling_frequency_hz": 100,
        },
    )
    container = DataContainer(data)
    transform = MultiTaperTransform(
        fs=100,
        n_fft=128,
        avg_over_time_windows=True,
        avg_within_freq_bands=True,
        freq_ranges={"theta": (4.0, 8.0), "beta": (12.0, 30.0)},
    )

    result = transform.transform(container)

    assert result.data.attrs["sampling_frequency_hz"] == 100
    assert [entry["class"] for entry in result.data.attrs["data_history"]] == [
        "PreviousTransform",
        "MultiTaperTransform",
    ]
    assert [entry["class"] for entry in container.data.attrs["data_history"]] == ["PreviousTransform"]
