"""Synthetic LFP data generator for tutorials and testing."""

import numpy as np
import xarray as xr


def make_synthetic_lfp(
    n_trials: int = 1000,
    n_channels: int = 128,
    fs: float = 500.0,
    time_before_ms: float = 200.0,
    time_segment_len_ms: float = 2200.0,
    stimuli_config: dict | None = None,
    sessions: list[str] | None = None,
    animals: list[int] | None = None,
    trial_amplitude_sigma: float = 0.3,
    channel_gain_sigma: float = 0.2,
    white_noise_std: float = 0.05,
    seed: int = 42,
) -> xr.DataArray:
    """Generate synthetic trial-aligned LFP data with class-separable spectral features.

    Creates realistic 1/f-spectrum LFP signals with per-stimulus frequency band
    power boosts, making the data separable by standard classifiers using
    bandpower or covariance-based features.

    Parameters
    ----------
    n_trials : int
        Total number of trials (distributed evenly across stimuli).
    n_channels : int
        Number of recording channels.
    fs : float
        Sampling frequency in Hz.
    time_before_ms : float
        Pre-event time in milliseconds.
    time_segment_len_ms : float
        Total segment length in milliseconds.
    stimuli_config : dict, optional
        Mapping of stimulus name to band-boost dict.
        Each band-boost dict maps ``(f_low, f_high)`` tuples to power multipliers.
        Defaults to three odor classes with distinct spectral signatures::

            {
                "odorA": {(4, 8): 3.0, (30, 45): 1.5},    # theta + mild gamma
                "odorB": {(12, 30): 3.0, (8, 12): 2.0},   # beta + alpha
                "odorC": {(60, 90): 2.5, (4, 8): 1.2},    # high gamma + mild theta
            }

    sessions : list of str, optional
        Session labels to sample from. Defaults to ["session_1", "session_2", "session_3"].
    animals : list of int, optional
        Animal IDs to sample from. Defaults to [1, 2, 3, 4].
    trial_amplitude_sigma : float
        Lognormal sigma for trial-to-trial amplitude variability.
    channel_gain_sigma : float
        Lognormal sigma for channel-to-channel gain variability.
    white_noise_std : float
        Additive white noise standard deviation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xr.DataArray
        Shape ``(n_trials, n_channels, n_times)`` with dimensions
        ``["trial", "channel", "time"]`` and coordinate variables
        ``session``, ``stimulus``, ``animal`` on the trial dimension.

    Examples
    --------
    >>> da = make_synthetic_lfp(n_trials=300, n_channels=32)
    >>> da.shape
    (300, 32, 1100)

    >>> # Custom stimuli with two classes
    >>> da = make_synthetic_lfp(
    ...     stimuli_config={
    ...         "target": {(4, 8): 4.0},
    ...         "nontarget": {(20, 30): 2.0},
    ...     }
    ... )
    """
    rng = np.random.default_rng(seed)

    if stimuli_config is None:
        stimuli_config = {
            "odorA": {(4, 8): 3.0, (30, 45): 1.5},
            "odorB": {(12, 30): 3.0, (8, 12): 2.0},
            "odorC": {(60, 90): 2.5, (4, 8): 1.2},
        }
    if sessions is None:
        sessions = ["session_1", "session_2", "session_3"]
    if animals is None:
        animals = [1, 2, 3, 4]

    # Time axis
    dt = 1000.0 / fs
    time = np.arange(-time_before_ms, time_segment_len_ms - time_before_ms, dt)
    n_times = len(time)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)

    # Trial metadata
    stim_names = list(stimuli_config.keys())
    trial_ids = np.arange(1, n_trials + 1)
    trial_stimuli = rng.choice(stim_names, size=n_trials)
    trial_sessions = rng.choice(sessions, size=n_trials)
    trial_animals = rng.choice(animals, size=n_trials)
    channel_labels = [str(i) for i in range(n_channels)]

    # --- Generate signals ---
    data = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)

    for stim_name, band_boosts in stimuli_config.items():
        mask = trial_stimuli == stim_name
        n_stim = mask.sum()
        if n_stim == 0:
            continue

        # 1/f amplitude spectrum
        safe_freqs = np.where(freqs > 0, freqs, 1.0)
        amp_spectrum = np.where(freqs > 0, 1.0 / np.sqrt(safe_freqs), 0.0)

        # Apply band boosts
        for (f_lo, f_hi), boost in band_boosts.items():
            band_mask = (freqs >= f_lo) & (freqs <= f_hi)
            amp_spectrum[band_mask] *= boost

        # Generate trials for this stimulus
        trial_indices = np.where(mask)[0]
        for idx in trial_indices:
            for ch in range(n_channels):
                phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
                spectrum = amp_spectrum * np.exp(1j * phases)
                signal = np.fft.irfft(spectrum, n=n_times).astype(np.float32)
                signal += rng.normal(0, white_noise_std, size=n_times).astype(np.float32)
                data[idx, ch, :] = signal

    # Trial-to-trial and channel-to-channel variability
    if trial_amplitude_sigma > 0:
        trial_scales = rng.lognormal(0, trial_amplitude_sigma, size=(n_trials, 1, 1))
        data *= trial_scales.astype(np.float32)
    if channel_gain_sigma > 0:
        ch_gains = rng.lognormal(0, channel_gain_sigma, size=(1, n_channels, 1))
        data *= ch_gains.astype(np.float32)

    return xr.DataArray(
        data,
        dims=["trial", "channel", "time"],
        coords={
            "trial": trial_ids,
            "session": ("trial", trial_sessions),
            "stimulus": ("trial", trial_stimuli),
            "animal": ("trial", trial_animals),
            "channel": channel_labels,
            "time": time,
        },
        attrs={
            "description": ("Synthetic trial-aligned LFP with class-separable spectral features."),
            "time_before_ms": time_before_ms,
            "time_segment_len_ms": time_segment_len_ms,
            "sampling_frequency_hz": fs,
        },
    )


if __name__ == "__main__":
    da = make_synthetic_lfp()
    print(da)
    print(f"\nShape: {da.shape}, Size: {da.size:,} values")
    print(f"Stimuli: {np.unique(da.stimulus.values, return_counts=True)}")
