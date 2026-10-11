"""Shared pytest fixtures including a spec-compliant binary EDF generator for testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pad_ascii(value: str, width: int) -> bytes:
    """Format an ASCII header field padded or truncated to exact byte width."""
    encoded = value.encode("ascii", errors="replace")[:width]
    return encoded.ljust(width, b" ")


def write_synthetic_edf(
    file_path: Path,
    channel_names: list[str],
    signals: np.ndarray,
    sfreq: int = 128,
    patient_id: str = "X X X X",
    recording_id: str = "Startdate X X X X",
    physical_dims: list[str] | None = None,
) -> Path:
    """Write a spec-compliant 16-bit binary EDF file for unit testing.

    Parameters
    ----------
    file_path : Path
        Destination file path.
    channel_names : list[str]
        List of channel names (length ``n_channels``).
    signals : np.ndarray
        2D float array of shape ``(n_channels, n_samples)``, where ``n_samples`` is a
        multiple of ``sfreq`` (1-second data records).
    sfreq : int
        Sampling rate in Hz (samples per 1-second data record).
    patient_id : str
        80-byte local patient identification string.
    recording_id : str
        80-byte local recording identification string.
    physical_dims : list[str] | None
        Optional physical dimension strings per channel (defaults to ``"uV"``).

    Returns
    -------
    Path
        Path to the written EDF file.
    """
    n_channels, n_samples = signals.shape
    if n_channels != len(channel_names):
        raise ValueError("signals.shape[0] must match len(channel_names).")
    if n_samples % sfreq != 0:
        raise ValueError("n_samples must be a multiple of sfreq for 1-second EDF records.")

    n_records = n_samples // sfreq
    header_bytes = 256 + n_channels * 256
    phys_dims = physical_dims or ["uV"] * n_channels

    fixed_header = b"".join(
        [
            _pad_ascii("0", 8),
            _pad_ascii(patient_id, 80),
            _pad_ascii(recording_id, 80),
            _pad_ascii("01.01.26", 8),
            _pad_ascii("12.00.00", 8),
            _pad_ascii(str(header_bytes), 8),
            _pad_ascii("", 44),
            _pad_ascii(str(n_records), 8),
            _pad_ascii("1", 8),
            _pad_ascii(str(n_channels), 4),
        ]
    )
    assert len(fixed_header) == 256

    phys_mins: list[float] = []
    phys_maxs: list[float] = []
    dig_min = -32768
    dig_max = 32767

    for ch_idx in range(n_channels):
        s_min = float(np.min(signals[ch_idx]))
        s_max = float(np.max(signals[ch_idx]))
        if np.isclose(s_min, s_max):
            s_min -= 100.0
            s_max += 100.0
        else:
            margin = 0.05 * (s_max - s_min)
            s_min -= margin
            s_max += margin
        phys_mins.append(s_min)
        phys_maxs.append(s_max)

    ch_header = b"".join(
        [
            b"".join(_pad_ascii(name, 16) for name in channel_names),
            b"".join(_pad_ascii("AgAgCl", 80) for _ in range(n_channels)),
            b"".join(_pad_ascii(dim, 8) for dim in phys_dims),
            b"".join(_pad_ascii(f"{pmin:.2f}", 8) for pmin in phys_mins),
            b"".join(_pad_ascii(f"{pmax:.2f}", 8) for pmax in phys_maxs),
            b"".join(_pad_ascii(str(dig_min), 8) for _ in range(n_channels)),
            b"".join(_pad_ascii(str(dig_max), 8) for _ in range(n_channels)),
            b"".join(_pad_ascii("HP:0.1Hz", 80) for _ in range(n_channels)),
            b"".join(_pad_ascii(str(sfreq), 8) for _ in range(n_channels)),
            b"".join(_pad_ascii("", 32) for _ in range(n_channels)),
        ]
    )
    assert len(ch_header) == n_channels * 256

    # Convert physical signals to 16-bit signed integers interleaved by 1-second record
    digital_signals = np.empty((n_channels, n_samples), dtype="<i2")
    for ch_idx in range(n_channels):
        pmin = phys_mins[ch_idx]
        pmax = phys_maxs[ch_idx]
        scale = (dig_max - dig_min) / (pmax - pmin)
        dig_vals = np.round((signals[ch_idx] - pmin) * scale + dig_min)
        digital_signals[ch_idx] = np.clip(dig_vals, dig_min, dig_max).astype("<i2")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as handle:
        handle.write(fixed_header)
        handle.write(ch_header)
        for rec_idx in range(n_records):
            start = rec_idx * sfreq
            stop = start + sfreq
            for ch_idx in range(n_channels):
                handle.write(digital_signals[ch_idx, start:stop].tobytes())

    return file_path
