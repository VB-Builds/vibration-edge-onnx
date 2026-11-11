"""Feature extraction for one acceleration window (m/s^2) plus ISO-style zone label."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.simulator import Machine

FEATURE_NAMES: tuple[str, ...] = (
    "rms", "peak", "crest", "kurtosis", "vel_rms_mms",
    "amp_1x", "amp_2x",
    "log_band_10_100", "log_band_100_400", "log_band_400_800", "log_band_800_1600",
    "log_env_bpfo", "log_env_bpfi",
)

BANDS_HZ = ((10, 100), (100, 400), (400, 800), (800, 1600))
ENVELOPE_PASSBAND_HZ = (600.0, 1500.0)
LINE_BW_HZ = 2.0  # half-width around a spectral line

# Illustrative zone limits for velocity RMS in mm/s: A/B, B/C, C/D.
# They resemble the commonly quoted values for medium-size machines (group 2)
# on rigid foundations in the ISO 10816 / ISO 20816 family of standards. Real
# limits depend on machine class, foundation and the standard edition: take
# them from the applicable standard for YOUR machine. Not a compliance tool.
ZONE_LIMITS_MM_S: tuple[float, float, float] = (1.4, 2.8, 4.5)


def iso_zone(vel_rms_mms: float, limits: tuple[float, float, float] = ZONE_LIMITS_MM_S) -> str:
    """Map velocity RMS (mm/s) to a zone letter A (new machine) .. D (damaging)."""
    for zone, limit in zip("ABC", limits):
        if vel_rms_mms < limit:
            return zone
    return "D"


HANN_ENBW_BINS = 1.5  # equivalent noise bandwidth of a Hann window, in bins


def _amp_spectrum(
    x: NDArray[np.float64], hann: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One-sided amplitude spectrum (peak amplitude per bin) and bin frequencies (unit fs).

    ``hann=True`` windows the data so a line that falls between bins is not under-read
    (use with ``_line_amp``, which sums the lobe energy and divides out the window's ENBW).
    """
    n = len(x)
    w = np.hanning(n) if hann else np.ones(n)
    amp = 2.0 * np.abs(np.fft.rfft(x * w)) / w.sum()
    amp[0] /= 2.0
    return amp, np.fft.rfftfreq(n)


def _line_amp(amp: NDArray[np.float64], freqs: NDArray[np.float64], f0: float, bw: float) -> float:
    m = np.abs(freqs - f0) <= bw
    return float(np.sqrt(np.sum(amp[m] ** 2) / HANN_ENBW_BINS))


def envelope(x: NDArray[np.float64], fs: float, band: tuple[float, float]) -> NDArray[np.float64]:
    """Envelope of the band-passed signal via the analytic signal (FFT Hilbert)."""
    n = len(x)
    spec = np.fft.fft(x)
    f = np.abs(np.fft.fftfreq(n, 1 / fs))
    spec[(f < band[0]) | (f > band[1])] = 0.0
    h = np.zeros(n)
    h[0] = 1.0
    h[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    return np.abs(np.fft.ifft(spec * h))


def velocity_rms_mms(x: NDArray[np.float64], fs: float, band: tuple[float, float] = (10.0, 1000.0)) -> float:
    """Velocity RMS in mm/s by frequency-domain integration over 10-1000 Hz."""
    amp, f = _amp_spectrum(x)
    f = f * fs
    m = (f >= band[0]) & (f <= band[1])
    vel_amp = amp[m] / (2 * np.pi * f[m])  # m/s peak per bin
    return float(np.sqrt(np.sum(vel_amp**2) / 2.0) * 1000.0)


def extract_features(window: NDArray[np.float64], machine: Machine) -> NDArray[np.float64]:
    """Feature vector ordered as ``FEATURE_NAMES``."""
    fs = machine.fs
    x = np.asarray(window, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    kurt = float(np.mean(x**4) / (np.mean(x**2) ** 2 + 1e-18))

    amp, fn = _amp_spectrum(x)
    f = fn * fs
    bw = LINE_BW_HZ / fs
    hamp, _ = _amp_spectrum(x, hann=True)
    a1 = _line_amp(hamp, fn, machine.shaft_hz / fs, bw)
    a2 = _line_amp(hamp, fn, 2 * machine.shaft_hz / fs, bw)

    # Band energy (mean square) from the one-sided spectrum; log10 to tame the dynamic range.
    bands = [
        float(np.log10(np.sum(amp[(f >= lo) & (f < hi)] ** 2) / 2.0 + 1e-12)) for lo, hi in BANDS_HZ
    ]

    env = envelope(x, fs, ENVELOPE_PASSBAND_HZ)
    eamp, efn = _amp_spectrum(env - env.mean(), hann=True)
    env_feats = []
    for fault_hz in (machine.bearing.bpfo(machine.shaft_hz), machine.bearing.bpfi(machine.shaft_hz)):
        e = sum(_line_amp(eamp, efn, h * fault_hz / fs, bw) ** 2 for h in (1, 2))
        env_feats.append(float(np.log10(e + 1e-12)))
    return np.array(
        [rms, peak, peak / (rms + 1e-12), kurt, velocity_rms_mms(x, fs), a1, a2, *bands, *env_feats]
    )
