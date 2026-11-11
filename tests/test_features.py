import math

import numpy as np
import pytest

from features import FEATURE_NAMES, extract_features, iso_zone, velocity_rms_mms
from sim import Machine

M = Machine()
T = np.arange(M.n_samples) / M.fs


def feat(x: np.ndarray) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, extract_features(x, M)))


def test_sine_rms_peak_crest_kurtosis() -> None:
    a = 2.0
    f = feat(a * np.sin(2 * np.pi * 50 * T))
    assert f["rms"] == pytest.approx(a / math.sqrt(2), rel=1e-6)
    assert f["peak"] == pytest.approx(a, rel=1e-3)
    assert f["crest"] == pytest.approx(math.sqrt(2), rel=1e-3)
    assert f["kurtosis"] == pytest.approx(1.5, rel=1e-3)  # sine kurtosis = 3/2


def test_shaft_harmonic_amplitudes() -> None:
    x = 0.8 * np.sin(2 * np.pi * M.shaft_hz * T) + 0.3 * np.sin(2 * np.pi * 2 * M.shaft_hz * T)
    f = feat(x)
    assert f["amp_1x"] == pytest.approx(0.8, rel=0.02)
    assert f["amp_2x"] == pytest.approx(0.3, rel=0.02)


def test_velocity_rms_of_sine() -> None:
    # acceleration amplitude A at f -> velocity rms = A / (sqrt2 * 2*pi*f)
    a, f = 1.0, 20.0
    expect = a / (math.sqrt(2) * 2 * math.pi * f) * 1000
    assert velocity_rms_mms(a * np.sin(2 * np.pi * f * T), M.fs) == pytest.approx(expect, rel=1e-3)


def test_envelope_energy_at_bpfo_for_amplitude_modulated_carrier() -> None:
    fo = M.bearing.bpfo(M.shaft_hz)
    x = (1 + 0.8 * np.sin(2 * np.pi * fo * T)) * np.sin(2 * np.pi * 1000 * T)
    f = feat(x)
    assert f["log_env_bpfo"] > f["log_env_bpfi"] + 3  # >3 decades above the other line


@pytest.mark.parametrize(
    "v,zone", [(0.5, "A"), (1.39, "A"), (1.4, "B"), (2.79, "B"), (2.8, "C"), (4.49, "C"), (4.5, "D"), (20, "D")]
)
def test_iso_zone_mapping(v: float, zone: str) -> None:
    assert iso_zone(v) == zone


def test_iso_zone_custom_limits() -> None:
    assert iso_zone(2.0, (0.7, 1.8, 4.5)) == "C"
