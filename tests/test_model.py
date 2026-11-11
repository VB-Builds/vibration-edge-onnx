"""Model quality on simulated data and ONNX parity. Prints the numbers quoted in the README."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from edge.infer import DEFAULT_MODEL, HealthModel
from sim import Machine, generate_healthy, run_to_failure
from train import Fitted, featurise, fit, to_onnx

M = Machine()
N, ONSET = 400, 200


@pytest.fixture(scope="module")
def fitted() -> Fitted:
    return fit(M, seed=0)


@pytest.fixture(scope="module")
def run_feats() -> np.ndarray:
    rows = list(run_to_failure(M, N, ONSET, seed=5))
    return featurise(np.stack([r[2] for r in rows]), M)


def first_sustained(flags: np.ndarray, k: int = 3) -> int:
    for i in range(len(flags) - k + 1):
        if flags[i : i + k].all():
            return i
    return -1


def test_onnx_matches_python_model(fitted: Fitted, run_feats: np.ndarray, tmp_path: Path) -> None:
    path = tmp_path / "m.onnx"
    onnx.save(to_onnx(fitted, M), path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ratio, z = sess.run(None, {"features": run_feats.astype(np.float32)})
    ref = fitted.error(run_feats) / fitted.threshold
    # healthy-to-moderate range: tight absolute 1e-5 on the ratio
    sel = ref < 5
    assert sel.sum() > 100
    assert np.max(np.abs(ratio[sel] - ref[sel])) < 1e-5
    # whole range incl. extreme faults: float32 relative accuracy
    np.testing.assert_allclose(ratio, ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(z, fitted.standardise(run_feats), rtol=1e-4, atol=1e-5)


def test_detects_late_windows_not_healthy(fitted: Fitted, run_feats: np.ndarray) -> None:
    alarm = fitted.error(run_feats) > fitted.threshold
    held_out = featurise(generate_healthy(M, 500, seed=99), M)
    fp_rate = float(np.mean(fitted.error(held_out) > fitted.threshold))
    pre_fp = float(alarm[:ONSET].mean())
    first = first_sustained(alarm)
    print(f"\nheld-out healthy FP rate={fp_rate:.3%}  pre-onset FP rate={pre_fp:.3%}")
    print(f"onset={ONSET} first 3-in-a-row alarm={first} delay={first - ONSET} windows, "
          f"lead before failure={N - 1 - first} windows")
    assert fp_rate < 0.05 and pre_fp < 0.05
    assert alarm[N - 50 :].mean() > 0.95
    assert ONSET <= first <= ONSET + 40


def test_committed_model_loads_and_flags_failure() -> None:
    model = HealthModel(DEFAULT_MODEL)
    healthy = featurise(generate_healthy(M, 30, seed=123), M)
    late = featurise(np.stack([r[2] for r in run_to_failure(M, 60, 10, seed=3)][-10:]), M)
    assert np.mean([model.score(f)[0] > 1 for f in healthy]) < 0.1
    assert all(model.score(f)[0] > 1 for f in late)
