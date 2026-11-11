"""Train a PCA reconstruction-error anomaly model on healthy windows only and export ONNX.

The standardisation (healthy mean/std), PCA projection/reconstruction, per-feature
residuals and threshold scaling are all inside the ONNX graph, so the edge
runtime only feeds raw feature vectors.

ONNX outputs:
  ratio [N]   reconstruction error / threshold (> 1 means anomaly)
  z     [N,F] standardised features (used to name the top deviating feature)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper
from sklearn.decomposition import PCA

from features import FEATURE_NAMES, extract_features
from sim import Machine, generate_healthy

MODEL_PATH = Path("models/vib_anomaly.onnx")
THRESHOLD_PERCENTILE = 99.0


@dataclass
class Fitted:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    pca: PCA
    threshold: float

    def standardise(self, feats: NDArray[np.float64]) -> NDArray[np.float64]:
        return (feats - self.mean) / self.std

    def error(self, feats: NDArray[np.float64]) -> NDArray[np.float64]:
        """Python reference: squared PCA reconstruction error in standardised space."""
        z = self.standardise(feats)
        rec = self.pca.inverse_transform(self.pca.transform(z))
        return np.sum((z - rec) ** 2, axis=1)


def featurise(windows: NDArray[np.float64], machine: Machine) -> NDArray[np.float64]:
    return np.stack([extract_features(w, machine) for w in windows])


def fit(machine: Machine, n_fit: int = 500, n_calib: int = 500, seed: int = 0,
        variance: float = 0.9) -> Fitted:
    """Fit on ``n_fit`` healthy windows; pick the threshold on ``n_calib`` other healthy windows."""
    fit_x = featurise(generate_healthy(machine, n_fit, seed), machine)
    calib_x = featurise(generate_healthy(machine, n_calib, seed + 1), machine)
    mean, std = fit_x.mean(0), fit_x.std(0) + 1e-9
    pca = PCA(n_components=variance, svd_solver="full").fit((fit_x - mean) / std)
    model = Fitted(mean, std, pca, 1.0)
    model.threshold = float(np.percentile(model.error(calib_x), THRESHOLD_PERCENTILE))
    return model


def to_onnx(m: Fitted, machine: Machine) -> onnx.ModelProto:
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    comps = f32(m.pca.components_)  # [k, F]
    inits = [
        numpy_helper.from_array(f32(m.mean), "mean"),
        numpy_helper.from_array(f32(m.std), "std"),
        numpy_helper.from_array(f32(m.pca.mean_), "pca_mean"),
        numpy_helper.from_array(comps.T.copy(), "comp_t"),
        numpy_helper.from_array(comps, "comp"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
        numpy_helper.from_array(f32(m.threshold), "threshold"),
    ]
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["centred"]),
        helper.make_node("Div", ["centred", "std"], ["z"]),
        helper.make_node("Sub", ["z", "pca_mean"], ["zc"]),
        helper.make_node("MatMul", ["zc", "comp_t"], ["proj"]),
        helper.make_node("MatMul", ["proj", "comp"], ["rec_c"]),
        helper.make_node("Sub", ["zc", "rec_c"], ["resid"]),
        helper.make_node("Mul", ["resid", "resid"], ["resid2"]),
        helper.make_node("ReduceSum", ["resid2", "axes"], ["err"], keepdims=0),
        helper.make_node("Div", ["err", "threshold"], ["ratio"]),
    ]
    nf = len(FEATURE_NAMES)
    graph = helper.make_graph(
        nodes, "vib_anomaly",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", nf])],
        [helper.make_tensor_value_info("ratio", TensorProto.FLOAT, ["N"]),
         helper.make_tensor_value_info("z", TensorProto.FLOAT, ["N", nf])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="vibration-edge-onnx")
    model.ir_version = 8
    meta = {
        "feature_names": json.dumps(FEATURE_NAMES),
        "threshold_sq_err": repr(m.threshold),
        "threshold_percentile": str(THRESHOLD_PERCENTILE),
        "machine": json.dumps({"name": machine.name, "shaft_hz": machine.shaft_hz, "fs": machine.fs}),
        "note": "trained on simulated healthy windows only",
    }
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    machine = Machine()
    m = fit(machine, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(to_onnx(m, machine), args.out)
    print(f"features={len(FEATURE_NAMES)} pca_components={m.pca.n_components_} "
          f"threshold(sq err, p{THRESHOLD_PERCENTILE:g})={m.threshold:.3f}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
