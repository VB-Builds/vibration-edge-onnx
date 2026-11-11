"""Edge inference: acceleration windows -> features -> ONNX Runtime -> health JSON.

Input (pick one): --sim (built-in run-to-failure simulator), --csv FILE, or --csv -
(stdin). CSV rows are one window each: fs*window_s comma-separated samples (m/s^2).
Output: one JSON line per window on stdout and, with --mqtt-host, on
``plant/<machine>/health``. Latency percentiles go to stderr at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from features import extract_features, iso_zone, FEATURE_NAMES
from sim import Machine, run_to_failure

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "vib_anomaly.onnx"


class HealthModel:
    """ONNX session plus the metadata baked into the model file."""

    def __init__(self, path: Path = DEFAULT_MODEL) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # tiny model; avoids thread-pool overhead on small ARM cores
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.feature_names: list[str] = json.loads(meta["feature_names"])
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("model feature list does not match features.py; retrain with train.py")
        self.threshold = float(meta["threshold_sq_err"])

    def score(self, feats: NDArray[np.float64]) -> tuple[float, NDArray[np.float32]]:
        """Return (error/threshold ratio, standardised features) for one feature vector."""
        ratio, z = self.session.run(None, {"features": feats.astype(np.float32)[None, :]})
        return float(ratio[0]), z[0]


def health_score(ratio: float) -> float:
    """Map error/threshold ratio to (0, 1]: 1 = healthy-like, 0.5 = at threshold."""
    return 1.0 / (1.0 + ratio * ratio)


def read_csv(stream: Iterable[str]) -> Iterator[NDArray[np.float64]]:
    for line in stream:
        line = line.strip()
        if line and not line.startswith("#"):
            yield np.array([float(v) for v in line.split(",")])


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def run(windows: Iterable[NDArray[np.float64]], model: HealthModel, machine: Machine,
        publish: Any = None, interval_s: float = 0.0, quiet: bool = False) -> dict[str, float]:
    topic = f"plant/{machine.name}/health"
    infer_ms: list[float] = []
    total_ms: list[float] = []
    for w in windows:
        if len(w) != machine.n_samples:
            raise ValueError(f"window has {len(w)} samples, expected {machine.n_samples}")
        t0 = time.perf_counter()
        feats = extract_features(w, machine)
        t1 = time.perf_counter()
        ratio, z = model.score(feats)
        t2 = time.perf_counter()
        infer_ms.append((t2 - t1) * 1e3)
        total_ms.append((t2 - t0) * 1e3)
        msg = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "machine": machine.name,
            "health_score": round(health_score(ratio), 3),
            "anomaly": ratio > 1.0,
            "iso_zone": iso_zone(float(feats[FEATURE_NAMES.index("vel_rms_mms")])),
            "top_feature": FEATURE_NAMES[int(np.argmax(np.abs(z)))],
        }
        line = json.dumps(msg)
        if not quiet:
            print(line, flush=True)
        if publish is not None:
            publish(topic, line)
        if interval_s > 0:
            time.sleep(interval_s)
    return {
        "windows": len(total_ms),
        "onnx_p50_ms": percentile(infer_ms, 50), "onnx_p95_ms": percentile(infer_ms, 95),
        "pipeline_p50_ms": percentile(total_ms, 50), "pipeline_p95_ms": percentile(total_ms, 95),
    }


def make_publisher(host: str, port: int) -> tuple[Any, Any]:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("paho-mqtt is not installed; pip install paho-mqtt or drop --mqtt-host")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) if hasattr(mqtt, "CallbackAPIVersion") \
        else mqtt.Client()
    client.connect(host, port, keepalive=30)
    client.loop_start()

    def close() -> None:
        client.loop_stop()
        client.disconnect()

    return (lambda topic, payload: client.publish(topic, payload, qos=1)), close


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", action="store_true", help="simulated run-to-failure")
    src.add_argument("--csv", metavar="FILE", help="CSV of windows, '-' for stdin")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--machine", default="pump-01")
    ap.add_argument("--windows", type=int, default=90, help="sim: windows per run")
    ap.add_argument("--onset", type=int, default=30, help="sim: window where degradation starts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--loop", action="store_true", help="sim: repeat the run forever")
    ap.add_argument("--interval", type=float, default=0.0, help="seconds between windows")
    ap.add_argument("--mqtt-host", default=None)
    ap.add_argument("--mqtt-port", type=int, default=1883)
    args = ap.parse_args(argv)

    machine = Machine(name=args.machine)
    model = HealthModel(args.model)
    publish, close = (None, lambda: None)
    if args.mqtt_host:
        publish, close = make_publisher(args.mqtt_host, args.mqtt_port)

    def sim_windows() -> Iterator[NDArray[np.float64]]:
        seed = args.seed
        while True:
            for _, _, w in run_to_failure(machine, args.windows, args.onset, seed):
                yield w
            if not args.loop:
                return
            seed += 1

    if args.sim:
        windows = sim_windows()
    elif args.csv == "-":
        windows = read_csv(sys.stdin)
    else:
        f = open(args.csv)
        windows = read_csv(f)
    try:
        stats = run(windows, model, machine, publish, args.interval)
    except KeyboardInterrupt:
        stats = {}
    finally:
        close()
    print("latency " + json.dumps({k: round(v, 3) for k, v in stats.items()}), file=sys.stderr)


if __name__ == "__main__":
    main()
