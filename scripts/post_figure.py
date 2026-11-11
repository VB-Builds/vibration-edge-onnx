"""Write docs/post-figure.png (960x700, dark) from a simulated run-to-failure scored by the committed model.

    python scripts/post_figure.py      (needs matplotlib)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from edge.infer import DEFAULT_MODEL, HealthModel  # noqa: E402
from features import iso_zone, velocity_rms_mms  # noqa: E402
from sim import Machine, run_to_failure  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED = "#38BDF8", "#34D399", "#FBBF24", "#F87171"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
N, ONSET = 400, 200


def main() -> None:
    from train import featurise

    m = Machine()
    model = HealthModel(DEFAULT_MODEL)
    rows = list(run_to_failure(m, N, ONSET, seed=5))
    wins = np.stack([r[2] for r in rows])
    feats = featurise(wins, m)
    ratio = np.array([model.score(f)[0] for f in feats])
    alarm = ratio > 1
    first = next(i for i in range(N - 2) if alarm[i : i + 3].all())
    zones = np.array(["ABCD".index(iso_zone(velocity_rms_mms(w, m.fs))) for w in wins])
    print(f"first 3-in-a-row alarm={first} (onset {ONSET}); zone at first alarm={'ABCD'[zones[first]]}; "
          f"max ratio={ratio.max():.0f}")

    fig, (ax, zx) = plt.subplots(2, 1, figsize=(9.6, 7.0), dpi=100, facecolor=BG,
                                 gridspec_kw={"height_ratios": [6, 0.8], "hspace": 0.28})
    fig.subplots_adjust(left=0.13, right=0.96, top=0.80, bottom=0.12)
    fig.text(0.04, 0.93, "Bearing fault caught before the vibration level moves", color=TEXT, fontsize=19,
             fontweight="bold")
    fig.text(0.04, 0.88, "Simulated data: 400 one-second windows, 1.7 KB ONNX model", color=MUTED, fontsize=15)
    ax.set_facecolor(BG)
    ax.set_yscale("log")
    ax.plot(np.arange(N), np.maximum(ratio, 0.02), color=ACCENT, lw=2.8)
    ax.axhline(1, color=AMBER, lw=2.5, ls="--")
    ax.axvline(ONSET, color=TEXT, lw=2, ls=":")
    ax.plot([first], [ratio[first]], "o", color=RED, ms=12)
    ax.set_ylim(0.03, 2e4)
    ax.set_xlim(0, N)
    ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["0.1", "1", "10", "100", "1k", "10k"])
    ax.text(8, 1.4, "alarm threshold", color=AMBER, fontsize=15)
    ax.text(ONSET - 6, 3000, "fault begins", color=TEXT, fontsize=15, ha="right")
    ax.annotate(f"first alarm: window {first}", (first, ratio[first]), (235, 0.12), color=RED, fontsize=15,
                fontweight="bold", arrowprops={"arrowstyle": "-", "color": RED, "lw": 1.5})
    ax.set_ylabel("score / threshold", color=MUTED, fontsize=15)
    ax.set_xticklabels([])
    zx.imshow(zones[None, :], aspect="auto", extent=(0, N, 0, 1),
              cmap=ListedColormap([GREEN, "#A3E635", AMBER, RED]), vmin=0, vmax=3)
    zx.set_yticks([])
    zx.set_xlabel("window index", color=MUTED, fontsize=15)
    zx.text(-0.01, 0.5, "ISO zone", color=MUTED, fontsize=15, ha="right", va="center", transform=zx.transAxes)
    for a in (ax, zx):
        a.tick_params(colors=TEXT, labelsize=15)
        for sp in a.spines.values():
            sp.set_color(GRID)
    ax.grid(color=GRID, lw=1)
    fig.savefig(ROOT / "docs" / "post-figure.png", facecolor=BG)


if __name__ == "__main__":
    main()
