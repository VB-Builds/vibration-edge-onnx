"""Physics-flavoured acceleration simulator for a rotating machine.

Components (all in m/s^2):
  * shaft 1x and 2x harmonics (residual imbalance / misalignment floor)
  * imbalance fault: 1x amplitude grows
  * looseness fault: sub-harmonic 0.5x and 3x..6x harmonics grow
  * bearing outer/inner race defects: impulse trains at BPFO/BPFI, each impulse
    rings a structural resonance (decaying sinusoid); inner-race impulses are
    amplitude-modulated at 1x (load zone)
  * broadband noise, per-window load variation and small speed jitter
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from sim.bearing import Bearing

RESONANCE_HZ = 1100.0
RESONANCE_TAU_S = 0.0015


@dataclass(frozen=True)
class Machine:
    name: str = "pump-01"
    shaft_hz: float = 29.5  # about 1770 rpm
    fs: float = 3200.0
    window_s: float = 1.0
    bearing: Bearing = field(default_factory=Bearing)

    @property
    def n_samples(self) -> int:
        return int(round(self.fs * self.window_s))


@dataclass(frozen=True)
class Health:
    """Fault severities, each in [0, 1]."""

    outer: float = 0.0
    inner: float = 0.0
    imbalance: float = 0.0
    looseness: float = 0.0


def _kernel(fs: float) -> NDArray[np.float64]:
    t = np.arange(int(0.025 * fs)) / fs
    return np.exp(-t / RESONANCE_TAU_S) * np.sin(2 * np.pi * RESONANCE_HZ * t)


def _impulses(
    n: int, fs: float, rate_hz: float, amp: float, rng: np.random.Generator,
    shaft_hz: float = 0.0, modulate: bool = False,
) -> NDArray[np.float64]:
    """Impulse train at ``rate_hz`` (1 % random slip) convolved with the resonance."""
    train = np.zeros(n)
    t = rng.uniform(0, 1 / rate_hz)
    while t < n / fs:
        a = amp * rng.uniform(0.7, 1.3)
        if modulate:
            a *= 0.5 * (1 + np.cos(2 * np.pi * shaft_hz * t))
        train[int(t * fs)] += a
        t += (1 / rate_hz) * (1 + 0.01 * rng.standard_normal())
    return np.convolve(train, _kernel(fs))[:n]


def synth_window(
    machine: Machine, health: Health, rng: np.random.Generator
) -> NDArray[np.float64]:
    """One acceleration window (``machine.n_samples`` samples) for the given health."""
    n, fs = machine.n_samples, machine.fs
    t = np.arange(n) / fs
    f = machine.shaft_hz * (1 + 0.003 * rng.standard_normal())
    load = float(np.exp(0.12 * rng.standard_normal()))
    ph = rng.uniform(0, 2 * np.pi, size=8)

    x = load * (0.25 * np.sin(2 * np.pi * f * t + ph[0]) + 0.08 * np.sin(4 * np.pi * f * t + ph[1]))
    x += 2.0 * health.imbalance * load * np.sin(2 * np.pi * f * t + ph[0])
    if health.looseness > 0:
        x += 0.5 * health.looseness * np.sin(np.pi * f * t + ph[2])
        for k in range(3, 7):
            x += 0.7 * health.looseness / (k - 2) * np.sin(2 * np.pi * k * f * t + ph[k])
    if health.outer > 0:
        x += _impulses(n, fs, machine.bearing.bpfo(f), 12.0 * health.outer * load, rng)
    if health.inner > 0:
        x += _impulses(n, fs, machine.bearing.bpfi(f), 12.0 * health.inner * load, rng, f, True)
    x += 0.25 * rng.standard_normal(n)
    return x


def generate_healthy(
    machine: Machine, n_windows: int, seed: int
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.stack([synth_window(machine, Health(), rng) for _ in range(n_windows)])


def _ramp(r: float, start: float) -> float:
    return float(np.clip((r - start) / (1 - start), 0.0, 1.0))


def health_at(i: int, n_windows: int, onset: int) -> Health:
    """Run-to-failure schedule: outer race first, then imbalance, inner race, looseness."""
    r = float(np.clip((i - onset) / max(n_windows - 1 - onset, 1), 0.0, 1.0))
    return Health(
        outer=r,
        inner=_ramp(r, 0.35),
        imbalance=0.75 * _ramp(r, 0.2),
        looseness=_ramp(r, 0.5),
    )


def run_to_failure(
    machine: Machine, n_windows: int = 400, onset: int = 200, seed: int = 1
) -> Iterator[tuple[int, Health, NDArray[np.float64]]]:
    """Yield ``(index, true_health, window)``; healthy until ``onset``, then degrading."""
    rng = np.random.default_rng(seed)
    for i in range(n_windows):
        h = health_at(i, n_windows, onset)
        yield i, h, synth_window(machine, h, rng)
