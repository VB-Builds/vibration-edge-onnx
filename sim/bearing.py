"""Rolling-element bearing geometry and kinematic defect frequencies."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bearing:
    """Bearing geometry. Defaults approximate a 6205-type deep-groove ball bearing.

    Defect frequencies for a stationary outer race (shaft speed ``f`` in Hz):

        BPFO = (n / 2) * f * (1 - (d / D) * cos(theta))   ball pass, outer race
        BPFI = (n / 2) * f * (1 + (d / D) * cos(theta))   ball pass, inner race

    n = number of balls, d = ball diameter, D = pitch diameter,
    theta = contact angle. Pure rolling, no slip is assumed; real bearings
    deviate by about 1-2 %, which the simulator mimics with random slip.
    """

    n_balls: int = 9
    ball_d_mm: float = 7.94
    pitch_d_mm: float = 39.04
    contact_angle_deg: float = 0.0

    def _ratio(self) -> float:
        return self.ball_d_mm / self.pitch_d_mm * math.cos(math.radians(self.contact_angle_deg))

    def bpfo(self, shaft_hz: float) -> float:
        return self.n_balls / 2.0 * shaft_hz * (1.0 - self._ratio())

    def bpfi(self, shaft_hz: float) -> float:
        return self.n_balls / 2.0 * shaft_hz * (1.0 + self._ratio())
