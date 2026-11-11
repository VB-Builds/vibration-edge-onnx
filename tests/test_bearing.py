import math

from sim import Bearing


def test_bpfo_bpfi_hand_computed_zero_contact_angle() -> None:
    # n=8, d=10 mm, D=50 mm, theta=0, f=25 Hz -> d/D = 0.2
    # BPFO = 8/2 * 25 * (1 - 0.2) = 80 Hz ; BPFI = 4 * 25 * 1.2 = 120 Hz
    b = Bearing(n_balls=8, ball_d_mm=10.0, pitch_d_mm=50.0, contact_angle_deg=0.0)
    assert math.isclose(b.bpfo(25.0), 80.0)
    assert math.isclose(b.bpfi(25.0), 120.0)


def test_contact_angle_changes_frequencies() -> None:
    # theta=60 deg -> cos = 0.5 -> ratio 0.1: BPFO = 100*0.9 = 90, BPFI = 110
    b = Bearing(n_balls=8, ball_d_mm=10.0, pitch_d_mm=50.0, contact_angle_deg=60.0)
    assert math.isclose(b.bpfo(25.0), 90.0)
    assert math.isclose(b.bpfi(25.0), 110.0)


def test_bpfo_plus_bpfi_is_n_times_shaft() -> None:
    b = Bearing()
    assert math.isclose(b.bpfo(30.0) + b.bpfi(30.0), b.n_balls * 30.0)
