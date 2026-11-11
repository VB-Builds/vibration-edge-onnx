"""Rotating-machine vibration simulator."""
from sim.bearing import Bearing
from sim.simulator import Health, Machine, generate_healthy, run_to_failure, synth_window

__all__ = ["Bearing", "Health", "Machine", "generate_healthy", "run_to_failure", "synth_window"]
