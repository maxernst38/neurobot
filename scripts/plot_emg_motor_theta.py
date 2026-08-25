"""
Drive lerobot motor angles from EMG activation.

Pipeline per muscle group:
  1. Load all sample CSVs (Time_s, ADC) and RMS-filter each (sliding-window
     RMS envelope), then average across samples onto a shared time grid.
  2. Threshold that RMS curve at its configured activation_level (a
     fraction of its own peak amplitude) to get a binary activation
     function (0 = resting, 1 = activated).
  3. Integrate a constant angular rate while activated: the motor rotates
     at the configured rotation_rate for as long as the muscle is active,
     and holds its position while resting (a velocity command, not a
     position command) — clamped to that motor's real observed angle
     range.

Each muscle group gets its own graph (flexion and extension are NOT
merged) even though a flexion/extension pair shares one physical motor,
so each muscle's individual contribution is visible on its own.

Per-muscle motor id, rotation_rate (rad/s, signed — negative rotates the
opposite direction, used for the antagonist half of a pair),
activation_level, starting_angle, angle_min, and angle_max are all read
from config/motor_mapping.yaml. Display label/color are cosmetic and
kept here in MUSCLE_DISPLAY.

Motor ids are from the SO-101 arm's motor order: 1 shoulder_pan,
2 shoulder_lift, 3 elbow_flex, 4 wrist_flex, 5 wrist_roll, 6 gripper.
motor 2 and motor 5 have no recorded EMG channel and are left unmapped.

Usage:
    python scripts/plot_emg_motor_theta.py
"""

import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "emg")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "motor_mapping.yaml")

RMS_WINDOW_S = 0.08  # sliding window used to compute the RMS envelope
GRID_POINTS = 1200

MOTOR_NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
UNMAPPED_MOTORS = {2: "shoulder_lift", 5: "wrist_roll"}  # no EMG channel recorded for these

# Cosmetic, per-muscle display config not carried in config/motor_mapping.yaml.
MUSCLE_DISPLAY = {
    "shoulder": {"label": "Shoulder", "color": "#008300"},
    "elbow_flexion": {"label": "Elbow Flexion", "color": "#2a78d6"},
    "elbow_extension": {"label": "Elbow Extension", "color": "#eb6834"},
    "wrist_flexion": {"label": "Wrist Flexion", "color": "#1baf7a"},
    "wrist_extension": {"label": "Wrist Extension", "color": "#eda100"},
    "grip": {"label": "Grip", "color": "#e87ba4"},
}
MUSCLE_ORDER = ["shoulder", "elbow_flexion", "elbow_extension",
                "wrist_flexion", "wrist_extension", "grip"]


def load_muscle_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)["muscles"]


def load_samples(group_dir):
    samples = []
    for path in sorted(glob.glob(os.path.join(group_dir, "*.csv"))):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        t = df["time_s"].to_numpy(dtype=float)
        adc = df["adc"].to_numpy(dtype=float)
        order = np.argsort(t, kind="stable")
        samples.append((t[order], adc[order]))
    return samples


def rms_envelope(t, adc):
    # time_s is rounded to 10ms, so consecutive diffs are frequently 0 (many rows
    # share a timestamp); mean spacing over the whole recording is a more
    # reliable estimate of the true per-sample interval than a diff-based one.
    dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    dt = dt if dt > 0 else 0.001
    window = max(1, int(round(RMS_WINDOW_S / dt)))
    kernel = np.ones(window) / window
    return np.convolve(adc ** 2, kernel, mode="same") ** 0.5


def group_rms_curve(group_key):
    """Average RMS envelope for a muscle group, on its own absolute time grid."""
    samples = load_samples(os.path.join(DATA_DIR, group_key))
    if not samples:
        raise FileNotFoundError(f"No CSV files found for muscle group '{group_key}'")

    t_max = max(t[-1] for t, _ in samples)
    t_grid = np.linspace(0.0, t_max, GRID_POINTS)
    curves = [np.interp(t_grid, t, rms_envelope(t, adc)) for t, adc in samples]
    return t_grid, np.mean(curves, axis=0)


def activation_from_rms(rms_curve, activation_level):
    threshold = activation_level * np.max(rms_curve)
    return (rms_curve >= threshold).astype(float), threshold


def runs_where(mask):
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def integrate_theta(t, activation, rate_deg_s, rest_angle, lo, hi):
    """Rotate at a constant rate while activated, hold otherwise; clamp to [lo, hi]
    at every step so the motor can't wind up past its physical limit."""
    theta = np.empty_like(t)
    theta[0] = min(max(rest_angle, lo), hi)
    for i in range(1, len(t)):
        rate = rate_deg_s if activation[i - 1] >= 0.5 else 0.0
        theta[i] = min(max(theta[i - 1] + rate * (t[i] - t[i - 1]), lo), hi)
    return theta


def plot_muscle(ax_theta, ax_emg, group, muscle_cfg):
    display = MUSCLE_DISPLAY[group]
    motor_id = muscle_cfg["motor_id"]
    motor_name = MOTOR_NAMES[motor_id]
    rotation_rate_deg_s = math.degrees(muscle_cfg["rotation_rate"])

    t_grid, rms_curve = group_rms_curve(group)
    activation, threshold = activation_from_rms(rms_curve, muscle_cfg["activation_level"])

    lo, hi = muscle_cfg["angle_min"], muscle_cfg["angle_max"]
    rest_angle = muscle_cfg["starting_angle"]
    theta = integrate_theta(t_grid, activation, rotation_rate_deg_s, rest_angle, lo, hi)

    color = display["color"]
    ax_theta.plot(t_grid, theta, color=color, linewidth=2.5)
    ax_theta.axhline(rest_angle, color=color, linewidth=1.0, linestyle=":", alpha=0.5)
    ax_theta.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax_theta.set_title(f"{display['label']} — motor {motor_id} ({motor_name})",
                        color=color, fontweight="bold")
    ax_theta.set_ylabel("Angle theta (deg)")
    ax_theta.grid(axis="y", alpha=0.3)

    for start_idx, end_idx in runs_where(activation > 0.5):
        ax_emg.axvspan(t_grid[start_idx], t_grid[end_idx], color=color, alpha=0.1)
    ax_emg.plot(t_grid, rms_curve, color=color, linewidth=1.5)
    ax_emg.axhline(threshold, color=color, linewidth=1.0, linestyle=":", alpha=0.6)
    ax_emg.set_ylabel("Amplitude (ADC)")
    ax_emg.set_xlabel("Time (s)")
    ax_emg.grid(axis="y", alpha=0.3)


def main():
    muscle_config = load_muscle_config(CONFIG_PATH)

    n_cols = 3
    n_group_rows = math.ceil(len(MUSCLE_ORDER) / n_cols)
    fig, axes = plt.subplots(2 * n_group_rows, n_cols, figsize=(4.5 * n_cols, 8 * n_group_rows))

    for i, group in enumerate(MUSCLE_ORDER):
        group_row, col = divmod(i, n_cols)
        theta_row, emg_row = 2 * group_row, 2 * group_row + 1
        plot_muscle(axes[theta_row, col], axes[emg_row, col], group, muscle_config[group])

    mapping_rows = [
        {"muscle_group": group, "motor_id": cfg["motor_id"], "motor_name": MOTOR_NAMES[cfg["motor_id"]],
         "rotation_rate_rad_s": cfg["rotation_rate"], "activation_level": cfg["activation_level"],
         "starting_angle": cfg["starting_angle"], "angle_min": cfg["angle_min"], "angle_max": cfg["angle_max"]}
        for group, cfg in muscle_config.items()
    ]
    for motor_id, motor_name in UNMAPPED_MOTORS.items():
        mapping_rows.append({"muscle_group": None, "motor_id": motor_id, "motor_name": motor_name,
                              "rotation_rate_rad_s": None, "activation_level": None,
                              "starting_angle": None, "angle_min": None, "angle_max": None})

    fig.suptitle("Motor Angle Driven by EMG Activation (all parameters from config/motor_mapping.yaml)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "emg_motor_theta.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "muscle_motor_mapping.csv")
    pd.DataFrame(mapping_rows).sort_values("motor_id").to_csv(mapping_path, index=False)
    print(f"Saved muscle-to-motor mapping to {mapping_path}")


if __name__ == "__main__":
    main()
