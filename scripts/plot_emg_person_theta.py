"""
Simulate one person's whole-arm muscle activation, staggered on one graph.

Same visual style as plot_emg_motor_theta.py (top row = theta(t), bottom
row = the EMG RMS envelope that drove it), but each column here is a
PERSON instead of a muscle: for each person, all six of their own muscle
recordings are plotted together on shared axes rather than one muscle per
column.

The six muscles still overlap in time (each keeps its own full ~5.45s
recording), but each one's start is staggered 2 seconds after the
previous, in this order: shoulder, elbow_extension (tricep),
elbow_flexion (bicep), wrist_extension, wrist_flexion, grip. That keeps
them from landing exactly on top of each other so each muscle's own
spike, and the theta reaction it caused, stays visually traceable.

Same EMG -> RMS -> activation -> constant-rate rotation pipeline as
plot_emg_motor_theta.py: motor id, rotation_rate, activation_level,
starting_angle, angle_min, and angle_max are all read from
config/motor_mapping.yaml.

Usage:
    python scripts/plot_emg_person_theta.py
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
START_STAGGER_S = 2.0  # each muscle starts this many seconds after the previous one

PEOPLE = ["adhith", "raymond", "yousif"]

# Cosmetic, per-muscle display config (colors match the other emg_* scripts).
MUSCLE_DISPLAY = {
    "shoulder": {"label": "Shoulder", "color": "#008300"},
    "elbow_extension": {"label": "Tricep", "color": "#eb6834"},
    "elbow_flexion": {"label": "Bicep", "color": "#2a78d6"},
    "wrist_extension": {"label": "Wrist Extension", "color": "#eda100"},
    "wrist_flexion": {"label": "Wrist Flexion", "color": "#1baf7a"},
    "grip": {"label": "Grip", "color": "#e87ba4"},
}
# Playback order: shoulder, tricep, bicep, wrist extension, wrist flexion, grip.
MUSCLE_ORDER = ["shoulder", "elbow_extension", "elbow_flexion",
                "wrist_extension", "wrist_flexion", "grip"]


def load_muscle_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)["muscles"]


def find_person_csv(group_dir, person):
    matches = [
        path for path in glob.glob(os.path.join(group_dir, "*.csv"))
        if os.path.basename(path).lower().startswith(person.lower())
    ]
    if not matches:
        raise FileNotFoundError(f"No CSV for '{person}' in {group_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple CSVs for '{person}' in {group_dir}: {matches}")
    return matches[0]


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    t = df["time_s"].to_numpy(dtype=float)
    adc = df["adc"].to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    return t[order], adc[order]


def rms_envelope(t, adc):
    # time_s is rounded to 10ms, so consecutive diffs are frequently 0 (many rows
    # share a timestamp); mean spacing over the whole recording is a more
    # reliable estimate of the true per-sample interval than a diff-based one.
    dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    dt = dt if dt > 0 else 0.001
    window = max(1, int(round(RMS_WINDOW_S / dt)))
    kernel = np.ones(window) / window
    return np.convolve(adc ** 2, kernel, mode="same") ** 0.5


def integrate_theta(t, activation, rate_deg_s, rest_angle, lo, hi):
    """Rotate at a constant rate while activated, hold otherwise; clamp to [lo, hi]
    at every step so the motor can't wind up past its physical limit."""
    theta = np.empty_like(t)
    theta[0] = min(max(rest_angle, lo), hi)
    for i in range(1, len(t)):
        rate = rate_deg_s if activation[i - 1] >= 0.5 else 0.0
        theta[i] = min(max(theta[i - 1] + rate * (t[i] - t[i - 1]), lo), hi)
    return theta


def person_muscle_curves(person, group, muscle_cfg):
    t, adc = load_csv(find_person_csv(os.path.join(DATA_DIR, group), person))
    rms_curve = rms_envelope(t, adc)
    threshold = muscle_cfg["activation_level"] * np.max(rms_curve)
    activation = (rms_curve >= threshold).astype(float)

    lo, hi = muscle_cfg["angle_min"], muscle_cfg["angle_max"]
    rest_angle = muscle_cfg["starting_angle"]
    rate_deg_s = math.degrees(muscle_cfg["rotation_rate"])

    theta = integrate_theta(t, activation, rate_deg_s, rest_angle, lo, hi)
    return t, rms_curve, threshold, theta


def plot_person(ax_theta, ax_emg, person, muscle_config):
    """ax_theta is the top row, ax_emg is the bottom row (matches graph_emg_motor_theta.py)."""
    curves = []
    for i, group in enumerate(MUSCLE_ORDER):
        t, rms_curve, threshold, theta = person_muscle_curves(person, group, muscle_config[group])
        start = i * START_STAGGER_S
        t_shifted = t - t[0] + start
        curves.append((group, t_shifted, rms_curve, threshold, theta))

    # A motor holds its last commanded angle once its muscle's recording ends (nothing
    # moves it further), so extend every theta line flat out to the experiment's end
    # instead of letting it stop mid-graph where that muscle's own data happens to end.
    experiment_end = max(t_shifted[-1] for _, t_shifted, _, _, _ in curves)

    for group, t_shifted, rms_curve, threshold, theta in curves:
        display = MUSCLE_DISPLAY[group]
        t_theta, theta_extended = t_shifted, theta
        if t_shifted[-1] < experiment_end:
            t_theta = np.append(t_shifted, experiment_end)
            theta_extended = np.append(theta, theta[-1])

        ax_theta.plot(t_theta, theta_extended, color=display["color"], linewidth=1.8, label=display["label"])
        ax_emg.plot(t_shifted, rms_curve, color=display["color"], linewidth=1.2, label=display["label"])
        ax_emg.hlines(threshold, t_shifted[0], t_shifted[-1],
                       color=display["color"], linewidth=0.8, linestyle=":", alpha=0.6)

    ax_theta.set_title(person.capitalize(), fontweight="bold")
    ax_theta.set_ylabel("Angle theta (deg)")
    ax_theta.grid(alpha=0.3)

    ax_emg.set_xlabel("Time (s)")
    ax_emg.set_ylabel("Amplitude (ADC)")
    ax_emg.grid(alpha=0.3)


def main():
    muscle_config = load_muscle_config(CONFIG_PATH)

    fig, axes = plt.subplots(2, len(PEOPLE), figsize=(6 * len(PEOPLE), 8))
    for col, person in enumerate(PEOPLE):
        plot_person(axes[0, col], axes[1, col], person, muscle_config)

    fig.suptitle(
        "Simulated Whole-Arm Motor Activation per Person "
        f"(each muscle starts {START_STAGGER_S:.0f}s after the previous)",
        fontsize=14, y=0.99,
    )
    # One shared legend (same muscle -> color mapping in every panel) instead of a
    # legend box on each subplot, so labels never sit on top of the data.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MUSCLE_ORDER), fontsize=9, bbox_to_anchor=(0.5, 0.94))

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "emg_person_theta.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
