"""
Plot theta (motor angle) vs time for all 6 leader and follower motors.
Each motor gets its own hue; leader and follower share that hue but are
plotted at different brightness levels so pairs are easy to tell apart
while still visually matched.

Usage:
    python scripts/plot_teleop_latency.py data/latency_data.csv [cutoff_seconds]

    cutoff_seconds (optional): only plot data up to this time (in seconds).
    If omitted, the full recording is plotted.
"""

import os
import sys
import colorsys

import matplotlib.pyplot as plt
import pandas as pd

# Base hues (0-1) for each motor, evenly spaced around the color wheel
MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def hue_to_rgb(hue, lightness):
    """Convert a hue (0-1) + lightness (0-1) to an RGB tuple using HLS,
    keeping saturation fixed and high so colors stay vivid."""
    r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.75)
    return (r, g, b)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_teleop_latency.py <csv_file> [cutoff_seconds]")
        sys.exit(1)

    cutoff = None
    if len(sys.argv) >= 3:
        try:
            cutoff = float(sys.argv[2])
        except ValueError:
            print(f"Invalid cutoff_seconds value: {sys.argv[2]!r} (must be a number)")
            sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    if cutoff is not None:
        before = len(df)
        df = df[(df["t_leader"] <= cutoff) & (df["t_follower"] <= cutoff)].reset_index(drop=True)
        print(f"Cutoff={cutoff}s: kept {len(df)}/{before} rows")
        if df.empty:
            print("No data remains after applying cutoff. Check the value and try again.")
            sys.exit(1)

    # Figure out actual column names (they end in ".pos", e.g. "leader__shoulder_pan.pos")
    leader_cols = {c[len("leader__"):]: c for c in df.columns if c.startswith("leader__")}
    follower_cols = {c[len("follower__"):]: c for c in df.columns if c.startswith("follower__")}

    # Preserve a sensible order: known motor names first, then anything else found
    motor_keys = list(leader_cols.keys())
    ordered = [k for name in MOTOR_ORDER for k in motor_keys if k.startswith(name)]
    ordered += [k for k in motor_keys if k not in ordered]

    n_motors = len(ordered)
    hues = [i / n_motors for i in range(n_motors)]

    fig, ax = plt.subplots(figsize=(12, 7))

    for hue, key in zip(hues, ordered):
        leader_color = hue_to_rgb(hue, lightness=0.70)   # lighter shade
        follower_color = hue_to_rgb(hue, lightness=0.40)  # darker shade

        motor_label = key.replace(".pos", "").replace("_", " ")

        ax.plot(
            df["t_leader"], df[leader_cols[key]],
            color=leader_color, linewidth=1.6, linestyle="-",
            label=f"{motor_label} (leader)",
        )
        ax.plot(
            df["t_follower"], df[follower_cols[key]],
            color=follower_color, linewidth=1.6, linestyle="--",
            label=f"{motor_label} (follower)",
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Leader vs Follower Motor Angles Over Time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "motor_angles_plot.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()