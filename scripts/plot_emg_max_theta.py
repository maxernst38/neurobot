"""
Standalone version of plot_emg_person_theta.py's "Max" panel.

Reuses the exact same per-person plotting logic (theta on top, EMG RMS
envelope below) but renders only Max instead of all four people, so his
graphs can be shared on their own instead of cropped out of the combined
emg_person_theta.png.

Usage:
    python scripts/plot_emg_max_theta.py
"""

import os

import matplotlib.pyplot as plt

from plot_emg_person_theta import (
    CONFIG_PATH,
    MUSCLE_ORDER,
    load_muscle_config,
    plot_person,
)

PERSON = "max"


def main():
    muscle_config = load_muscle_config(CONFIG_PATH)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    plot_person(axes[0], axes[1], PERSON, muscle_config)

    fig.suptitle(f"Simulated Whole-Arm Motor Activation — {PERSON.capitalize()}", fontsize=14, y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.955))

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "emg_max_theta.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
