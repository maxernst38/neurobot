"""
Plot a single EMG CSV (columns Time_s, ADC).

Usage:
    python scripts/plot_emg_single.py <csv_file>
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR = "#2a78d6"
RMS_COLOR = "#eb6834"
RMS_WINDOW_S = 0.08  # sliding window used to compute the RMS envelope


def main():
    if len(sys.argv) != 2:
        print("Usage: python graph_emg_single.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # time_s is rounded to 10ms, so consecutive diffs are frequently 0 (many rows
    # share a timestamp); the mean spacing over the whole recording is a more
    # reliable estimate of the true per-sample interval than a diff-based one.
    dt = (df["time_s"].iloc[-1] - df["time_s"].iloc[0]) / max(len(df) - 1, 1)
    window = max(1, int(round(RMS_WINDOW_S / dt))) if dt > 0 else 1
    rms = (df["adc"] ** 2).rolling(window, center=True, min_periods=1).mean() ** 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["time_s"], df["adc"], color=COLOR, linewidth=1.0, alpha=0.5, label="Raw")
    ax.plot(df["time_s"], rms, color=RMS_COLOR, linewidth=2.0, label=f"RMS ({RMS_WINDOW_S * 1000:.0f}ms window)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (ADC)")
    ax.set_title(os.path.basename(csv_path))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path = os.path.splitext(csv_path)[0] + "_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
