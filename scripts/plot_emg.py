"""
Plot EMG amplitude (ADC) vs time for all 6 muscle groups.

Each muscle group's CSVs live in data/emg/<group>/*.csv with columns
Time_s, ADC. Each sample is independently pruned down to the window
around its first spike (detected as the first sustained rise above its
own baseline noise) and re-centered so that spike's peak sits at t=0.
That lets samples from the same muscle group visually overlap despite
firing at different absolute times in the 5s recording. Every sample is
drawn as a thin, transparent line in the group's color; the average
across a group's aligned samples is drawn as a thick, opaque line in the
same color.

Usage:
    python scripts/plot_emg.py
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "emg")

# Display name + fixed categorical color per muscle group (folder name -> (label, color)).
# Order here controls subplot layout order (left-to-right, top-to-bottom).
MUSCLE_GROUPS = {
    "shoulder": ("Shoulder Pitch", "#008300"),
    "elbow_flexion": ("Elbow Flexion (Bicep)", "#2a78d6"),
    "elbow_extension": ("Elbow Extension (Tricep)", "#eb6834"),
    "wrist_flexion": ("Wrist Flexion (Flexor)", "#1baf7a"),
    "wrist_extension": ("Wrist Extension (Extensor)", "#eda100"),
    "grip": ("Grip", "#e87ba4"),
}

SMOOTH_WINDOW_S = 0.08  # envelope smoothing window, in seconds, for spike detection
SPIKE_THRESHOLD_SD = 4.0  # multiples of baseline noise above baseline to call a "spike"
SPIKE_THRESHOLD_RATIO = 1.5  # spike must also clear this multiple of the baseline itself
MIN_SPIKE_DURATION_S = 0.25  # a rise must hold this long to count as a spike, not noise

PRE_PEAK_S = 1.0  # how much time before the peak to keep
POST_PEAK_S = 1.5  # how much time after the peak to keep
GRID_POINTS = 500  # ~5ms spacing over the 2.5s aligned window


def load_samples(group_dir):
    """Read every CSV in a muscle-group folder into a list of (time_s, adc) arrays."""
    samples = []
    for path in sorted(glob.glob(os.path.join(group_dir, "*.csv"))):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        t = df["time_s"].to_numpy(dtype=float)
        adc = df["adc"].to_numpy(dtype=float)
        order = np.argsort(t, kind="stable")
        samples.append((t[order], adc[order], os.path.basename(path)))
    return samples


def moving_average(x, window):
    if window < 2:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def runs_above(mask):
    """Return (start_idx, end_idx) for each contiguous run of True in mask."""
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


def find_first_spike_peak(t, adc):
    """Return the time of the peak within the first sustained spike, or None."""
    # time_s is rounded to 10ms, so consecutive diffs are frequently 0 (many rows
    # share a timestamp); the mean spacing over the whole recording is a more
    # reliable estimate of the true per-sample interval than a diff-based one.
    dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    dt = dt if dt > 0 else 0.001
    window = max(1, int(round(SMOOTH_WINDOW_S / dt)))
    envelope = moving_average(adc, window)

    baseline = np.percentile(envelope, 20)
    calm = envelope[envelope <= np.percentile(envelope, 50)]
    noise_sd = calm.std() if calm.size else envelope.std()
    threshold = max(baseline + SPIKE_THRESHOLD_SD * max(noise_sd, 1e-9), baseline * SPIKE_THRESHOLD_RATIO)

    for start_idx, end_idx in runs_above(envelope > threshold):
        if t[end_idx] - t[start_idx] >= MIN_SPIKE_DURATION_S:
            peak_idx = start_idx + int(np.argmax(adc[start_idx:end_idx + 1]))
            return t[peak_idx]
    return None


def align_to_first_spike(samples):
    """Crop + shift each sample to [-PRE_PEAK_S, POST_PEAK_S] around its first spike peak."""
    aligned = []
    for t, adc, name in samples:
        peak_time = find_first_spike_peak(t, adc)
        if peak_time is None:
            print(f"  no spike detected in {name}, skipping")
            continue
        t_rel = t - peak_time
        mask = (t_rel >= -PRE_PEAK_S) & (t_rel <= POST_PEAK_S)
        aligned.append((t_rel[mask], adc[mask], name))
    return aligned


def average_on_grid(aligned, t_grid):
    """Interpolate each aligned sample onto a shared relative-time grid and average,
    ignoring points outside a given sample's own aligned range."""
    stacked = np.vstack([
        np.interp(t_grid, t_rel, adc, left=np.nan, right=np.nan)
        for t_rel, adc, _name in aligned
    ])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def describe_spike(t_grid, avg_adc):
    """Peak amplitude of the averaged curve, and the [start, end] interval (relative
    to the peak, which sits at t=0) of the most significant spike around it."""
    peak_idx = int(np.nanargmax(avg_adc))
    peak_val = avg_adc[peak_idx]

    baseline = np.nanpercentile(avg_adc, 20)
    calm = avg_adc[avg_adc <= np.nanpercentile(avg_adc, 50)]
    noise_sd = np.nanstd(calm) if calm.size else np.nanstd(avg_adc)
    threshold = max(baseline + SPIKE_THRESHOLD_SD * max(noise_sd, 1e-9), baseline * SPIKE_THRESHOLD_RATIO)

    mask = np.nan_to_num(avg_adc, nan=-np.inf) > threshold
    for start_idx, end_idx in runs_above(mask):
        if start_idx <= peak_idx <= end_idx:
            return peak_val, t_grid[start_idx], t_grid[end_idx]
    return peak_val, t_grid[peak_idx], t_grid[peak_idx]


def plot_group(ax, group_key, label, color):
    group_dir = os.path.join(DATA_DIR, group_key)
    samples = load_samples(group_dir)
    if not samples:
        print(f"Skipping {group_key}: no CSV files found in {group_dir}")
        return None

    print(f"{label}:")
    aligned = align_to_first_spike(samples)
    if not aligned:
        print(f"  no samples with a detectable spike, skipping")
        return None

    for t_rel, adc, _name in aligned:
        ax.plot(t_rel, adc, color=color, alpha=0.25, linewidth=1.0)

    t_grid = np.linspace(-PRE_PEAK_S, POST_PEAK_S, GRID_POINTS)
    avg_adc = average_on_grid(aligned, t_grid)
    ax.plot(t_grid, avg_adc, color=color, alpha=1.0, linewidth=2.5)
    ax.axvline(0.0, color=color, alpha=0.3, linewidth=1.0, linestyle="--")

    peak_val, spike_start, spike_end = describe_spike(t_grid, avg_adc)
    ax.text(
        0.98, 0.95,
        f"Peak: {peak_val:.0f} ADC\nSpike length: {spike_end - spike_start:.2f}s",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#333333",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )

    ax.set_xticks([])
    ax.set_ylabel("Amplitude (ADC)")
    ax.set_title(label, color=color, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    return {
        "muscle_group": group_key,
        "label": label,
        "peak_amplitude_adc": peak_val,
        "spike_start_s": spike_start,
        "spike_end_s": spike_end,
        "spike_length_s": spike_end - spike_start,
    }


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    summary_rows = []
    for ax, (group_key, (label, color)) in zip(axes.flat, MUSCLE_GROUPS.items()):
        row = plot_group(ax, group_key, label, color)
        if row is not None:
            summary_rows.append(row)

    fig.suptitle("EMG Signal by Muscle Group (samples aligned on first-spike peak)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "emg_plot.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "emg_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    plt.show()


if __name__ == "__main__":
    main()
