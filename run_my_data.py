"""
Run sleepeegpy on Duke EEG data (BrainVision .vhdr/.eeg/.vmrk).

This script is adapted from notebooks/0_complete_pipeline.ipynb for a dataset that:
  * has 64 standard 10-20 channels (so we attach a standard montage),
  * has ONE EOG channel called "EOG" with no location (marked as an EOG-type
    channel: excluded from EEG steps, but still used for sleep staging),
  * has NO EMG channel,
  * has NO hypnogram, NO bad-channels file, NO annotations file
    (so all three are generated automatically).

Edit the CONFIG block below, then run:  python run_my_data.py

Quantification outputs (per electrode × 5 sleep stages):
  - spindle_counts.csv / spindle_counts_heatmap.png
  - theta_power.csv   / theta_power_heatmap.png
  - slow_wave_density.csv / slow_wave_density_heatmap.png
"""

import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sleepeegpy.pipeline import (
    CleaningPipe,
    ICAPipe,
    SpectralPipe,
    SpindlesPipe,
    SlowWavesPipe,
)

# ============================ CONFIG (edit me) ============================ #
# To switch subjects, change these two lines:
SUBJECT_CODE = "DUHR006"
# Input EEG header file — derived automatically from SUBJECT_CODE.
PATH_TO_EEG = rf"Y:\Nap\data\rawdata\sub-{SUBJECT_CODE}\eeg\sub-{SUBJECT_CODE}_task-nap_eeg.vhdr"

# Outputs go into a per-subject folder: <OUTPUT_BASE_DIR>\<SUBJECT_CODE>\
OUTPUT_BASE_DIR = r"C:\Users\zy248\Box\Suthana_Lab\Zidan\NapRecording"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, SUBJECT_CODE)

# --- Channel roles ---------------------------------------------------------
EOG_CHANNEL = "EOG"          # the single EOG channel in your file
MONTAGE = "standard_1005"    # covers your extended names (FTT9h, TPP9h, ...)

# --- Preprocessing ---------------------------------------------------------
RESAMPLE_SFREQ = 250         # downsample target (raw is 1000-2000 Hz)
FILTER_L_FREQ = 0.75         # high-pass (Hz)
FILTER_H_FREQ = 40           # low-pass (Hz)
NOTCH = "60s"                # 60 Hz + harmonics (US line noise). Use "50s" in EU.

# --- Automatic sleep staging (YASA) ---------------------------------------
# YASA works best with a central EEG referenced to a contralateral mastoid.
# You have no mastoid electrode, so we use TPP9h (left, near-mastoid) as a proxy.
HYPNO_EEG_NAME = "C3"        # central EEG channel
HYPNO_EOG_NAME = EOG_CHANNEL
HYPNO_EMG_NAME = None        # no EMG available
HYPNO_REF_NAME = "TPP9h"     # left near-mastoid reference proxy

# Channel used for hypnogram/spectrogram/PSD display.
PICKED_CHANNEL = "Cz"

# Sleep stages used throughout (all 5 stages, N2 and N3 kept separate for quantification)
SLEEP_STAGES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
STAGE_ORDER  = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_MAP    = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# --- What to run -----------------------------------------------------------
RUN_ICA = False              # set True to run ICA artifact removal (interactive-ish)
N_ICA_COMPONENTS = 20
RUN_SPECTRAL = True          # PSD + spectrogram + topomaps
RUN_EVENTS = True            # spindles + slow waves detection
# ========================================================================= #


def attach_montage_and_types(raw):
    """Mark EOG as an EOG-type channel and attach the electrode montage.

    Marking EOG as 'eog' keeps it out of EEG-only steps (bad-channel
    detection, interpolation, average reference, PSD) while remaining
    available by name for YASA sleep staging.
    """
    if EOG_CHANNEL in raw.ch_names:
        raw.set_channel_types({EOG_CHANNEL: "eog"})
    montage = mne.channels.make_standard_montage(MONTAGE)
    # on_missing='ignore' so non-EEG channels without positions don't error.
    raw.set_montage(montage, on_missing="ignore")
    return raw


def _save_heatmap(df, path, title, fmt=None, cmap="YlOrRd", annot=True):
    """Save a channel × stage heatmap. Annotations are skipped for large channel sets."""
    n_ch = len(df)
    fig_h = max(6, n_ch * 0.28)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    do_annot = annot and n_ch <= 40  # skip annotation text when too many rows
    kwargs = dict(fmt=fmt) if (fmt and do_annot) else {}
    sns.heatmap(df, annot=do_annot, cmap=cmap, ax=ax, linewidths=0.3, **kwargs)
    ax.set_title(title)
    ax.set_ylabel("Channel")
    ax.set_xlabel("Sleep Stage")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def quantify_spindles(spindles_pipe, out_dir):
    """Spindle count per channel per sleep stage → CSV + heatmap."""
    if not getattr(spindles_pipe, "results", None):
        print("No spindle results; skipping spindle quantification.")
        return

    events = spindles_pipe.results._events.copy()
    events["Stage"] = events["Stage"].map(STAGE_MAP)
    counts = (
        events.groupby(["Channel", "Stage"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STAGE_ORDER, fill_value=0)
    )

    csv_path = os.path.join(out_dir, "spindle_counts.csv")
    counts.to_csv(csv_path)

    _save_heatmap(
        counts,
        path=os.path.join(out_dir, "spindle_counts_heatmap.png"),
        title=f"Spindle Count per Channel per Stage — {SUBJECT_CODE}",
        fmt="d",
        cmap="YlOrRd",
    )
    print(f"Spindle counts → {csv_path}")


def quantify_theta_power(spectral_pipe, out_dir):
    """Mean theta power (4–8 Hz) per channel per sleep stage → CSV + heatmap.

    Uses the already-computed PSDs from spectral_pipe.psds.
    Power is in µV²/Hz (linear scale).
    """
    THETA_LOW, THETA_HIGH = 4.0, 8.0

    records = {}
    ch_names = None
    for stage, spectrum in spectral_pipe.psds.items():
        if stage not in STAGE_ORDER:
            continue
        psd, freqs = spectrum.get_data(return_freqs=True)   # (n_ch, n_freq)
        mask = (freqs >= THETA_LOW) & (freqs <= THETA_HIGH)
        records[stage] = psd[:, mask].mean(axis=1)
        if ch_names is None:
            ch_names = spectrum.ch_names

    if not records:
        print("No PSD data found; skipping theta power quantification.")
        return

    df = pd.DataFrame(records, index=ch_names).reindex(columns=STAGE_ORDER)

    csv_path = os.path.join(out_dir, "theta_power.csv")
    df.to_csv(csv_path)

    _save_heatmap(
        df,
        path=os.path.join(out_dir, "theta_power_heatmap.png"),
        title=f"Mean Theta Power 4–8 Hz (µV²/Hz) per Channel per Stage — {SUBJECT_CODE}",
        cmap="viridis",
        annot=False,  # continuous values — colorbar is more readable
    )
    print(f"Theta power → {csv_path}")


def quantify_slow_waves(sw_pipe, spectral_pipe, out_dir):
    """Slow wave density (events/min of stage) per channel per sleep stage → CSV + heatmap."""
    if not getattr(sw_pipe, "results", None):
        print("No slow wave results; skipping slow wave quantification.")
        return

    # Stage durations in minutes from the 30-s epoch hypnogram.
    hypno = spectral_pipe.hypno
    stage_dur_min = {
        name: float(np.sum(hypno == code)) * 30.0 / 60.0
        for code, name in STAGE_MAP.items()
    }

    events = sw_pipe.results._events.copy()
    events["Stage"] = events["Stage"].map(STAGE_MAP)
    counts = (
        events.groupby(["Channel", "Stage"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STAGE_ORDER, fill_value=0)
    ).astype(float)

    # Convert raw counts → density (events per minute of that stage).
    for stage, dur_min in stage_dur_min.items():
        if stage in counts.columns and dur_min > 0:
            counts[stage] /= dur_min

    csv_path = os.path.join(out_dir, "slow_wave_density.csv")
    counts.to_csv(csv_path)

    _save_heatmap(
        counts,
        path=os.path.join(out_dir, "slow_wave_density_heatmap.png"),
        title=f"Slow Wave Density (events/min) per Channel per Stage — {SUBJECT_CODE}",
        fmt=".2f",
        cmap="Blues",
    )
    print(f"Slow wave density → {csv_path}")


def plot_per_channel_spectrograms(spectral_pipe, out_dir):
    """Save one hypnogram+spectrogram figure per EEG channel."""
    save_dir = os.path.join(out_dir, "spectrograms_per_channel")
    os.makedirs(save_dir, exist_ok=True)
    eeg_channels = spectral_pipe.mne_raw.copy().pick("eeg").ch_names
    for ch in eeg_channels:
        spectral_pipe.plot_hypnospectrogram(picks=[ch], save=False)
        fig = plt.gcf()
        fig.suptitle(ch)
        fig.savefig(os.path.join(save_dir, f"spectrogram_{ch}.png"),
                    bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(eeg_channels)} per-channel spectrograms to {save_dir}")


def plot_per_channel_event_average(event_pipe, out_dir, subdir, center,
                                   time_before, time_after):
    """Save one event-average figure per channel."""
    if not getattr(event_pipe, "results", None):
        print(f"No detection results for {subdir}; skipping per-channel plots.")
        return

    df = event_pipe.results.get_sync_events(
        center=center, time_before=time_before, time_after=time_after
    )
    if "Stage" in df.columns:
        df = df.copy()
        df["Stage"] = df["Stage"].map(STAGE_MAP).fillna(df["Stage"])

    save_dir = os.path.join(out_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    for ch, df_ch in df.groupby("Channel"):
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_ch, x="Time", y="Amplitude",
            hue="Stage" if "Stage" in df_ch else None, ax=ax,
        )
        ax.set_title(ch)
        ax.set_ylabel("Amplitude (µV)")
        fig.savefig(os.path.join(save_dir, f"{subdir}_{ch}.png"),
                    bbox_inches="tight")
        plt.close(fig)
    print(f"Saved per-channel plots for {len(df['Channel'].unique())} "
          f"channels to {save_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1) Cleaning: load -> montage/types -> resample -> filter -> notch
    # ---------------------------------------------------------------- #
    pipe = CleaningPipe(path_to_eeg=PATH_TO_EEG, output_dir=OUTPUT_DIR)
    attach_montage_and_types(pipe.mne_raw)

    pipe.resample(sfreq=RESAMPLE_SFREQ)
    pipe.filter(l_freq=FILTER_L_FREQ, h_freq=FILTER_H_FREQ)
    pipe.notch(freqs=NOTCH)

    # ---------------------------------------------------------------- #
    # 2) Automatic bad-channel detection (pyprep) + interpolation
    # ---------------------------------------------------------------- #
    bad_channels_path = pipe.auto_detect_bad_channels()
    pipe.read_bad_channels(path=bad_channels_path)
    print(f"Auto-detected bad channels: {pipe.mne_raw.info['bads']}")
    pipe.interpolate_bads(reset_bads=True)

    # ---------------------------------------------------------------- #
    # 3) Automatic bad-epoch annotations (amplitude-based)
    # ---------------------------------------------------------------- #
    pipe.auto_set_annotations()
    print(f"Bad data after auto-annotation: {pipe.bad_data_percent}%")

    pipe.save_raw("cleaned_raw.fif", overwrite=True)

    # ---------------------------------------------------------------- #
    # 4) Optional ICA artifact removal
    # ---------------------------------------------------------------- #
    prec = pipe
    if RUN_ICA:
        ica_pipe = ICAPipe(prec_pipe=pipe, n_components=N_ICA_COMPONENTS)
        ica_pipe.fit()
        ica_pipe.apply()
        prec = ica_pipe

    # ---------------------------------------------------------------- #
    # 5) Automatic sleep staging (YASA) -> predicted hypnogram
    # ---------------------------------------------------------------- #
    spectral_pipe = SpectralPipe(prec_pipe=prec, path_to_hypno=None)
    spectral_pipe.predict_hypno(
        eeg_name=HYPNO_EEG_NAME,
        eog_name=HYPNO_EOG_NAME,
        emg_name=HYPNO_EMG_NAME,
        ref_name=HYPNO_REF_NAME,
        save=True,
    )
    spectral_pipe.sleep_stats(save=True)

    # ---------------------------------------------------------------- #
    # 6) Spectral analysis (N2 and N3 kept separate for theta quantification)
    # ---------------------------------------------------------------- #
    if RUN_SPECTRAL:
        spectral_pipe.plot_hypnospectrogram(picks=[PICKED_CHANNEL], save=True)
        plot_per_channel_spectrograms(spectral_pipe, OUTPUT_DIR)
        spectral_pipe.compute_psd(
            sleep_stages=SLEEP_STAGES,
            reference="average",
            n_fft=1024,
            n_per_seg=1024,
            n_overlap=512,
            window="hamming",
            verbose=False,
        )
        spectral_pipe.plot_psds(picks=[PICKED_CHANNEL], psd_range=(-30, 30), save=True)
        spectral_pipe.plot_topomap_collage(save=True)

        # --- Theta power quantification (requires PSDs to be computed first) ---
        quantify_theta_power(spectral_pipe, OUTPUT_DIR)

    # ---------------------------------------------------------------- #
    # 7) Event detection (spindles + slow waves) + quantification
    #    NOTE: REM detection skipped — needs two EOG channels (LOC + ROC).
    # ---------------------------------------------------------------- #
    if RUN_EVENTS:
        spindles_pipe = SpindlesPipe(prec_pipe=spectral_pipe)
        spindles_pipe.detect(save=True)
        spindles_pipe.plot_average(
            center="Peak", hue="Stage", time_before=1, time_after=1, save=True
        )
        plot_per_channel_event_average(
            spindles_pipe, OUTPUT_DIR, "spindles_per_channel",
            center="Peak", time_before=1, time_after=1,
        )
        quantify_spindles(spindles_pipe, OUTPUT_DIR)

        slow_waves_pipe = SlowWavesPipe(prec_pipe=spindles_pipe)
        slow_waves_pipe.detect(save=True)
        slow_waves_pipe.plot_average(
            center="NegPeak", hue="Stage", time_before=0.4, time_after=0.8, save=True
        )
        plot_per_channel_event_average(
            slow_waves_pipe, OUTPUT_DIR, "slowwaves_per_channel",
            center="NegPeak", time_before=0.4, time_after=0.8,
        )
        quantify_slow_waves(slow_waves_pipe, spectral_pipe, OUTPUT_DIR)

    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
