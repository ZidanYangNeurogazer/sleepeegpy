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
"""

import os
import mne
import numpy as np
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
# Input EEG header file. MNE reads the .eeg and .vmrk siblings automatically.
PATH_TO_EEG = r"Y:\Nap\data\rawdata\sub-DUHR006\eeg\sub-DUHR006_task-nap_eeg.vhdr"

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


def plot_per_channel_spectrograms(spectral_pipe, out_dir):
    """Save one hypnogram+spectrogram figure per EEG channel.

    The hypnogram on top is the single predicted hypnogram (same for every
    figure); only the spectrogram below changes per channel. Files are named
    spectrogram_<channel>.png.
    """
    save_dir = os.path.join(out_dir, "spectrograms_per_channel")
    os.makedirs(save_dir, exist_ok=True)
    # EEG channels only (excludes the EOG-type channel).
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
    """Save one event-average figure per channel (not pooled across channels).

    Reuses the already-computed detection results (event_pipe.results) and
    splits the per-event waveforms by channel, so detection is NOT re-run.
    Each figure shows the average waveform per sleep stage for that channel.
    Files are named <subdir>_<channel>.png.
    """
    import seaborn as sns

    if not getattr(event_pipe, "results", None):
        print(f"No detection results for {subdir}; skipping per-channel plots.")
        return

    stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    df = event_pipe.results.get_sync_events(
        center=center, time_before=time_before, time_after=time_after
    )
    if "Stage" in df:
        df = df.copy()
        df["Stage"] = df["Stage"].map(stage_names).fillna(df["Stage"])

    save_dir = os.path.join(out_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    for ch, df_ch in df.groupby("Channel"):
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_ch, x="Time", y="Amplitude",
            hue="Stage" if "Stage" in df_ch else None, ax=ax,
        )
        ax.set_title(ch)
        ax.set_ylabel("Amplitude (uV)")
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
    bad_channels_path = pipe.auto_detect_bad_channels()  # writes bad_channels.txt
    pipe.read_bad_channels(path=bad_channels_path)
    print(f"Auto-detected bad channels: {pipe.mne_raw.info['bads']}")
    pipe.interpolate_bads(reset_bads=True)

    # ---------------------------------------------------------------- #
    # 3) Automatic bad-epoch annotations (amplitude-based)
    # ---------------------------------------------------------------- #
    pipe.auto_set_annotations()
    print(f"Bad data after auto-annotation: {pipe.bad_data_percent}%")

    # Save the cleaned continuous data.
    pipe.save_raw("cleaned_raw.fif", overwrite=True)

    # ---------------------------------------------------------------- #
    # 4) Optional ICA artifact removal
    # ---------------------------------------------------------------- #
    prec = pipe
    if RUN_ICA:
        ica_pipe = ICAPipe(prec_pipe=pipe, n_components=N_ICA_COMPONENTS)
        ica_pipe.fit()
        # Without manual inspection, exclude nothing automatically here.
        # Inspect with ica_pipe.plot_sources() / plot_components() and set
        # ica_pipe.mne_ica.exclude = [...] before apply() if desired.
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
        save=True,  # writes predicted_hypno.txt + probability figure
    )
    spectral_pipe.sleep_stats(save=True)  # writes sleep_stats.csv

    # ---------------------------------------------------------------- #
    # 6) Optional spectral analysis
    # ---------------------------------------------------------------- #
    if RUN_SPECTRAL:
        spectral_pipe.plot_hypnospectrogram(picks=[PICKED_CHANNEL], save=True)
        # One spectrogram figure per individual EEG channel.
        plot_per_channel_spectrograms(spectral_pipe, OUTPUT_DIR)
        spectral_pipe.compute_psd(
            sleep_stages={"Wake": 0, "N1": 1, "N2/3": (2, 3), "REM": 4},
            reference="average",
            n_fft=1024,
            n_per_seg=1024,
            n_overlap=512,
            window="hamming",
            verbose=False,
        )
        spectral_pipe.plot_psds(picks=[PICKED_CHANNEL], psd_range=(-30, 30), save=True)
        spectral_pipe.plot_topomap_collage(save=True)

    # ---------------------------------------------------------------- #
    # 7) Optional event detection (spindles + slow waves)
    #    NOTE: REM detection is skipped - it needs two EOG channels
    #    (LOC + ROC) and you only have one.
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

        slow_waves_pipe = SlowWavesPipe(prec_pipe=spindles_pipe)
        slow_waves_pipe.detect(save=True)
        slow_waves_pipe.plot_average(
            center="NegPeak", hue="Stage", time_before=0.4, time_after=0.8, save=True
        )
        plot_per_channel_event_average(
            slow_waves_pipe, OUTPUT_DIR, "slowwaves_per_channel",
            center="NegPeak", time_before=0.4, time_after=0.8,
        )

    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
