"""
BCI Alpha Power Calculator
==========================
Computes mean alpha power (µV²) for occipital channels O1, Oz, O2
from a raw OpenBCI CSV file exported by BrainFlow.

YOUR DATA FORMAT: Raw ADC counts (ADS1299 24-bit)
  - The CSV contains raw integer ADC values, NOT microvolts or nanovolts
  - Must apply ADS1299 formula to convert to µV before any analysis

Pipeline (per channel):
  Step 1 — Load CSV
  Step 2 — Convert ADC counts → µV  (ADS1299 formula)
  Step 3 — Slice the 30-second trial window (7500 samples)
  Step 4 — Remove DC offset (subtract mean)
  Step 5 — Bandpass filter 8–13 Hz (4th-order zero-phase Butterworth)
  Step 6 — Compute mean squared amplitude → alpha power in µV²
  Step 7 — Average across O1, Oz, O2

Requirements:
  pip install numpy scipy pandas
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these to match your recording
# ─────────────────────────────────────────────────────────────────────────────

CSV_FILE     = "Trial__1_Open_Eye_12.csv"  # your OpenBCI BrainFlow CSV

FS           = 250        # sample rate (Hz) — OpenBCI Mark IV default
ALPHA_LOW    = 8.0        # alpha band lower cutoff (Hz)
ALPHA_HIGH   = 13.0       # alpha band upper cutoff (Hz)
FILTER_ORDER = 4          # Butterworth filter order

TRIAL_START  = 0.0        # trial start time in the recording (seconds)
TRIAL_DUR    = 30.0       # trial duration (seconds)

# ADS1299 ADC → µV conversion constants (OpenBCI Mark IV)
VREF         = 4.5        # reference voltage (volts)
ADC_BITS     = 24         # 24-bit ADC
GAIN         = 24         # OpenBCI default gain

# Column indices for O1, Oz, O2 in your CSV
# Check the "Column reference" table below after running to confirm
COL_O1       = 2          # O1 — adjust if needed
COL_OZ       = 5          # Oz — adjust if needed
COL_O2       = 7          # O2 — adjust if needed

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath):
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            try:
                vals = [float(x) for x in parts]
                rows.append(vals)
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric data found in {filepath}")
    max_cols = max(len(r) for r in rows)
    padded   = [r + [np.nan] * (max_cols - len(r)) for r in rows]
    return np.array(padded)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CONVERT ADC COUNTS → µV
# ─────────────────────────────────────────────────────────────────────────────

def adc_to_uv(adc_signal, vref=VREF, bits=ADC_BITS, gain=GAIN):
    """
    Convert raw ADS1299 ADC counts to microvolts.

    Formula:
        µV = ADC_count × (Vref / (2^(bits-1) - 1)) / gain × 1,000,000

    For OpenBCI Mark IV default settings (Vref=4.5V, gain=24, 24-bit):
        scale = (4.5 / 8,388,607) / 24 × 1,000,000 = 0.022352 µV/count
    """
    scale = (vref / (2 ** (bits - 1) - 1)) / gain * 1e6
    return adc_signal * scale

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SLICE TRIAL WINDOW
# ─────────────────────────────────────────────────────────────────────────────

def slice_window(signal, start_s, dur_s, fs):
    start_n = int(start_s * fs)
    end_n   = min(int((start_s + dur_s) * fs), len(signal))
    return signal[start_n:end_n]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — REMOVE DC OFFSET
# ─────────────────────────────────────────────────────────────────────────────

def remove_dc(signal):
    """
    Subtract the mean amplitude of the window to remove DC offset.

    Without this step, the large constant voltage from electrode-skin
    contact would dominate and produce unrealistically high power values.
    """
    dc = np.mean(signal)
    return signal - dc, dc

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BANDPASS FILTER 8–13 Hz
# ─────────────────────────────────────────────────────────────────────────────

def design_filter(lo, hi, fs, order=4):
    """
    4th-order zero-phase Butterworth bandpass filter.
    SOS (second-order sections) format for numerical stability.
    """
    nyq = fs / 2.0
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sos

def apply_filter(signal, sos):
    """
    sosfiltfilt = forward pass then reverse pass → zero phase shift.
    """
    return sosfiltfilt(sos, signal)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MEAN ALPHA POWER
# ─────────────────────────────────────────────────────────────────────────────

def mean_alpha_power(filtered_signal):
    """
    P_alpha = (1/N) × Σ x_filtered(i)²

    Mean squared amplitude of the bandpass-filtered signal.
    Equal to signal variance (since filtered signal has zero mean).
    Unit: µV²
    """
    return float(np.mean(filtered_signal ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE — ONE CHANNEL
# ─────────────────────────────────────────────────────────────────────────────

def process_channel(data, col_idx, ch_name, sos, verbose=True):
    adc_raw  = data[:, col_idx]
    adc_raw  = adc_raw[~np.isnan(adc_raw)]

    # Step 2: ADC → µV
    uv_full  = adc_to_uv(adc_raw)

    # Step 3: slice window
    window   = slice_window(uv_full, TRIAL_START, TRIAL_DUR, FS)
    n        = len(window)

    # Step 4: remove DC
    centered, dc = remove_dc(window)

    # Step 5: filter
    filtered = apply_filter(centered, sos)

    # Step 6: power
    power    = mean_alpha_power(filtered)

    if verbose:
        scale = (VREF / (2**(ADC_BITS-1) - 1)) / GAIN * 1e6
        print(f"\n  {ch_name} (column {col_idx})")
        print(f"    ADC counts   : mean = {np.mean(adc_raw):.1f}  std = {np.std(adc_raw):.1f}")
        print(f"    ADC → µV     : scale = {scale:.6f} µV/count")
        print(f"    µV signal    : mean = {np.mean(uv_full):.2f} µV  std = {np.std(uv_full):.2f} µV")
        print(f"    Window       : {TRIAL_START}s – {TRIAL_START+TRIAL_DUR}s  ({n} samples)")
        print(f"    DC removed   : offset = {dc:.2f} µV")
        print(f"    Filtered std : {np.std(filtered):.4f} µV  (8–13 Hz only)")
        print(f"    Alpha power  : {power:.4f} µV²")

    return power

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 62

    print(sep)
    print("  BCI Alpha Power Calculator")
    print(sep)
    print(f"  File         : {CSV_FILE}")
    print(f"  Sample rate  : {FS} Hz")
    print(f"  Trial window : {TRIAL_START}s – {TRIAL_START+TRIAL_DUR}s  ({int(TRIAL_DUR*FS)} samples)")
    print(f"  Alpha band   : {ALPHA_LOW}–{ALPHA_HIGH} Hz")
    print(f"  Columns      : O1=col{COL_O1}  Oz=col{COL_OZ}  O2=col{COL_O2}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 1 — Loading CSV")
    data = load_csv(CSV_FILE)
    n_rows, n_cols = data.shape
    rec_sec = n_rows / FS
    print(f"    {n_rows:,} rows × {n_cols} columns  |  {rec_sec:.1f}s at {FS} Hz")

    # ── ADC conversion info ───────────────────────────────────────────────────
    scale = (VREF / (2**(ADC_BITS-1) - 1)) / GAIN * 1e6
    print(f"\n{'─'*62}")
    print("  STEP 2 — ADS1299 ADC → µV conversion")
    print(f"    Formula : µV = ADC × (Vref / (2²³-1)) / gain × 1,000,000")
    print(f"    Values  : Vref={VREF}V  bits={ADC_BITS}  gain={GAIN}")
    print(f"    Scale   : {scale:.6f} µV per ADC count")

    # ── Design filter ─────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 5 — Bandpass filter design")
    sos = design_filter(ALPHA_LOW, ALPHA_HIGH, FS, FILTER_ORDER)
    print(f"    {FILTER_ORDER}th-order Butterworth  |  {ALPHA_LOW}–{ALPHA_HIGH} Hz  |  zero-phase")

    # ── Process channels ──────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEPS 3–6 — Process each channel")

    p_o1 = process_channel(data, COL_O1, "O1", sos)
    p_oz  = process_channel(data, COL_OZ,  "Oz",  sos)
    p_o2 = process_channel(data, COL_O2, "O2", sos)

    # Step 7: average
    avg = (p_o1 + p_oz + p_o2) / 3

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESULTS")
    print(sep)
    print(f"  {'Channel':<22} {'Alpha Power (µV²)':>18}")
    print(f"  {'─'*40}")
    print(f"  {'O1':<22} {p_o1:>18.4f}")
    print(f"  {'Oz':<22} {p_oz:>18.4f}")
    print(f"  {'O2':<22} {p_o2:>18.4f}")
    print(f"  {'─'*40}")
    print(f"  {'Average (O1+Oz+O2)/3':<22} {avg:>18.4f}  µV²")
    print(sep)
    print(f"\n  Samples   : {int(TRIAL_DUR*FS):,}  ({int(TRIAL_DUR)}s × {FS} Hz)")
    print(f"  Filter    : {FILTER_ORDER}th-order Butterworth  {ALPHA_LOW}–{ALPHA_HIGH} Hz  zero-phase")
    print(f"  Unit conv : ADC counts × {scale:.6f} = µV")
    print(f"  Power     : mean squared amplitude of filtered signal")
    print(f"  Unit      : µV²")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = CSV_FILE.replace(".csv", "_alpha_power.csv")
    pd.DataFrame([{
        "file":           CSV_FILE,
        "trial_start_s":  TRIAL_START,
        "trial_end_s":    TRIAL_START + TRIAL_DUR,
        "n_samples":      int(TRIAL_DUR * FS),
        "fs_hz":          FS,
        "alpha_band_hz":  f"{ALPHA_LOW}–{ALPHA_HIGH}",
        "adc_scale_uV":   round(scale, 8),
        "O1_uV2":         round(p_o1,  4),
        "Oz_uV2":         round(p_oz,   4),
        "O2_uV2":         round(p_o2,  4),
        "avg_uV2":        round(avg,   4),
    }]).to_csv(out, index=False)
    print(f"\n  Saved to  : {out}\n")


if __name__ == "__main__":
    main()
