# Phase 2: EEG Signal Quality Analysis and Visualization

This document describes the methodology, scientific safeguards, generated artifacts, and exact execution commands for the Phase 2 EEG signal-quality analysis pipeline (`imagined_speech.data.signal_quality`).

---

## 1. Purpose and Scope

Before designing any filtering, epoching, baseline correction, or machine-learning pipeline, Phase 2 inspects the raw EDF recordings discovered in Phase 1 to answer:

1. **Channel Consistency**: Which configured EEG channels (`AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`) are present in each recording, which are missing, and which non-EEG or unrecognized channels exist?
2. **Sampling Rate & Duration Variability**: What are the actual sampling frequencies (`sfreq_hz`), sample counts (`n_samples`), and recording durations (`duration_sec`) across `Char`, `Digit`, and `Image`?
3. **Physical Units & Amplitude Distributions**: What physical units (`uV`, `mV`, etc.) are declared in the binary EDF headers, what are the measured distributions in those native units, and are there large DC offsets or uncalibrated unit strings?
4. **Signal Integrity Indicators**: Are there non-finite samples (`NaN`/`Inf`), flatline or near-flatline traces, rail saturation / potential clipping, wrapping sample counters, or suspicious repeated/quantized sample runs?

---

## 2. Scientific Safeguards

- **Raw Signal Preservation**: Signals are read in-memory in their original EDF physical units. No bandpass filtering, notch filtering, resampling, ICA, baseline correction, normalization, or artifact removal is applied.
- **No Unverified Timing Assumptions**: Imagined-speech onset/offset intervals are not inferred without verified protocol documentation or event markers.
- **No Unverified Participant Assumptions**: Filename prefixes are retained strictly as `unverified_candidate_prefix` and are not assumed to represent human subject identifiers.
- **Measured Values vs. Heuristic Flags**:
  - Columns prefixed with `measured_*` report direct mathematical measurements in original physical units (`measured_min`, `measured_max`, `measured_mean`, `measured_std`, `measured_median`, `measured_iqr`, `measured_amplitude_range_ptp`, `measured_max_constant_run_samples`, `measured_extreme_rail_fraction`, `measured_unique_value_ratio`, `measured_mode_value_fraction`, `measured_repeated_adjacent_fraction`).
  - Columns prefixed with `heuristic_*` (`heuristic_is_flatline`, `heuristic_is_near_flatline`, `heuristic_potential_clipping`, `heuristic_suspicious_repeated_values`, `heuristic_is_counter_like`) are configurable diagnostic indicators for human review. They do **not** constitute definitive proof that a trace is a physiological or hardware artifact.

---

## 3. Output Artifacts

Running the Phase 2 pipeline writes the following outputs to `--output-dir`:

1. `signal_statistics.csv`: Channel-by-channel table across all inspected recordings containing provenance metadata, EDF header ranges, measured statistics in original physical units, and heuristic quality flags.
2. `signal_quality_summary.json`: Structured JSON summary covering representative recording selection, duration variability by task, channel consistency, missing expected EEG channels, amplitude summaries, heuristic flag counts, and unreadable recording errors.
3. `signal_quality_report.md`: Formatted Markdown report with summary tables and diagnostic findings.
4. `figures/`:
   - `raw_waveform_<task>_<recording>.png`: Multi-channel stacked raw EEG waveforms against time with native units and flagged-channel highlighting.
   - `channel_detail_<task>_<recording>_<channel>.png`: Individual-channel raw waveform with mean reference line and measured metrics.
   - `amplitude_distribution_<task>_<recording>.png`: Per-channel mean-centered amplitude distribution (IQR and whiskers) in original physical units.
   - `task_comparison_Char_Digit_Image.png`: Side-by-side comparison of representative recordings across `Char`, `Digit`, and `Image`.
   - `dataset_channel_amplitude_distribution.png`: Aggregated per-channel standard deviation distribution across all inspected recordings.

---

## 4. Exact Execution Commands

### Local Execution (PowerShell / Bash)

Run both Phase 1 audit and Phase 2 signal-quality analysis in one command:

```powershell
python -m imagined_speech.cli `
  --mode both `
  --config configs/default_config.yaml `
  --data-root "./data/raw" `
  --output-dir "./outputs/phase2_signal_quality" `
  --recordings-per-task 3 `
  --seed 42
```

Run Phase 2 signal-quality analysis using an existing `manifest.csv` and inspect a specific file in addition to representative samples:

```powershell
python -m imagined_speech.cli `
  --mode signal-quality `
  --config configs/default_config.yaml `
  --data-root "./data/raw" `
  --manifest "./outputs/phase1_audit/manifest.csv" `
  --output-dir "./outputs/phase2_signal_quality" `
  --recordings-per-task 3 `
  --recording "Char/Sub01_A.edf"
```

### Google Colab Execution

Open `notebooks/02_signal_quality.ipynb` in Google Colab, or run the following commands in a Colab cell after mounting Google Drive at `/content/drive`:

```bash
!git clone https://github.com/ACT2039/imagined-speech-eeg.git /content/imagined-speech-eeg
%cd /content/imagined-speech-eeg
!python -m pip install -q -e ".[dev]"

# Run Phase 1 Audit + Phase 2 Signal Quality Analysis and save reports/figures to Google Drive
!python -m imagined_speech.cli \
  --mode both \
  --config configs/default_config.yaml \
  --data-root "/content/data/raw" \
  --output-dir "/content/drive/MyDrive/imagined_speech_project/outputs/phase2_signal_quality" \
  --recordings-per-task 3 \
  --seed 42
```
