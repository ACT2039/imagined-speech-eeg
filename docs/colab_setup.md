# Google Colab & Google Drive Setup Guide

This guide explains how to connect the `imagined-speech-recognition` repository to the raw EEG dataset stored in Google Drive and run Phase 1 dataset discovery and EDF metadata auditing in Google Colab.

---

## 1. Google Drive Storage Layout

Store the original raw dataset archive (or extracted directory) in your Google Drive. Do **not** place raw `.edf` files or `.zip` archives inside the Git repository.

Expected Google Drive structure for Phase 1:

```text
MyDrive/
├── datasets/
│   └── Imagined_speech_EEG_edf.zip      # Original read-only dataset archive
└── imagined_speech_outputs/
    └── phase1_audit/                    # Persistent Phase 1 audit outputs (manifest, channel report, summary)
```

Once extracted, the dataset root directory contains the three task directories:
- `Char/`
- `Digit/`
- `Image/`

---

## 2. Mounting Google Drive and Installing the Package in Colab

Open `notebooks/01_phase1_dataset_audit.ipynb` in Google Colab, or run the following steps in a fresh Colab runtime:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone (or safely update) the GitHub repository into the Colab runtime and install the package in editable mode:

```bash
!git clone https://github.com/ACT2039/imagined-speech-eeg.git /content/imagined-speech-eeg
%cd /content/imagined-speech-eeg
!python -m pip install -e ".[dev]"
```

Verify the installation by running the unit test suite:

```bash
!python -m pytest
```

---

## 3. Extracting the Dataset Archive Safely and Locating `DATA_ROOT`

For fast disk I/O and to ensure the original archive on Google Drive is never modified, verify and extract `/content/drive/MyDrive/datasets/Imagined_speech_EEG_edf.zip` into the Colab local filesystem (`/content/data/raw`) without redundant re-extraction, then automatically locate the directory containing `Char/`, `Digit/`, and `Image/`:

```python
import zipfile
from pathlib import Path

DRIVE_ZIP_PATH = Path("/content/drive/MyDrive/datasets/Imagined_speech_EEG_edf.zip")
EXTRACT_ROOT = Path("/content/data/raw")

existing_edfs = (
    [p for p in EXTRACT_ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".edf"]
    if EXTRACT_ROOT.exists()
    else []
)
if not existing_edfs:
    if not DRIVE_ZIP_PATH.is_file():
        raise FileNotFoundError(f"Dataset ZIP archive not found at {DRIVE_ZIP_PATH}")
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DRIVE_ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_ROOT)

# Automatically locate the folder containing Char/, Digit/, and Image/
DATA_ROOT = next(
    cand.resolve()
    for cand in [EXTRACT_ROOT, *sorted(d for d in EXTRACT_ROOT.rglob("*") if d.is_dir())]
    if {"char", "digit", "image"}.issubset({c.name.lower() for c in cand.iterdir() if c.is_dir()})
)
print("Resolved DATA_ROOT:", DATA_ROOT)
```

---

## 4. Running the Phase 1 Dataset Discovery & EDF Audit

In `notebooks/01_phase1_dataset_audit.ipynb`, the notebook calls `run_dataset_audit` directly and saves outputs to `/content/drive/MyDrive/imagined_speech_outputs/phase1_audit/`.

You can also run the equivalent CLI command using `--data-root` (pointing to the resolved `DATA_ROOT` containing `Char`, `Digit`, and `Image`) and `--output-dir`:

```bash
!python -m imagined_speech.cli \
  --config configs/default_config.yaml \
  --data-root "/content/data/raw" \
  --output-dir "/content/drive/MyDrive/imagined_speech_outputs/phase1_audit"
```

Alternatively, configure paths via environment variables:

```python
import os

os.environ["ISR_DATA_ROOT"] = str(DATA_ROOT)
os.environ["ISR_OUTPUT_DIR"] = "/content/drive/MyDrive/imagined_speech_outputs/phase1_audit"
os.environ["ISR_SEED"] = "42"
```

```bash
!python -m imagined_speech.cli --config configs/default_config.yaml
```

---

## 5. Generated Audit Outputs

The Phase 1 audit writes three files to `--output-dir`:
1. `manifest.csv`: Recording-level inventory preserving `original_filename`, `relative_source_path`, `task_category`, `raw_label`, `normalized_label`, `unverified_candidate_prefix`, EDF header metadata (`edf_patient_id_raw`, `sfreq_hz`, `duration_sec`), and channel counts.
2. `channel_audit.csv`: Channel-level verification report comparing MNE's default type assignment against audited channel roles (`candidate_eeg`, `non_eeg_metadata_or_aux`, `non_eeg_counter_signal`, `unknown_needs_review`) and in-memory signal diagnostics (`is_flat`, `is_counter_like`, `has_nan_or_inf`).
3. `audit_summary.json`: Dataset-level summary statistics, class distributions per task, sampling rate consistency checks, and validation warnings.

---

## 6. Running Phase 2 EEG Signal Quality Analysis & Visualization

To run the Phase 2 signal-quality pipeline in Google Colab (or open `notebooks/02_signal_quality.ipynb`):

```bash
!python -m imagined_speech.cli \
  --mode signal-quality \
  --config configs/default_config.yaml \
  --data-root "/content/data/raw/Imagined_speech_EEG_edf" \
  --manifest "/content/drive/MyDrive/imagined_speech_outputs/phase1_audit/manifest.csv" \
  --output-dir "/content/drive/MyDrive/imagined_speech_outputs/phase2_signal_quality" \
  --recordings-per-task 3 \
  --seed 42
```

See `docs/signal_quality.md` for details on the generated `signal_statistics.csv`, `signal_quality_summary.json`, `signal_quality_report.md`, and `figures/` visualizations.

