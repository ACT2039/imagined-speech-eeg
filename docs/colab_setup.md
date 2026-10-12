# Google Colab & Google Drive Setup Guide

This guide explains how to connect the `imagined-speech-recognition` repository to the raw EEG dataset stored in Google Drive and run Phase 1 dataset discovery and EDF metadata auditing in Google Colab.

---

## 1. Google Drive Storage Layout

Store the original raw dataset archive (or extracted directory) in your Google Drive. Do **not** place raw `.edf` files or `.zip` archives inside the Git repository.

Recommended Google Drive structure:

```text
MyDrive/
└── imagined_speech_project/
    ├── raw/
    │   └── imagined_speech_edf.zip      # Original read-only dataset archive
    └── outputs/
        └── phase1_audit/                # Persistent audit outputs (manifest, channel report, summary)
```

Once extracted, the dataset root directory should contain the three task directories:
- `Char/`
- `Digit/`
- `Image/`

---

## 2. Mounting Google Drive and Installing the Package in Colab

Run the following cells in a Google Colab notebook:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the GitHub repository into the Colab runtime and install the package in editable mode:

```bash
!git clone <YOUR_GITHUB_REPO_URL> /content/imagined-speech-recognition
%cd /content/imagined-speech-recognition
!python -m pip install -q -e ".[dev]"
```

Verify the installation by running the unit test suite:

```bash
!python -m pytest
```

---

## 3. Extracting the Dataset Archive Safely (Read-Only Source)

For fast disk I/O and to ensure the original archive on Google Drive is never modified, extract the ZIP archive into the Colab local filesystem (`/content/data/raw`) rather than extracting directly inside Google Drive:

```python
import zipfile
from pathlib import Path

# Set via environment variables or edit to match your Google Drive path
DRIVE_ZIP_PATH = Path("/content/drive/MyDrive/imagined_speech_project/raw/imagined_speech_edf.zip")
LOCAL_EXTRACT_DIR = Path("/content/data/raw")

LOCAL_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(DRIVE_ZIP_PATH, "r") as zf:
    zf.extractall(LOCAL_EXTRACT_DIR)

print("Extracted top-level entries:", [p.name for p in sorted(LOCAL_EXTRACT_DIR.iterdir())])
```

> **Note:** If the ZIP archive wraps `Char/`, `Digit/`, and `Image/` inside an outer parent folder (e.g., `/content/data/raw/Dataset/Char`), pass that inner folder as `--data-root`.

---

## 4. Running the Phase 1 Dataset Discovery & EDF Audit

Run the CLI using `--data-root` (pointing to the folder containing `Char`, `Digit`, and `Image`) and `--output-dir` (pointing to a persistent folder on Google Drive):

```bash
!python -m imagined_speech.cli \
  --config configs/default_config.yaml \
  --data-root "/content/data/raw" \
  --output-dir "/content/drive/MyDrive/imagined_speech_project/outputs/phase1_audit"
```

Alternatively, configure paths via environment variables:

```python
import os

os.environ["ISR_DATA_ROOT"] = "/content/data/raw"
os.environ["ISR_OUTPUT_DIR"] = "/content/drive/MyDrive/imagined_speech_project/outputs/phase1_audit"
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
  --data-root "/content/data/raw" \
  --manifest "/content/drive/MyDrive/imagined_speech_project/outputs/phase1_audit/manifest.csv" \
  --output-dir "/content/drive/MyDrive/imagined_speech_project/outputs/phase2_signal_quality" \
  --recordings-per-task 3 \
  --seed 42
```

See `docs/signal_quality.md` for details on the generated `signal_statistics.csv`, `signal_quality_summary.json`, `signal_quality_report.md`, and `figures/` visualizations.

