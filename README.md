# Imagined Speech EEG

A Python research project exploring imagined-speech recognition from EEG recordings, with a focus on reliable data validation, signal analysis, and reproducible machine-learning experiments.

## Project Overview

Imagined speech refers to internally rehearsing speech without producing audible words. This project investigates how EEG recordings can be analyzed to distinguish the intended labels in an imagined-speech dataset.

The raw dataset contains EDF recordings organized into three tasks:

- **Character recognition:** letter labels
- **Digit recognition:** numerical labels
- **Object-label recognition:** object-related labels

The experimental protocol and recording metadata must be validated before making claims about participant-independent recognition or the exact timing of imagined speech.

## Current Development Status

The project is being developed incrementally.

- [x] Initial Python project structure
- [x] EDF discovery and metadata-audit implementation
- [x] Unit tests for configuration, filename parsing, discovery, and EDF auditing
- [ ] Verification against the complete raw dataset in Google Colab
- [x] EEG signal-quality analysis
- [ ] Validated preprocessing pipeline
- [ ] Classical machine-learning baselines
- [ ] Deep-learning experiments
- [ ] Final model comparison and reproducibility report

A completed implementation step does not imply that its results have already been verified on the complete real dataset.

## Technology Stack

- Python
- MNE-Python
- NumPy and SciPy
- pandas
- Matplotlib
- scikit-learn
- Google Colab and Google Drive
- PyTorch for planned deep-learning experiments

## Project Design

The project uses a modular Python package for data discovery, signal processing, feature extraction, training, and evaluation. Notebooks orchestrate experiments and visualize results, while reusable logic lives in the `src/` package.

## Dataset

The raw EEG dataset is maintained separately from this repository. Place the original archive in Google Drive and follow the instructions in `docs/colab_setup.md`.

Raw EDF files, dataset archives, processed arrays, model checkpoints, and other generated outputs must not be committed to Git.

## Getting Started

Install the project and its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

Inspect the available CLI commands:

```bash
python -m imagined_speech.cli --help
```

For dataset access and execution in Google Colab, consult `docs/colab_setup.md` and `docs/signal_quality.md`.

## Research and Evaluation Principles

- Preserve original recordings and track data provenance.
- Validate labels, sampling frequencies, signal units, and channel metadata.
- Prevent recording-level and participant-level data leakage.
- Fit learned preprocessing only on training data.
- Report actual experimental results and disclose dataset limitations.

## License

The license applies to the original project code only. The external dataset remains subject to its own terms and conditions.
