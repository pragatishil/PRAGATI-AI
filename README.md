# PRAGATI-AI

## Pseudo-Temporal Agricultural Disease Prognosis from Static Imagery

PRAGATI-AI is a research framework for agricultural disease analysis and prognosis using computer vision, ordinal severity modelling, pseudo-temporal sequence construction, and uncertainty-aware forecasting.

The repository contains the code, processed research artifacts, trained models, evaluation results, figures, and reproducibility metadata associated with the PRAGATI-AI research workflow.

> **Research status:** This repository contains research-stage implementations and experimental results. Reported results should be interpreted in the context of the accompanying datasets, experimental configuration, and reproducibility documentation.

---

## Overview

PRAGATI-AI investigates whether disease progression information can be approximated from collections of static agricultural images by constructing pseudo-temporal severity sequences.

The V10 workflow combines:

* visual representation learning;
* ordinal disease-severity modelling;
* composite severity estimation;
* pseudo-temporal sequence construction;
* temporal prognosis modelling;
* attention-based forecasting;
* monotonicity constraints;
* uncertainty estimation;
* conformal prediction;
* baseline comparison;
* ablation analysis;
* robustness evaluation.

The primary prognosis experiments forecast disease severity over multiple future pseudo-temporal steps.

---

## Repository Structure

```text
PRAGATI-AI/
│
├── archive/
│   └── notebook_cell_exports/
│
├── configs/
│
├── data/
│   ├── calibration/
│   ├── processed/
│   │   ├── coffee/
│   │   └── prognosis/
│   └── representations/
│
├── docs/
│
├── figures/
│   ├── eda/
│   └── manuscript/
│
├── metadata/
│
├── models/
│   ├── ablations/
│   ├── baselines/
│   ├── core/
│   └── proposed/
│
├── notebooks/
│
├── results/
│   ├── analysis/
│   ├── calibration/
│   ├── metrics/
│   ├── predictions/
│   ├── representations/
│   └── tables/
│
├── scripts/
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## V10 Experimental Pipeline

The V10 prognosis workflow can be summarized as:

```text
Agricultural Images
        │
        ▼
Visual Representation Learning
        │
        ▼
Composite Disease Severity
        │
        ▼
Pseudo-Temporal Sequence Construction
        │
        ▼
Prognosis Network
        │
        ├── Temporal modelling
        ├── Attention
        ├── Visual-delta information
        └── Monotonicity constraints
        │
        ▼
Multi-step Severity Forecast
        │
        ├── t+1
        ├── t+2
        └── t+3
        │
        ▼
Uncertainty Quantification
        │
        ▼
Conformal Prediction
```

---

## Main V10 Components

### Prognosis

The V10 prognosis implementation is primarily contained in:

```text
scripts/00_prognosis_master_builder_v10.py
scripts/01_research_grade_v10.py
scripts/02_prognosisnet_v10.py
```

### Cross-domain evaluation

```text
scripts/03_coffee_sequence_builder_v10.py
scripts/04_cross_domain_coffee_evaluation.py
```

### Baselines and ablations

```text
scripts/05_baselines_ablations_v10.py
```

### Figure generation

```text
scripts/06_elsevier_figure_generation.py
```

### Dataset inventory

```text
scripts/07_inventory_dataset_files.py
```

---

## Models

The repository contains V10 checkpoints for:

* the visual model;
* severity modelling;
* CORAL-based ordinal modelling;
* prognosis modelling;
* baseline architectures;
* ablation configurations.

Model checkpoints are organized under:

```text
models/core/
models/proposed/
models/baselines/
models/ablations/
```

---

## Results

V10 evaluation outputs are organized under:

```text
results/metrics/
results/predictions/
results/analysis/
results/calibration/
results/tables/
```

Important prognosis outputs include:

```text
results/metrics/prognosis_metrics_v10.csv
results/metrics/robustness_metrics_v10.csv
results/analysis/prognosis_train_log_v10.csv
results/predictions/predictions_test_v10.csv
results/predictions/predictions_val_v10.csv
```

The repository also contains calibration artifacts, representations, analysis files, and manuscript tables.

---

## Reproducibility

Reproducibility information is maintained in:

```text
docs/reproducibility.md
metadata/CHECKPOINT_MANIFEST.csv
metadata/FILE_HASHES_SHA256.csv
requirements.txt
```

The V10 repository also contains the original research notebook:

```text
notebooks/version-v10.ipynb
```

and archived notebook-cell exports under:

```text
archive/notebook_cell_exports/
```

Where applicable, SHA-256 hashes are provided for important research artifacts.

---

## Important Note on Data and Models

The repository contains research datasets, derived data, trained model checkpoints, and other experimental artifacts.

The MIT License applies to the original code and other material for which the repository authors hold the necessary rights. Third-party datasets, pretrained models, imagery, or other externally sourced material may be subject to their own licenses and terms.

Users are responsible for checking the applicable terms before redistributing or using third-party material.

---

## Installation

Create a Python environment and install the required dependencies:

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

---

## Research Reproduction

The V10 workflow is organized into numbered scripts so that the major stages can be identified from the repository:

```text
00  → master/processed data construction
01  → research-grade processing/training workflow
02  → prognosis network
03  → cross-domain sequence construction
04  → cross-domain evaluation
05  → baselines and ablations
06  → manuscript figure generation
07  → dataset inventory
```

For detailed experimental provenance and artifact relationships, see:

```text
docs/reproducibility.md
```

---

## Figures

Manuscript and exploratory figures are stored under:

```text
figures/eda/
figures/manuscript/
```

Generated analysis figures are also available under:

```text
results/analysis/
```

---

## Citation

If this repository is used in academic work, please cite the associated publication once the manuscript is publicly available.

A formal citation will be added here when the publication record is established.

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for the full license text.

---

## Status

**Current research checkpoint:** V10

The V10 checkpoint is preserved in Git history and is intended to provide a reproducible reference state for subsequent research development.
