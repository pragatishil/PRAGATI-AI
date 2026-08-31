# Reproducibility Plan

## Goal

The repository should allow an independent researcher to understand and, where the source datasets and computational resources permit, reproduce the principal experiments reported in the PRAGATI-AI manuscript.

## Required layers

### 1. Data
- Dataset acquisition instructions
- Dataset versions/access dates
- Class mappings
- Train/validation/test split manifests
- Sequence manifests
- Preprocessing metadata

### 2. Code
- Dataset construction
- Severity proxy construction
- Visual representation learning
- CORAL/ordinal modelling
- Severity modelling
- Pseudo-temporal sequence construction
- Sequence augmentation
- PrognosisNet training
- Baseline training
- Ablation training
- Calibration
- Robustness
- Evaluation
- Figure/table generation

### 3. Models
Retain the final checkpoints required to reproduce reported results. Intermediate checkpoints can be archived after the complete inventory is available.

### 4. Results
Preserve the CSV/NPY/PKL outputs that directly support manuscript tables, metrics, predictions, calibration, robustness and failure analysis.

### 5. Documentation
Every reported table and figure should eventually have a clear path from:
dataset → script → output → manuscript item.

## Double-anonymous review

The manuscript currently states that the repository containing preprocessed sequences and trained model weights is omitted during double-anonymized review and restored after peer review. Any public repository release should therefore be synchronized with the journal's review requirements.
