# PRAGATI-AI v10 — Reproducibility Repository

**Manuscript:** *Weakly Supervised Pseudo-Temporal Forecasting of Crop Disease Progression from Static Plant Imagery*

This repository is an organized working repository built from the supplied `version-v10.ipynb`.

## Important status

This is the **complete repository scaffold / consolidation point** for v10. It intentionally preserves the supplied notebook logic rather than silently refactoring or changing the research implementation.

The repository is designed so that generated artifacts can be moved into a single location first. After consolidation, we can decide what should be:
- retained as public reproducibility material,
- archived,
- excluded because it is an intermediate/debug artifact,
- or linked to an external dataset/repository.

## Current organization

```text
PRAGATI-AI-v10-repository/
├── notebooks/
│   └── version-v10.ipynb
├── scripts/
│   ├── 00_prognosis_master_builder_v10.py
│   ├── 01_research_grade_v10.py
│   ├── 02_prognosisnet_v10.py
│   ├── 03_coffee_sequence_builder_v10.py
│   ├── 04_cross_domain_coffee_evaluation.py
│   ├── 05_baselines_ablations_v10.py
│   ├── 06_elsevier_figure_generation.py
│   └── 07_inventory_dataset_files.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── models/
│   └── checkpoints/
├── results/
│   ├── tables/
│   ├── metrics/
│   ├── predictions/
│   ├── calibration/
│   ├── robustness/
│   └── analysis/
├── figures/
│   ├── elsevier/
│   └── eda/
├── configs/
├── docs/
└── archive/
    └── notebook_cell_exports/
```

## Pipeline represented by v10

1. Build the prognosis master dataset and severity/quality reports.
2. Perform research-grade visual representation, morphology, CORAL, severity modelling and pseudo-temporal sequence construction.
3. Train/evaluate the prognosis network and uncertainty/calibration components.
4. Build the held-out coffee sequence dataset.
5. Perform cross-domain evaluation.
6. Run baseline comparisons and ablations.
7. Generate Elsevier figures, graphical abstract, highlights, captions and checklist.
8. Inventory the complete generated dataset/file collection.

## Source datasets

The original external datasets should **not be copied into this repository unless their licenses explicitly permit redistribution**.

The supplied manuscript identifies:
- PlantSeg
- Rice leaf disease dataset
- Field-acquired Plant Disease Dataset / Kaggle

The `data/` directory is therefore reserved for manifests, preprocessing outputs, split definitions and documentation unless redistribution rights are confirmed.

## Reproducibility materials

The final repository should eventually contain, where appropriate:
- source code,
- environment specification,
- configuration files,
- dataset acquisition instructions,
- preprocessing/split manifests,
- final model checkpoints,
- reported predictions,
- CSV files underlying manuscript tables,
- calibration/robustness outputs,
- figure-generation code and final figures,
- documentation describing how each manuscript table/figure is reproduced.

## v10 paths

The supplied notebook currently uses Kaggle-specific paths such as `/kaggle/input/...` and `/kaggle/working/...`.

**Do not refactor these paths yet.** First consolidate all generated artifacts. Then we can make a clean, platform-independent configuration layer without changing the scientific pipeline.

## Next consolidation step

Place the generated v10 files into this repository using the folders above. Then audit every file and classify it as:

**PUBLIC / ARCHIVE / REMOVE / EXTERNAL-LINK / REQUIRES-LICENSE-CHECK**

Only after that should we create the final GitHub/Zenodo release.

## License

No license has been selected yet. Choose one deliberately before public release.
