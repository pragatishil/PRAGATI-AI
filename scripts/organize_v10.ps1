$ErrorActionPreference = "Stop"

# ============================================================
# PRAGATI-AI v10 Repository Organizer
# Source      : C:\Users\hriti\Downloads\version v10
# Destination : E:\PRAGATI-AI
#
# IMPORTANT:
# - COPY only; source files are never deleted.
# - Does not execute Git commands.
# - Does not modify .git.
# - Creates an inventory of all copied files.
# ============================================================

$Source = "C:\Users\hriti\Downloads\version v10"
$Repo   = "E:\PRAGATI-AI"

if (!(Test-Path $Source)) {
    throw "Source directory not found: $Source"
}

if (!(Test-Path $Repo)) {
    throw "Repository directory not found: $Repo"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " PRAGATI-AI v10 Repository Organization" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Source      : $Source"
Write-Host "Repository  : $Repo"
Write-Host ""

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

function Copy-V10File {
    param(
        [string]$Name,
        [string]$Destination
    )

    $SourceFile = Join-Path $Source $Name
    $DestDir    = Join-Path $Repo $Destination
    $DestFile   = Join-Path $DestDir $Name

    if (!(Test-Path $SourceFile)) {
        Write-Warning "NOT FOUND: $Name"
        return
    }

    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

    if (Test-Path $DestFile) {
        Write-Host "EXISTS : $Destination\$Name" -ForegroundColor Yellow
    }
    else {
        Copy-Item -LiteralPath $SourceFile -Destination $DestFile
        Write-Host "COPIED : $Destination\$Name" -ForegroundColor Green
    }
}

# ------------------------------------------------------------
# Create repository directories
# ------------------------------------------------------------

$Directories = @(
    "data\processed\prognosis",
    "data\processed\coffee",
    "data\processed\splits",
    "data\manifests",
    "data\external",

    "models\checkpoints\final\prognosis",
    "models\checkpoints\final\core",
    "models\checkpoints\ablations",
    "models\checkpoints\baselines",
    "models\checkpoints\segmentation",
    "models\checkpoints\archive",

    "results\tables",
    "results\metrics",
    "results\predictions",
    "results\calibration",
    "results\calibration\coral",
    "results\robustness",
    "results\analysis",
    "results\representations",

    "figures\manuscript",
    "figures\eda",

    "archive\v10_original"
)

foreach ($Dir in $Directories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Repo $Dir) | Out-Null
}

# ------------------------------------------------------------
# 1. Preserve a complete original v10 copy
# ------------------------------------------------------------

Write-Host ""
Write-Host "Creating complete v10 archive..." -ForegroundColor Cyan

$OriginalArchive = Join-Path $Repo "archive\v10_original"

Get-ChildItem -LiteralPath $Source -Recurse -File | ForEach-Object {

    $Relative = $_.FullName.Substring($Source.Length).TrimStart("\")
    $Target   = Join-Path $OriginalArchive $Relative
    $TargetDir = Split-Path $Target -Parent

    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

    if (!(Test-Path $Target)) {
        Copy-Item -LiteralPath $_.FullName -Destination $Target
    }
}

Write-Host "Complete original archive created." -ForegroundColor Green

# ------------------------------------------------------------
# 2. Manuscript / Elsevier figures
# ------------------------------------------------------------

$Elsevier = Join-Path $Source "elsevier_figures"

if (Test-Path $Elsevier) {

    Write-Host ""
    Write-Host "Copying manuscript figures..." -ForegroundColor Cyan

    Get-ChildItem $Elsevier -File | ForEach-Object {

        Copy-V10File `
            -Name $_.Name `
            -Destination "figures\manuscript"
    }
}

# ------------------------------------------------------------
# 3. EDA figures
# ------------------------------------------------------------

$EDA = Join-Path $Source "part2_eda_v10"

if (Test-Path $EDA) {

    Write-Host ""
    Write-Host "Copying EDA figures..." -ForegroundColor Cyan

    Get-ChildItem $EDA -File | ForEach-Object {

        Copy-Item `
            -LiteralPath $_.FullName `
            -Destination (Join-Path $Repo "figures\eda")
    }

    $GradCAM = Join-Path $EDA "gradcam"

    if (Test-Path $GradCAM) {

        New-Item `
            -ItemType Directory `
            -Force `
            -Path (Join-Path $Repo "figures\eda\gradcam") |
            Out-Null

        Get-ChildItem $GradCAM -File | ForEach-Object {

            Copy-Item `
                -LiteralPath $_.FullName `
                -Destination (Join-Path $Repo "figures\eda\gradcam")
        }
    }
}

# ------------------------------------------------------------
# 4. Dataset / processed data
# ------------------------------------------------------------

$PrognosisFiles = @(
    "prognosis_master.csv",
    "prognosis_classes_v10.pkl"
)

$CoffeeFiles = @(
    "coffee_v10.csv",
    "coffee_sequences_v10.csv"
)

$SplitFiles = @(
    "train_v10.csv",
    "val_v10.csv",
    "test_v10.csv",
    "train_sequences_v10.csv",
    "val_sequences_v10.csv",
    "test_sequences_v10.csv",
    "cls_eval_splits_v10.csv"
)

Write-Host ""
Write-Host "Copying processed data..." -ForegroundColor Cyan

foreach ($f in $PrognosisFiles) {
    Copy-V10File $f "data\processed\prognosis"
}

foreach ($f in $CoffeeFiles) {
    Copy-V10File $f "data\processed\coffee"
}

foreach ($f in $SplitFiles) {
    Copy-V10File $f "data\processed\splits"
}

# ------------------------------------------------------------
# 5. Result tables
# ------------------------------------------------------------

$TableFiles = @(
    "table1_baseline_comparison.csv",
    "table2_ablation_study.csv",
    "table3_efficiency_metrics.csv",
    "table4_cross_domain.csv",
    "table5_severity_wise.csv"
)

Write-Host ""
Write-Host "Copying result tables..." -ForegroundColor Cyan

foreach ($f in $TableFiles) {
    Copy-V10File $f "results\tables"
}

# ------------------------------------------------------------
# 6. Metrics
# ------------------------------------------------------------

$MetricFiles = @(
    "classification_metrics_v10.csv",
    "prognosis_metrics_v10.csv",
    "robustness_metrics_v10.csv",
    "severity_statistics.csv",
    "test_metrics_segnet_v8.csv"
)

foreach ($f in $MetricFiles) {
    Copy-V10File $f "results\metrics"
}

# ------------------------------------------------------------
# 7. Predictions and classification reports
# ------------------------------------------------------------

$PredictionFiles = @(
    "predictions_test_v10.csv",
    "predictions_val_v10.csv",
    "detailed_predictions_test.csv",
    "classification_report_v10_test.csv",
    "classification_report_v10_test_seed_123.csv",
    "classification_report_v10_test_seed_42.csv",
    "classification_report_v10_test_seed_999.csv",
    "classification_report_v10_val.csv"
)

foreach ($f in $PredictionFiles) {
    Copy-V10File $f "results\predictions"
}

# ------------------------------------------------------------
# 8. Analysis
# ------------------------------------------------------------

$AnalysisFiles = @(
    "baselines_ablations_results.csv",
    "class_distribution.csv",
    "eligible_classes.csv",
    "failure_analysis_report.csv",
    "prognosis_train_log_v10.csv",
    "training_log_segnet_v8.csv",
    "sequence_augmentation_stats_v10.csv"
)

foreach ($f in $AnalysisFiles) {
    Copy-V10File $f "results\analysis"
}

# ------------------------------------------------------------
# 9. Calibration
# ------------------------------------------------------------

$CalibrationFiles = @(
    "calibration_coverage.csv",
    "calibration_reliability_curve.png",
    "conformal_quantiles_v10.pkl",
    "curve_residuals_v10.pkl"
)

foreach ($f in $CalibrationFiles) {
    Copy-V10File $f "results\calibration"
}

# CORAL probability / severity arrays

$CoralFiles = @(
    "coral_prob_cf_v10.npy",
    "coral_prob_te_v10.npy",
    "coral_prob_tr_v10.npy",
    "coral_prob_va_v10.npy",
    "coral_sev_cf_v10.npy",
    "coral_sev_te_v10.npy",
    "coral_sev_tr_v10.npy",
    "coral_sev_va_v10.npy"
)

foreach ($f in $CoralFiles) {
    Copy-V10File $f "results\calibration\coral"
}

# ------------------------------------------------------------
# 10. Representation artifacts
# ------------------------------------------------------------

$RepresentationFiles = @(
    "vis_cf_v10.npy",
    "vis_te_v10.npy",
    "vis_tr_v10.npy",
    "vis_va_v10.npy",
    "attention_sample_trajectories.png"
)

foreach ($f in $RepresentationFiles) {
    Copy-V10File $f "results\representations"
}

# ------------------------------------------------------------
# 11. Final / core model checkpoints
# ------------------------------------------------------------

$CoreCheckpoints = @(
    "best_coral_v10.pth",
    "best_segnet_v8.pth",
    "best_severity_net_v10.pth",
    "best_visual_model_v10.pth",
    "checkpoint_prognosis_v10.pth",
    "checkpoint_segnet_v8.pth",
    "coral_ckpt_v10.pth",
    "sev_ckpt_v10.pth",
    "vis_ckpt_v10.pth"
)

foreach ($f in $CoreCheckpoints) {
    Copy-V10File $f "models\checkpoints\final\core"
}

# ------------------------------------------------------------
# 12. Final prognosis five-seed checkpoints
# ------------------------------------------------------------

$Seeds = @("123","2025","42","777","999")

foreach ($Seed in $Seeds) {

    $File = "best_prognosis_net_v10_seed$Seed.pth"

    Copy-V10File `
        $File `
        "models\checkpoints\final\prognosis"
}

# Non-seed best prognosis checkpoint

Copy-V10File `
    "best_prognosis_net_v10.pth" `
    "models\checkpoints\final\prognosis"

# ------------------------------------------------------------
# 13. Ablation checkpoints
# ------------------------------------------------------------

$Ablations = @{
    "NoCBAM"          = "no_cbam"
    "NoCORAL"         = "no_coral"
    "NoCrossAttn"     = "no_cross_attention"
    "NoMonotonicity"  = "no_monotonicity"
    "NoSeqAug"        = "no_sequence_augmentation"
    "NoVisDelta"      = "no_visual_delta"
}

foreach ($Ablation in $Ablations.Keys) {

    foreach ($Seed in $Seeds) {

        $File = "ckpt_Abl_${Ablation}_seed${Seed}.pth"

        $Destination = "models\checkpoints\ablations\$($Ablations[$Ablation])"

        Copy-V10File `
            $File `
            $Destination
    }
}

# ------------------------------------------------------------
# 14. Baseline checkpoints
# ------------------------------------------------------------

$Baselines = @{
    "B1_MLP"        = "mlp"
    "B2_EmbLSTM"    = "embedding_lstm"
    "B2b_GRU"       = "gru"
    "B3_TCN"        = "tcn"
    "B4_Transformer"= "transformer"
    "B5_NoTemporal" = "no_temporal"
}

foreach ($Baseline in $Baselines.Keys) {

    foreach ($Seed in $Seeds) {

        $File = "ckpt_${Baseline}_seed${Seed}.pth"

        $Destination = "models\checkpoints\baselines\$($Baselines[$Baseline])"

        Copy-V10File `
            $File `
            $Destination
    }
}

# ------------------------------------------------------------
# 15. Proposed model without coffee
# ------------------------------------------------------------

foreach ($Seed in $Seeds) {

    $File = "ckpt_Proposed_NoCoffee_seed${Seed}.pth"

    Copy-V10File `
        $File `
        "models\checkpoints\ablations\proposed_no_coffee"
}

# ------------------------------------------------------------
# 16. Temporary visual fold checkpoints
# ------------------------------------------------------------

for ($i = 0; $i -le 4; $i++) {

    $File = "vis_fold${i}_tmp_v10.pth"

    Copy-V10File `
        $File `
        "models\checkpoints\archive\visual_fold_checkpoints"
}

# ------------------------------------------------------------
# 17. Additional standalone artifacts
# ------------------------------------------------------------

Copy-V10File `
    "__huggingface_repos__.json" `
    "data\manifests"

# ------------------------------------------------------------
# 18. Inventory
# ------------------------------------------------------------

Write-Host ""
Write-Host "Creating repository inventory..." -ForegroundColor Cyan

$InventoryPath = Join-Path $Repo "docs\v10_repository_inventory.csv"

Get-ChildItem -LiteralPath $Repo -Recurse -File |
    Where-Object {
        $_.FullName -notmatch "\\\.git\\"
    } |
    Select-Object `
        @{Name="RelativePath";Expression={
            $_.FullName.Substring($Repo.Length).TrimStart("\")
        }},
        Name,
        Extension,
        Length,
        @{Name="SizeMB";Expression={
            [math]::Round($_.Length / 1MB, 2)
        }} |
    Export-Csv `
        -Path $InventoryPath `
        -NoTypeInformation `
        -Encoding UTF8

# ------------------------------------------------------------
# 19. Source inventory
# ------------------------------------------------------------

$SourceInventoryPath = Join-Path $Repo "docs\v10_source_inventory.csv"

Get-ChildItem -LiteralPath $Source -Recurse -File |
    Select-Object `
        @{Name="RelativePath";Expression={
            $_.FullName.Substring($Source.Length).TrimStart("\")
        }},
        Name,
        Extension,
        Length,
        @{Name="SizeMB";Expression={
            [math]::Round($_.Length / 1MB, 2)
        }} |
    Export-Csv `
        -Path $SourceInventoryPath `
        -NoTypeInformation `
        -Encoding UTF8

# ------------------------------------------------------------
# 20. Summary
# ------------------------------------------------------------

$SourceFiles = @(Get-ChildItem $Source -Recurse -File)
$RepoFiles   = @(Get-ChildItem $Repo -Recurse -File)

$SourceSize = ($SourceFiles | Measure-Object Length -Sum).Sum
$RepoSize   = ($RepoFiles | Measure-Object Length -Sum).Sum

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " ORGANIZATION COMPLETE" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host ("Source files : {0}" -f $SourceFiles.Count)
Write-Host ("Source size  : {0:N2} GB" -f ($SourceSize / 1GB))
Write-Host ("Repo files   : {0}" -f $RepoFiles.Count)
Write-Host ("Repo size    : {0:N2} GB" -f ($RepoSize / 1GB))
Write-Host ""
Write-Host "Inventory:"
Write-Host "  docs\v10_source_inventory.csv"
Write-Host "  docs\v10_repository_inventory.csv"
Write-Host ""
Write-Host "IMPORTANT: No Git commands were executed."
Write-Host "IMPORTANT: The original v10 directory was not modified."
Write-Host ""