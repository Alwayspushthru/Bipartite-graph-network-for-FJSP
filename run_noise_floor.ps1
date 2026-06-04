# Training-time noise floor driver.
#
# Retrains the model K times changing ONLY --seed_train (after removing
# setup_seed(seed_test) from train.py, seed_train alone governs weight init +
# data sampling + PPO optimization), tags each saved checkpoint by seed, then
# evaluates all K greedily and reports per-group mean gap and run-to-run std.
#
# Run from the repo root:   .\run_noise_floor.ps1

$Seeds      = 300..309                       # 10 independent training runs
$DataSource = 'SD3'                          # training data source (default)
$NJ         = 10                             # default
$NM         = 5                              # default
$TestData   = @('SD1', 'SD2', 'BenchData')   # eval sets (each split into groups)
$Tag        = 'nf'                           # checkpoint filename prefix

# train.py model_name: SD3 -> "{NJ}x{NM}"; SD2 -> "{NJ}x{NM}+<suffix>".
$ModelName = "${NJ}x${NM}"
$Src       = "./trained_network/$ModelName.pth"

$models = @()
foreach ($s in $Seeds) {
    Write-Host "==== training seed_train=$s ====" -ForegroundColor Cyan

    # Record the checkpoint mtime before training so we can detect the case
    # where validation never improved and save_model() was never called.
    if (Test-Path $Src) { $before = (Get-Item $Src).LastWriteTimeUtc }
    else                { $before = [datetime]::MinValue }

    python train.py --seed_train $s --data_source $DataSource --n_j $NJ --n_m $NM
    if ($LASTEXITCODE -ne 0) { Write-Error "train failed @ seed $s"; break }

    if (-not (Test-Path $Src)) {
        Write-Error "expected model $Src not found after seed $s; check data_source/n_j/n_m"
        break
    }
    $after = (Get-Item $Src).LastWriteTimeUtc
    if ($after -le $before) {
        Write-Warning "model $Src not updated for seed $s (no validation improvement?); skipping"
        continue
    }

    $dst = "./trained_network/${Tag}_seed$s.pth"
    Copy-Item $Src $dst -Force
    $models += "${Tag}_seed$s"
    Write-Host "  saved -> $dst" -ForegroundColor Green
}

if ($models.Count -lt 2) {
    Write-Error "only $($models.Count) model(s) produced; need >= 2 for a noise floor"
    exit 1
}

Write-Host "==== evaluating $($models.Count) models (greedy) on $($TestData -join ', ') ====" -ForegroundColor Cyan
python noise_floor_eval.py --test_data $TestData --test_model $models
