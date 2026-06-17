# run_ablations.ps1
# Run 4 ablation experiments serially: train -> test.
# IMPORTANT: test reuses the SAME --ablation as training, otherwise the eval
# runs the wrong forward path and the numbers are meaningless.
# Usage: .\run_ablations.ps1
#        .\run_ablations.ps1 -Only 10x5_nogru -SkipTrain

param(
    [string]$Only = "",                       # run only this model_name; empty = all
    [string[]]$TestData = @("BenchData"),     # test folders under ./data (BenchData has baseline -> gap%)
    [int]$BeamWidth = 10,                      # beam width; >1 enables beam, 1 = Greedy
    [bool]$BeamStochastic = $true,            # true = SBeam (stochastic Gumbel-top-k beam)
    [switch]$SkipTrain,                        # test only, reuse existing .pth
    [switch]$SkipTest                          # train only
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# (ablation switch, model_name) -- used by BOTH train and test
$experiments = @(
    @{ ablation = "none";         model = "10x5_full"    },
    @{ ablation = "mean_agg";     model = "10x5_meanagg" },
    @{ ablation = "no_pair_bias"; model = "10x5_nobias"  },
    @{ ablation = "no_gru";       model = "10x5_nogru"   }
)

if ($Only) { $experiments = $experiments | Where-Object { $_.model -eq $Only } }
if (-not $experiments) { Write-Error "No experiment matches: $Only"; exit 1 }

# torch lives in the conda env reFJSP_env, NOT in ./.venv. Prefer that env,
# fall back to whatever `python` is on PATH.
$python = "D:\Anaconda3\envs\reFJSP_env\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

foreach ($exp in $experiments) {
    $abl   = $exp.ablation
    $model = $exp.model
    Write-Host "`n==================== $model  (ablation=$abl) ====================" -ForegroundColor Cyan

    if (-not $SkipTrain) {
        Write-Host "[train] $model" -ForegroundColor Yellow
        & $python train.py --ablation $abl --model_name $model
        if ($LASTEXITCODE -ne 0) { Write-Error "TRAIN FAILED: $model (exit $LASTEXITCODE), skipping its test"; continue }
    }

    if (-not $SkipTest) {
        $mode = if ($BeamWidth -gt 1) { "$(if ($BeamStochastic){'SBeam'}else{'Beam'})x$BeamWidth" } else { "Greedy" }
        Write-Host "[test]  $model  on  $($TestData -join ', ')  ($mode)" -ForegroundColor Yellow
        & $python test.py --ablation $abl --test_model $model --test_data $TestData `
            --beam_width $BeamWidth --beam_stochastic $BeamStochastic
        if ($LASTEXITCODE -ne 0) { Write-Error "TEST FAILED: $model (exit $LASTEXITCODE)" }
    }
}

Write-Host "`nAll done. Summary: test_results\test_log.txt" -ForegroundColor Green
Write-Host "Per-instance: test_results\<dataset>\<model>_Bgnn-G_<timestamp>.xlsx" -ForegroundColor Green
