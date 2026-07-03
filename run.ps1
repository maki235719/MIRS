# Dev shortcuts for the fixed-point stress-evaluation script (Windows / no make needed).
# ASCII-only on purpose: Windows PowerShell 5.1 mis-parses UTF-8 (no BOM) multibyte comments.
# Usage:
#   .\run.ps1 run             # run the script
#   .\run.ps1 clean           # delete session outputs (log / graph / __pycache__)
#   .\run.ps1 clean-profiles  # delete learned baselines + person IDs (reset fixed-point data)
#   .\run.ps1 clean-all       # delete everything above
#   .\run.ps1 show-profiles   # pretty-print the saved profiles
param([string]$task = "run")

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Remove-Silent([string[]]$paths) {
    foreach ($p in $paths) {
        Remove-Item -Path $p -Recurse -Force -ErrorAction SilentlyContinue
    }
}

switch ($task) {
    "run" {
        & $python peroson_detect.py
    }
    "clean" {
        Remove-Silent @("stress_log.csv", "stress_graph.png", "__pycache__")
        Write-Host "Removed session outputs."
    }
    "clean-profiles" {
        Remove-Silent @("person_profiles.json", "session_history.jsonl")
        Write-Host "Removed fixed-point data (person_profiles.json / session_history.jsonl)."
    }
    "clean-all" {
        Remove-Silent @("stress_log.csv", "stress_graph.png", "__pycache__",
                        "person_profiles.json", "session_history.jsonl")
        Write-Host "Removed all outputs and fixed-point data."
    }
    "show-profiles" {
        if (Test-Path "person_profiles.json") {
            Get-Content -Path "person_profiles.json" -Raw -Encoding UTF8
        } else {
            Write-Host "person_profiles.json not found (run the script first)."
        }
    }
    default {
        Write-Host "unknown task: $task"
        Write-Host "targets: run | clean | clean-profiles | clean-all | show-profiles"
        exit 1
    }
}
