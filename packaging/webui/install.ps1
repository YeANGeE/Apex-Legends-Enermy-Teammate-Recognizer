$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}
Invoke-Checked 'py' @('-3.13', '-m', 'venv', '.venv-yolo')
$pythonPath = Join-Path $PSScriptRoot '.venv-yolo\Scripts\python.exe'
Invoke-Checked $pythonPath @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked $pythonPath @('-m', 'pip', 'install', 'torch==2.14.1+cu132', 'torchvision==0.29.1+cu132', '--index-url', 'https://download.pytorch.org/whl/cu132')
Invoke-Checked $pythonPath @('-m', 'pip', 'install', '-r', 'requirements.txt')
Invoke-Checked $pythonPath @('-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA GPU and driver"; print(torch.cuda.get_device_name(0))')
Write-Host 'Dependencies installed. FFmpeg on PATH is required for video export.'
Write-Host 'Run Start-WebUI.cmd to start the local application.'
