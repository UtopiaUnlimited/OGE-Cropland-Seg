param([int]$Port = 6006)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
python -m tensorboard.main --logdir (Join-Path $projectRoot 'runs') --host 127.0.0.1 --port $Port
