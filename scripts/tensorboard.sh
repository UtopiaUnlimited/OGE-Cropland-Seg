#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m tensorboard.main --logdir "$project_root/runs" --host 127.0.0.1 --port "${1:-6006}"
