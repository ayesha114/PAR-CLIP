#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
printf '\n[1/4] Checking Ubuntu tools...\n'
if ! command -v curl >/dev/null 2>&1; then
  echo 'curl is missing. Install it with: sudo apt update && sudo apt install -y curl'
  exit 1
fi
if [ ! -x "$HOME/.local/bin/uv" ]; then
  echo 'Installing uv...'
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$HOME/.local/bin/uv"
printf '\n[2/4] Setting up Python 3.11 (isolated from system Python)...\n'
"$UV" python install 3.11
"$UV" venv --python 3.11 .venv
printf '\n[3/4] Installing project dependencies (may download several GB)...\n'
"$UV" pip install --python .venv/bin/python -r requirements.txt
printf '\n[4/4] Checking GPU and running offline smoke tests...\n'
.venv/bin/python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED')
PY
.venv/bin/python test_smoke.py
printf '\nSETUP COMPLETE. Edit config.json with your dataset paths, then run:\n'
printf '  .venv/bin/python run.py check\n'
printf '  .venv/bin/python run.py baseline\n'
printf '  .venv/bin/python run.py parclip\n'
