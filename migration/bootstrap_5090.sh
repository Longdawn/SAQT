#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${CDTR_ENV_PREFIX:-/root/autodl-tmp/envs/cdtr5090-cu128}"
cd "$PROJECT_ROOT"
for tool in nvidia-smi nvcc g++; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; choose a CUDA 12.8 devel image." >&2; exit 1; }
done
nvcc --version | grep -q 'release 12.8' || { echo 'This bootstrap requires the CUDA 12.8 compiler.' >&2; exit 1; }
CONDA_BIN="$(command -v conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  for candidate in /root/miniconda3/bin/conda /root/miniconda/bin/conda /opt/conda/bin/conda; do
    if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
  done
fi
[[ -n "$CONDA_BIN" ]] || { echo 'Conda not found; install/use a Miniconda-based AutoDL image.' >&2; exit 1; }
mkdir -p migration_reports
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" python=3.11 pip
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$PY" -m pip install 'setuptools>=75,<81' wheel ninja
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r migration/requirements-cu128.txt
"$PY" -m pip check
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST='12.0'
export MAX_JOBS="${MAX_JOBS:-4}"
unset DTLR_SKIP_CUDA_VERSION_CHECK DTLR_MSDA_FORCE_PYTORCH DTLR_MSDA_FALLBACK_ON_ERROR
"$PY" - <<'PY'
import torch
assert torch.__version__.startswith('2.7.1'), torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_name(0)
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY
"$PY" -m pip install --no-build-isolation --no-deps ./models/dino/ops
"$PY" migration/smoke_5090.py --output migration_reports/msda_smoke.json
"$PY" -m pip freeze > migration_reports/target_pip_freeze.txt
nvidia-smi > migration_reports/nvidia-smi.txt
nvcc --version > migration_reports/nvcc-version.txt
printf 'Environment and CUDA operator check passed. Python: %s\n' "$PY"
printf 'Next: transfer assets, then run migration/preflight_data.py. No training has been launched.\n'
