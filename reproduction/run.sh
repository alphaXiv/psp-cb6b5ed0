#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir \
  "accelerate>=1.2,<2" \
  "diffusers @ git+https://github.com/huggingface/diffusers@af28ae2d5ba0ef80d99fff7859ebea730e1cf3f8" \
  "transformers==4.46.3" \
  "huggingface-hub>=0.27,<1" \
  "image-reward @ git+https://github.com/THUDM/ImageReward.git@2ca71bac4ed86b922fe53ddaec3109fe94d45fd3" \
  "hpsv2==1.2.0" \
  "google-genai>=1.0,<2" \
  "open-clip-torch==2.26.1" \
  "scipy>=1.11,<2" \
  "pandas>=2.1,<3" \
  "protobuf<5" \
  "sentencepiece>=0.2" \
  "ftfy>=6.1"

python - <<'PY'
import torch
print("ENVIRONMENT_JSON=" + __import__("json").dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
assert torch.cuda.is_available() and torch.cuda.device_count() == 4
PY

torchrun --standalone --nproc_per_node=4 reproduction/run_reproduction.py
