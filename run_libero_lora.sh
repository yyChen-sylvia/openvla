#!/usr/bin/env bash
# set -euo pipefail

# Resolve repo root based on script location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
mkdir -p hf_cache

# Redirect HF cache to project disk (avoid small $HOME quota)
export HF_HOME="${ROOT_DIR}/hf_cache"

# Use EGL for consistency with eval (harmless for training)
export MUJOCO_GL=egl

# Set model type for ACTION_TOKEN_BEGIN_IDX detection
export VLA_MODEL_TYPE=OPENVLA
export PYOPENGL_PLATFORM=egl

# Auto-activate local virtualenv if present
if [ -f "${ROOT_DIR}/env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/env/bin/activate"
fi

# Prefer the project-local torchrun if available
TORCHRUN_BIN="${ROOT_DIR}/env/bin/torchrun"
if [ ! -x "${TORCHRUN_BIN}" ]; then
  TORCHRUN_BIN="torchrun"
fi

# Configuration
data_name=libero_object_no_noops
current_time=$(date "+%Y-%m-%d_%H-%M-%S")


"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /datasets/zhangguoxi/modified_libero_rlds \
  --dataset_name libero_object_no_noops \
  --run_root_dir outputs \
  --adapter_tmp_dir ./outputs/adapter_tmp \
  --lora_rank 32 \
  --batch_size 12 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla-lora-origin \
  --wandb_entity yiyangchen-sylvia-bigai \
  --merge_lora_during_training False \
  --save_steps 10000 \
> "logs/openvla-lora-${data_name}--${current_time}.log" 2>&1