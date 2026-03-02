#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

# =========================
# Inputs
# =========================
LOSS_TYPE="${1:-ce}"
DATASET_TYPE="${2:-laptop14}"

# =========================
# Script helpers
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_loss_params.sh"

# Only support these dataset types
case "${DATASET_TYPE}" in
  laptop14|lap14)
    DATASET_TAG="Lap14"
    ;;
  rest14)
    DATASET_TAG="Rest14"
    ;;
  rest15)
    DATASET_TAG="Rest15"
    ;;
  rest16)
    DATASET_TAG="Rest16"
    ;;
  *)
    echo "❌ Unsupported dataset_type: ${DATASET_TYPE}"
    echo "Supported: laptop14 | lap14 | rest14 | rest15 | rest16"
    exit 1
    ;;
esac

# =========================
# Project root
# =========================
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/bert_spc_model_${LOSS_TYPE}_${DATASET_TYPE}_${CUDA_VISIBLE_DEVICES}.log"
USE_NOHUP="${USE_NOHUP:-1}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
MODE="BERTSPCModel"
export PYTHONPATH="$ROOT_DIR/src"
mkdir -p "$LOG_DIR"
TRAIN_PATH="$ROOT_DIR/data/${DATASET_TAG}_Train.json"
TEST_PATH="$ROOT_DIR/data/${DATASET_TAG}_Test.json"

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "❌ Missing train_path: $TRAIN_PATH"
  exit 1
fi
if [[ ! -f "$TEST_PATH" ]]; then
  echo "❌ Missing test_path: $TEST_PATH"
  exit 1
fi

# =========================
# Dataset-specific weights/gamma (ABSA 3-class)
# Canonical order: (pos, neg, neu)
# =========================
FUSION_METHOD="${FUSION_METHOD:-spc}"
OUTPUT_DIR="$ROOT_DIR/results/${DATASET_TYPE}"
OUTPUT_NAME="bert_spc_model_${DATASET_TYPE}_${LOSS_TYPE}.json"
dataset_loss_params "${DATASET_TYPE}"
COMMON_ARGS=(
  --mode "${MODE}"
  --model_name "${MODEL_NAME}"
  --run_mode single
  --benchmark_methods "${FUSION_METHOD}"
  --output_dir "${OUTPUT_DIR}"
  --output_name "${OUTPUT_NAME}"
  --train_path "${TRAIN_PATH}"
  --test_path "${TEST_PATH}"
)
confirm_label_order_and_build_weights "${COMMON_ARGS[@]}"

# =========================
# Loss-specific flags
# =========================
LOSS_FLAGS="--loss_type ${LOSS_TYPE}"

case "${LOSS_TYPE}" in
  ce)
    # Plain CE
    ;;
  weighted_ce)
    # Weighted CE uses dataset-specific alpha
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights ${CLASS_WEIGHTS}"
    ;;
  focal)
    # Focal uses dataset-specific alpha and gamma
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights ${CLASS_WEIGHTS} --focal_gamma ${FOCAL_GAMMA}"
    ;;
  *)
    echo "❌ Unsupported loss_type: ${LOSS_TYPE}"
    echo "Supported: ce | weighted_ce | focal"
    exit 1
    ;;
esac

echo "▶ Running bert spc model with:"
echo "  dataset_type   = ${DATASET_TYPE}"
echo "  loss_type      = ${LOSS_TYPE}"
echo "  class_weights  = ${CLASS_WEIGHTS}"
echo "  focal_gamma    = ${FOCAL_GAMMA}"
echo "  loss_flags     = ${LOSS_FLAGS}"
echo "  fusion_method  = ${FUSION_METHOD}"
echo

# =========================
# Run
# =========================
# Smoke checks:
#   python -m main --run_mode single --train_path ... --test_path ... --epochs 1 --num_seeds 1
#   python -m main --run_mode single --train_path ... --test_path ... --epochs 1 --seed 42 --num_seeds 3
if [[ "$USE_NOHUP" == "1" ]]; then
  nohup python -m main \
  "${COMMON_ARGS[@]}" \
  ${LOSS_FLAGS} \
  > "$LOG_FILE" 2>&1 &
else
  python -m main \
  "${COMMON_ARGS[@]}" \
  ${LOSS_FLAGS}
fi
