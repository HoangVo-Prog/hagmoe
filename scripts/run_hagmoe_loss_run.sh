#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_hagmoe_loss_run.sh \
    --run_id <run_id> \
    --dataset_name <dataset_name> \
    --loss_type <weighted_ce|focal> \
    --class_weights <w_pos,w_neg,w_neu> \
    [--focal_gamma <gamma>] \
    [--early_stop_patience <int>] \
    [--cuda <id>] \
    [--model_name <hf_model>] \
    [--fusion_methods <csv_methods>] \
    [--use_nohup 1|0]

Notes:
  --early_stop_patience default is 5.
USAGE
}

run_id=""
dataset_name=""
loss_type=""
class_weights=""
focal_gamma=""
cuda="0"
model_name="bert-base-uncased"
fusion_methods="concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction"
use_nohup="1"
early_stop_patience=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_id)
      run_id="$2"
      shift 2
      ;;
    --dataset_name)
      dataset_name="$2"
      shift 2
      ;;
    --loss_type)
      loss_type="$2"
      shift 2
      ;;
    --class_weights)
      class_weights="$2"
      shift 2
      ;;
    --focal_gamma)
      focal_gamma="$2"
      shift 2
      ;;
    --early_stop_patience)
      early_stop_patience="$2"
      shift 2
      ;;
    --cuda)
      cuda="$2"
      shift 2
      ;;
    --model_name)
      model_name="$2"
      shift 2
      ;;
    --fusion_methods)
      fusion_methods="$2"
      shift 2
      ;;
    --use_nohup)
      use_nohup="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$run_id" || -z "$dataset_name" || -z "$loss_type" || -z "$class_weights" ]]; then
  echo "Missing required arguments."
  usage
  exit 1
fi

if [[ -n "$early_stop_patience" ]]; then
  if ! [[ "$early_stop_patience" =~ ^[0-9]+$ ]]; then
    echo "Invalid --early_stop_patience: $early_stop_patience (must be integer >= 0)"
    exit 1
  fi
fi

case "$dataset_name" in
  Lap14)
    dataset_type="laptop14"
    DATASET_TAG="Lap14"
    ;;
  Rest14)
    dataset_type="rest14"
    DATASET_TAG="Rest14"
    ;;
  Rest15)
    dataset_type="rest15"
    DATASET_TAG="Rest15"
    ;;
  Rest16)
    dataset_type="rest16"
    DATASET_TAG="Rest16"
    ;;
  *)
    echo "Unsupported dataset_name: $dataset_name"
    echo "Supported: Lap14 | Rest14 | Rest15 | Rest16"
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"
TRAIN_PATH="$ROOT_DIR/data/${DATASET_TAG}_Train.json"
TEST_PATH="$ROOT_DIR/data/${DATASET_TAG}_Test.json"

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "Missing train_path: $TRAIN_PATH"
  exit 1
fi
if [[ ! -f "$TEST_PATH" ]]; then
  echo "Missing test_path: $TEST_PATH"
  exit 1
fi

OUTPUT_DIR="$ROOT_DIR/results/loss_tuning/${dataset_type}/${loss_type}/${run_id}"
OUTPUT_NAME="hagmoe_${run_id}.json"
LOG_DIR="$ROOT_DIR/logs/loss_tuning/${dataset_type}/${loss_type}"
LOG_FILE="$LOG_DIR/${run_id}_cuda${cuda}.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

COMMON_ARGS=(
  --mode "HAGMoE"
  --debug_aspect_span
  --model_name "${model_name}"
  --run_mode single
  --benchmark_methods "${fusion_methods}"
  --output_dir "${OUTPUT_DIR}"
  --output_name "${OUTPUT_NAME}"
  --train_path "${TRAIN_PATH}"
  --test_path "${TEST_PATH}"
)

COMMON_ARGS+=(--early_stop_patience "${early_stop_patience}")

LOSS_FLAGS="--loss_type ${loss_type}"
case "$loss_type" in
  weighted_ce)
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights ${class_weights}"
    ;;
  focal)
    if [[ -z "$focal_gamma" ]]; then
      echo "Missing --focal_gamma for loss_type=focal"
      exit 1
    fi
    LOSS_FLAGS="${LOSS_FLAGS} --class_weights ${class_weights} --focal_gamma ${focal_gamma}"
    ;;
  *)
    echo "Unsupported loss_type: $loss_type"
    echo "Supported: weighted_ce | focal"
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES="${cuda}"

echo "▶ Running HAGMoE loss run"
echo "  run_id        = ${run_id}"
echo "  dataset_name  = ${dataset_name}"
echo "  dataset_type  = ${dataset_type}"
echo "  loss_type     = ${loss_type}"
echo "  class_weights = ${class_weights}"
echo "  focal_gamma   = ${focal_gamma}"
echo "  early_stop_patience = ${early_stop_patience}"
echo "  cuda          = ${cuda}"
echo "  model_name    = ${model_name}"
echo "  fusion_methods= ${fusion_methods}"
echo "  output_dir    = ${OUTPUT_DIR}"
echo "  log_file      = ${LOG_FILE}"
echo

if [[ "$use_nohup" == "1" ]]; then
  nohup python -m main "${COMMON_ARGS[@]}" ${LOSS_FLAGS} > "${LOG_FILE}" 2>&1 &
  pid=$!
  echo "PID=${pid}"
  wait "${pid}"
  exit $?
else
  python -m main "${COMMON_ARGS[@]}" ${LOSS_FLAGS} 2>&1 | tee "${LOG_FILE}"
fi
