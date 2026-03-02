#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/grid_loss_tuning.sh \
    --mode weighted_ce|focal \
    [--dataset_name Lap14|Rest14|Rest15|Rest16] \
    --csv <path_to_loss_runs_csv> \
    [--n-runs <N>] \
    [--ith-run <I>] \
    [--cuda <id>] \
    [--model_name <hf_model>] \
    [--fusion_methods <csv_methods>] \
    [--use_nohup 1|0]

Notes:
  --ith-run is the 1-based start index in filtered CSV rows.
  Negative --ith-run counts from the end and runs backward.
  Example: with 60 rows, --ith-run -2 --n-runs 2 runs rows 59 then 58.
USAGE
}

mode="weighted_ce"
csv_path=""
dataset_name=""
cuda="0"
model_name="bert-base-uncased"
fusion_methods="concat,add,mul,cross,gated_concat,bilinear,coattn,late_interaction"
use_nohup="1"
detached="0"
n_runs="1"
ith_run="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --dataset_name)
      dataset_name="$2"
      shift 2
      ;;
    --csv)
      csv_path="$2"
      shift 2
      ;;
    --n-runs)
      n_runs="$2"
      shift 2
      ;;
    --ith-run)
      ith_run="$2"
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
    --_detached)
      detached="$2"
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

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -z "$csv_path" ]]; then
  if [[ -f "$ROOT_DIR/loss_runs.csv" ]]; then
    csv_path="$ROOT_DIR/loss_runs.csv"
  else
    echo "Missing --csv and default loss_runs.csv not found at $ROOT_DIR/loss_runs.csv"
    exit 1
  fi
fi

if [[ "$mode" != "weighted_ce" && "$mode" != "focal" ]]; then
  echo "Unsupported mode: $mode"
  echo "Supported: weighted_ce | focal"
  exit 1
fi

if [[ -n "$dataset_name" ]]; then
  case "$dataset_name" in
    Lap14|lap14|laptop14)
      dataset_name="Lap14"
      ;;
    Rest14|rest14)
      dataset_name="Rest14"
      ;;
    Rest15|rest15)
      dataset_name="Rest15"
      ;;
    Rest16|rest16)
      dataset_name="Rest16"
      ;;
    *)
      echo "Unsupported dataset_name: $dataset_name"
      echo "Supported: Lap14 | Rest14 | Rest15 | Rest16"
      exit 1
      ;;
  esac
fi

if [[ ! -f "$csv_path" ]]; then
  echo "CSV not found: $csv_path"
  exit 1
fi

if ! [[ "$n_runs" =~ ^[0-9]+$ ]] || [[ "$n_runs" -lt 1 ]]; then
  echo "Invalid --n-runs: $n_runs (must be integer >= 1)"
  exit 1
fi

if ! [[ "$ith_run" =~ ^-?[0-9]+$ ]]; then
  echo "Invalid --ith-run: $ith_run (must be a non-zero integer)"
  exit 1
fi

if [[ "$ith_run" -eq 0 ]]; then
  echo "Invalid --ith-run: 0 (must be a non-zero integer)"
  exit 1
fi

if [[ ! -x "$ROOT_DIR/scripts/run_hagmoe_loss_run.sh" ]]; then
  echo "Missing or non-executable: $ROOT_DIR/scripts/run_hagmoe_loss_run.sh"
  exit 1
fi

if [[ "$use_nohup" == "1" && "$detached" == "0" ]]; then
  LOG_DIR="$ROOT_DIR/logs/loss_tuning"
  mkdir -p "$LOG_DIR"
  GRID_LOG="$LOG_DIR/grid_${mode}_cuda${cuda}.log"
  cmd=(
    nohup bash "$ROOT_DIR/scripts/grid_loss_tuning.sh"
    --mode "$mode"
    --dataset_name "$dataset_name"
    --csv "$csv_path"
    --n-runs "$n_runs"
    --ith-run "$ith_run"
    --cuda "$cuda"
    --model_name "$model_name"
    --fusion_methods "$fusion_methods"
    --use_nohup 0
    --_detached 1
  )
  "${cmd[@]}" > "$GRID_LOG" 2>&1 &
  pid=$!
  echo "Detached grid run PID=${pid}"
  echo "Grid log: $GRID_LOG"
  if [[ -n "$dataset_name" ]]; then
    echo "Dataset filter: $dataset_name"
  fi
  exit 0
fi

mapfile -t rows < <(python - "$csv_path" "$mode" "$dataset_name" "$n_runs" "$ith_run" <<'PY'
import csv, sys
path=sys.argv[1]
mode=sys.argv[2]
dataset_name=sys.argv[3]
n_runs=int(sys.argv[4])
ith_run=int(sys.argv[5])
rows=[]
with open(path, newline="") as f:
    r=csv.DictReader(f)
    for row in r:
        if row.get("loss_type") != mode:
            continue
        if dataset_name and row.get("dataset_name") != dataset_name:
            continue
        rows.append(row)

selected=[]
total=len(rows)
if ith_run > 0:
    start=ith_run-1
    if 0 <= start < total:
        end=min(start+n_runs, total)
        selected=rows[start:end]
else:
    start=total + ith_run
    if 0 <= start < total:
        stop=max(start - n_runs + 1, 0)
        for idx in range(start, stop - 1, -1):
            selected.append(rows[idx])

for row in selected:
        gamma=row.get("focal_gamma", "") or ""
        if gamma.lower() == "nan":
            gamma = ""
        print("\t".join([
            row.get("run_id", ""),
            row.get("dataset_name", ""),
            row.get("loss_type", ""),
            row.get("class_weights", ""),
            gamma,
        ]))
PY
)

total=${#rows[@]}
if [[ "$total" -eq 0 ]]; then
  if [[ -n "$dataset_name" ]]; then
    echo "No runs selected for mode=$mode dataset_name=$dataset_name in $csv_path (ith-run=$ith_run n-runs=$n_runs)"
  else
    echo "No runs selected for mode=$mode in $csv_path (ith-run=$ith_run n-runs=$n_runs)"
  fi
  exit 0
fi

if [[ -n "$dataset_name" ]]; then
  echo "Total selected runs for mode=$mode dataset_name=$dataset_name: $total (ith-run=$ith_run n-runs=$n_runs)"
else
  echo "Total selected runs for mode=$mode: $total (ith-run=$ith_run n-runs=$n_runs)"
fi

i=0
for line in "${rows[@]}"; do
  IFS=$'\t' read -r run_id dataset_name loss_type class_weights focal_gamma <<< "$line"
  i=$((i + 1))

  case "$dataset_name" in
    Lap14)
      dataset_type="laptop14"
      ;;
    Rest14)
      dataset_type="rest14"
      ;;
    Rest15)
      dataset_type="rest15"
      ;;
    Rest16)
      dataset_type="rest16"
      ;;
    *)
      echo "Unsupported dataset_name: $dataset_name"
      exit 1
      ;;
  esac

  echo "[${i}/${total}] run_id=${run_id} dataset=${dataset_name} loss=${loss_type}"

  OUT_JSON="$ROOT_DIR/results/loss_tuning/${dataset_type}/${loss_type}/${run_id}/hagmoe_${run_id}.json"
  if [[ -s "$OUT_JSON" ]]; then
    echo "SKIP existing: $run_id"
    continue
  fi

  cmd=(
    bash "$ROOT_DIR/scripts/run_hagmoe_loss_run.sh"
    --run_id "$run_id"
    --dataset_name "$dataset_name"
    --loss_type "$loss_type"
    --class_weights "$class_weights"
    --early_stop_patience 5
    --cuda "$cuda"
    --model_name "$model_name"
    --fusion_methods "$fusion_methods"
    --use_nohup "$use_nohup"
  )

  if [[ -n "$focal_gamma" ]]; then
    cmd+=(--focal_gamma "$focal_gamma")
  fi

  "${cmd[@]}"
done
