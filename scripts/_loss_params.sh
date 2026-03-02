#!/usr/bin/env bash
set -e

dataset_loss_params() {
  local dataset_type="$1"
  case "$dataset_type" in
    laptop14|lap14)
      BASE_POS_W="0.700"
      BASE_NEG_W="0.798"
      BASE_NEU_W="1.502"
      BASE_FOCAL_GAMMA="2.0"
      ;;
    rest14|restaurant14)
      BASE_POS_W="0.422"
      BASE_NEG_W="1.135"
      BASE_NEU_W="1.443"
      BASE_FOCAL_GAMMA="2.0"
      ;;
    rest15|restaurant15)
      BASE_POS_W="0.100"
      BASE_NEG_W="0.358"
      BASE_NEU_W="2.543"
      BASE_FOCAL_GAMMA="2.5"
      ;;
    rest16|restaurant16)
      BASE_POS_W="0.138"
      BASE_NEG_W="0.389"
      BASE_NEU_W="2.473"
      BASE_FOCAL_GAMMA="2.5"
      ;;
    *)
      echo "❌ Unsupported dataset_type for loss params: ${dataset_type}"
      echo "Supported: laptop14 | lap14 | rest14 | rest15 | rest16"
      exit 1
      ;;
  esac
}

_normalize_label() {
  local label="${1,,}"
  case "$label" in
    pos|positive)
      echo "pos"
      return 0
      ;;
    neg|negative)
      echo "neg"
      return 0
      ;;
    neu|neutral)
      echo "neu"
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

confirm_label_order_and_build_weights() {
  if [[ -z "${BASE_POS_W:-}" || -z "${BASE_NEG_W:-}" || -z "${BASE_NEU_W:-}" ]]; then
    echo "❌ Base weights are not set. Call dataset_loss_params first."
    exit 1
  fi

  local mapping_output=""
  mapping_output="$(python - "$@" <<'PY'
import json
import sys

from src.core.config import Config
from src.core.utils.helper import get_tokenizer, get_dataset

cfg = Config.from_cli(sys.argv[1:]).finalize().validate()
tokenizer = get_tokenizer(cfg)
train_set, _ = get_dataset(cfg, tokenizer)
label2id = train_set.label2id
id2label = {int(v): k for k, v in label2id.items()}

print("LABEL2ID_RAW=" + json.dumps(label2id, ensure_ascii=True))
print("ID2LABEL_RAW=" + json.dumps(id2label, ensure_ascii=True))
for key, val in label2id.items():
    print(f"LABEL2ID_ITEM\t{key}\t{val}")
PY
)" || {
    echo "❌ Failed to read label2id via Config.from_cli().finalize().validate()."
    exit 1
  }

  local raw_label2id=""
  local raw_id2label=""
  local -A label_by_id=()
  local item_count=0
  local line=""
  while IFS= read -r line; do
    case "$line" in
      LABEL2ID_RAW=*)
        raw_label2id="${line#LABEL2ID_RAW=}"
        ;;
      ID2LABEL_RAW=*)
        raw_id2label="${line#ID2LABEL_RAW=}"
        ;;
      LABEL2ID_ITEM$'\t'*)
        IFS=$'\t' read -r _ label id <<< "$line"
        if [[ -z "$label" || -z "$id" ]]; then
          continue
        fi
        local norm_label=""
        if ! norm_label="$(_normalize_label "$label")"; then
          echo "❌ Unknown label name: ${label}"
          echo "Raw label2id: ${raw_label2id}"
          exit 1
        fi
        label_by_id["$id"]="$norm_label"
        item_count=$((item_count + 1))
        ;;
    esac
  done <<< "$mapping_output"

  if [[ "$item_count" -ne 3 ]]; then
    echo "❌ Expected exactly 3 labels, got ${item_count}."
    echo "Raw label2id: ${raw_label2id}"
    echo "Raw id2label: ${raw_id2label}"
    exit 1
  fi

  local ordered_labels=()
  local ordered_weights=()
  local id=""
  for id in 0 1 2; do
    local label="${label_by_id[$id]}"
    if [[ -z "$label" ]]; then
      echo "❌ Missing label for id=${id}."
      echo "Raw label2id: ${raw_label2id}"
      echo "Raw id2label: ${raw_id2label}"
      exit 1
    fi
    ordered_labels+=("$label")
    case "$label" in
      pos)
        ordered_weights+=("$BASE_POS_W")
        ;;
      neg)
        ordered_weights+=("$BASE_NEG_W")
        ;;
      neu)
        ordered_weights+=("$BASE_NEU_W")
        ;;
      *)
        echo "❌ Unsupported normalized label: ${label}"
        echo "Raw label2id: ${raw_label2id}"
        echo "Raw id2label: ${raw_id2label}"
        exit 1
        ;;
    esac
  done

  CLASS_WEIGHTS="$(IFS=,; echo "${ordered_weights[*]}")"
  FOCAL_GAMMA="${BASE_FOCAL_GAMMA}"

  echo "LABEL_ORDER: id0=${ordered_labels[0]} id1=${ordered_labels[1]} id2=${ordered_labels[2]} -> CLASS_WEIGHTS=${CLASS_WEIGHTS}"
}
