#!/bin/bash
# Unified experiment runner for UnderOneFacade baseline training.
#
# Usage:
#   ./run_experiments.sh sin_uk [device]          # Singapore -> Nottingham
#   ./run_experiments.sh uk_sin [device]          # Nottingham -> Singapore
#   ./run_experiments.sh resume [device] dir ...  # resume interrupted runs
#   ./run_experiments.sh retrain [device] dir ... # retrain in same output dirs
#
# device: auto (default) | cuda | cpu
set +e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CFG="$ROOT/config.yaml"

ALL_MODELS=("octformer" "ptv3" "kpconv" "dgcnn" "pointnet2" "ptv1")
LOFGS=("lofg2" "lofg3")
FEAT="rgbi"

CONDA_ENV="${CONDA_ENV:-underonefacade}"
PYTHON="${PYTHON:-python -u}"
if command -v conda >/dev/null 2>&1; then
    PYTHON="conda run --no-capture-output -n $CONDA_ENV python -u"
fi

usage() {
    sed -n '2,9p' "$0"
    exit 1
}

is_device() {
    [[ "$1" == "auto" || "$1" == "cuda" || "$1" == "cpu" ]]
}

run_train() {
    local train_countries="$1"
    local val_country="$2"
    local device="$3"
    local failed=""

    for model in "${ALL_MODELS[@]}"; do
        for lofg in "${LOFGS[@]}"; do
            echo "Training: $model | $lofg | $FEAT | $train_countries -> $val_country"
            if $PYTHON "$ROOT/train.py" \
                --model "$model" \
                --lofg "$lofg" \
                --features "$FEAT" \
                --train_countries "$train_countries" \
                --val_countries "$val_country" \
                --config "$CFG" \
                --device "$device"; then
                echo "Completed: $model | $lofg"
            else
                failed="$failed  - $model | $lofg\n"
            fi
        done
    done
    [[ -n "$failed" ]] && echo -e "FAILED runs:\n$failed"
}

infer_run_from_dir() {
    local name="$1"
    RUN_MODEL=$(echo "$name" | cut -d_ -f1)
    RUN_LOFG=$(echo "$name" | cut -d_ -f2)
    if [[ "$name" == *"_nottingham_"* ]]; then
        RUN_TRAIN="nottingham"
        RUN_VAL="singapore"
    elif [[ "$name" == *"_singapore_"* ]]; then
        RUN_TRAIN="singapore"
        RUN_VAL="nottingham"
    else
        return 1
    fi
    return 0
}

run_resume_dirs() {
    local mode="$1"
    local device="$2"
    shift 2
    local failed=""

    for rel in "$@"; do
        local out_dir="$ROOT/$rel"
        local name ckpt extra_args=()
        name=$(basename "$rel")

        if ! infer_run_from_dir "$name"; then
            echo "SKIP (unknown source in name): $rel"
            continue
        fi

        if [[ "$mode" == "resume" ]]; then
            ckpt="$out_dir/checkpoints/latest.pth"
            if [[ ! -f "$ckpt" ]]; then
                echo "SKIP (no latest.pth): $rel"
                continue
            fi
            extra_args=(--resume "$ckpt")
        fi

        echo "$mode: $RUN_MODEL | $RUN_LOFG | $FEAT | train=$RUN_TRAIN val=$RUN_VAL"
        if $PYTHON "$ROOT/train.py" \
            --model "$RUN_MODEL" \
            --lofg "$RUN_LOFG" \
            --features "$FEAT" \
            --train_countries "$RUN_TRAIN" \
            --val_countries "$RUN_VAL" \
            --config "$CFG" \
            --device "$device" \
            --out_dir "$out_dir" \
            "${extra_args[@]}"; then
            echo "Completed $mode: $RUN_MODEL | $RUN_LOFG"
        else
            failed="$failed  - $RUN_MODEL | $RUN_LOFG\n"
        fi
    done
    [[ -n "$failed" ]] && echo -e "FAILED runs:\n$failed"
}

STRATEGY="${1:-}"
[[ -z "$STRATEGY" ]] && usage

shift
DEVICE="auto"
if [[ $# -gt 0 ]] && is_device "$1"; then
    DEVICE="$1"
    shift
fi

case "$STRATEGY" in
    sin_uk)
        run_train "singapore" "nottingham" "$DEVICE"
        ;;
    uk_sin)
        run_train "nottingham" "singapore" "$DEVICE"
        ;;
    resume|retrain)
        [[ $# -eq 0 ]] && { echo "Provide one or more output dirs under outputs/train/"; exit 1; }
        run_resume_dirs "$STRATEGY" "$DEVICE" "$@"
        ;;
    *)
        usage
        ;;
esac
