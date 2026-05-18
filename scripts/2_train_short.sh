#!/bin/bash
# Step 2 (short schedule): antialiased 3DGS training,
#   T = 50,000 iterations, T_du = 27,000 (densification cutoff).
# Initialized from the GaussianPro-propagated cloud produced in step 1.
#
# Required env (see 1_gp_init.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${GS_DIR:?set GS_DIR to the gaussian-splatting submodule path}"
: "${DATA_DIR:?set DATA_DIR to all_scenes_train_views}"
: "${OUT_DIR:?set OUT_DIR to an experiment output root}"
: "${CONDA_ENV:=gaussian_splatting}"

export PYTHONPATH="$GS_DIR:${PYTHONPATH:-}"

OUT_ROOT="$OUT_DIR/train_50K_27K"
mkdir -p "$OUT_ROOT/logs"
cd "$REPO_ROOT"

ITER=50000
DU=27000
COMMON="-i images_2 -r 1 --antialiasing --disable_viewer"
FLAGS="--lambda_dssim 0.22 --densify_grad_threshold 0.00012 \
       --densification_interval 125 --densify_until_iter ${DU} \
       --iterations ${ITER} --test_iterations ${ITER} \
       --save_iterations ${ITER}"

for SCENE in 1 2 3 4 5; do
    JOB="train__scene-${SCENE}"
    MODEL="$OUT_ROOT/scene-${SCENE}"
    LOG="$OUT_ROOT/logs/${JOB}.log"
    SRC="$DATA_DIR/scene-${SCENE}-gp"

    if [ -f "$MODEL/point_cloud/iteration_${ITER}/point_cloud.ply" ]; then
        echo "[SKIP] $JOB"
        continue
    fi

    if [ ! -f "$SRC/sparse/0/points3D.ply" ]; then
        echo "[ERROR] Missing GP scene: $SRC/sparse/0/points3D.ply" >&2
        echo "        Run scripts/1_gp_init.sh first (it creates scene-${SCENE}-gp)." >&2
        exit 1
    fi

    echo "[TRAIN 50K/27K] scene-${SCENE}"
    conda run -n "$CONDA_ENV" python -u "$GS_DIR/train.py" \
        -s "$SRC" -m "$MODEL" \
        $COMMON $FLAGS \
        2>&1 | tee "$LOG"
    echo "[DONE] $JOB"
done
