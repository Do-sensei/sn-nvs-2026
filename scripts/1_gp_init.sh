#!/bin/bash
# Step 1: GaussianPro PatchMatch depth propagation.
#
# Runs GaussianPro's *native* train.py (from the GaussianPro submodule)
# for up to 12K iterations per scene, saving a checkpoint every 500
# iters. Densification typically runs out of VRAM before 12K; the last
# surviving checkpoint (= the densest cloud) is then converted to a
# COLMAP-style points3D.ply and written to $DATA_DIR/scene-{1..5}-gp
# for step 2.
#
# Required env (see README "Prerequisites"):
#   GP_DIR    Path to the GaussianPro submodule (its train.py is run here)
#   DATA_DIR  Path to SN-NVS-2026/all_scenes_train_views
#   OUT_DIR   Experiment output root
#   CONDA_ENV Python 3.10 env with the PatchMatch CUDA extension built
#             (default: gaussianpro; separate from the gaussian_splatting
#             env used in steps 2-3)
#
# Output: $DATA_DIR/scene-{1..5}-gp/sparse/0/points3D.ply
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${GP_DIR:?set GP_DIR to the GaussianPro submodule path}"
: "${DATA_DIR:?set DATA_DIR to all_scenes_train_views}"
: "${OUT_DIR:?set OUT_DIR to an experiment output root}"
: "${CONDA_ENV:=gaussianpro}"

OUT_ROOT="$OUT_DIR/gp_init"
mkdir -p "$OUT_ROOT/logs"

CONV="$REPO_ROOT/scripts/gp_ply_to_colmap.py"

# Checkpoint every 500 iters up to the 12K target. Densification OOMs
# before then on most scenes; whichever is the last to land is used.
SAVE_ITERS="500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 \
            6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000"
COMMON="-i images_2 -r 1 --eval --dataset free --depth_loss"

# Build $DATA_DIR/scene-X-gp from the densest surviving checkpoint:
# reuse scene-X cameras/images via symlink, convert the Gaussian-splat
# ply to a COLMAP-style points3D.ply.
make_gp_scene() {
    local SCENE="$1"
    local SRC_SCENE="$DATA_DIR/scene-${SCENE}"
    local GP_SCENE="$DATA_DIR/scene-${SCENE}-gp"
    local MODEL="$OUT_ROOT/scene-${SCENE}"

    local LAST_ITER
    # `|| true`: with `set -euo pipefail`, a no-match glob would abort the
    # script before the explicit check below could report a clear error.
    LAST_ITER=$(ls -d "$MODEL"/point_cloud/iteration_* 2>/dev/null \
        | sed 's/.*iteration_//' | sort -n | tail -1) || true
    if [ -z "${LAST_ITER:-}" ]; then
        echo "[ERROR] No saved checkpoint for scene-${SCENE} in $MODEL" >&2
        exit 1
    fi
    local GP_PLY="$MODEL/point_cloud/iteration_${LAST_ITER}/point_cloud.ply"

    mkdir -p "$GP_SCENE/sparse/0"
    for NAME in images_2 depth_maps_2; do
        if [ -e "$SRC_SCENE/$NAME" ] && [ ! -e "$GP_SCENE/$NAME" ]; then
            ln -s "../scene-${SCENE}/$NAME" "$GP_SCENE/$NAME"
        fi
    done
    for NAME in cameras.bin images.bin points3D.bin depth_params.json; do
        if [ -e "$SRC_SCENE/sparse/0/$NAME" ] && [ ! -e "$GP_SCENE/sparse/0/$NAME" ]; then
            (cd "$GP_SCENE/sparse/0" && ln -s "../../../scene-${SCENE}/sparse/0/$NAME" "$NAME")
        fi
    done

    conda run -n "$CONDA_ENV" python "$CONV" \
        "$GP_PLY" "$GP_SCENE/sparse/0/points3D.ply"
    echo "[GP scene] scene-${SCENE}: iter_${LAST_ITER} -> $GP_SCENE/sparse/0/points3D.ply"
}

# Apply our GaussianPro patch onto the official (pinned) submodule:
# spatial K-NN source-view selection + SIMPLE_PINHOLE K-matrix fix +
# num_neighbors option. Without it propagation falls back to
# index-based selection and the result will not match the report.
# Idempotent: skip if already applied.
PATCH="$REPO_ROOT/patches/gaussianpro_knn.patch"
if git -C "$GP_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[patch] GaussianPro patch already applied — skip"
elif git -C "$GP_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$GP_DIR" apply "$PATCH"
    echo "[patch] GaussianPro spatial-KNN patch applied to $GP_DIR"
else
    echo "[ERROR] $PATCH neither applies cleanly nor is already applied" >&2
    echo "        Check the GaussianPro submodule is at its pinned commit." >&2
    exit 1
fi

cd "$GP_DIR"
for SCENE in 1 2 3 4 5; do
    JOB="gp_init__scene-${SCENE}"
    MODEL="$OUT_ROOT/scene-${SCENE}"
    LOG="$OUT_ROOT/logs/${JOB}.log"

    if [ -f "$DATA_DIR/scene-${SCENE}-gp/sparse/0/points3D.ply" ]; then
        echo "[SKIP] $JOB"
        continue
    fi

    echo "[GP init] scene-${SCENE}"
    # OOM during densification is expected and not fatal: checkpoints are
    # already on disk, so swallow the failure and pick the last one.
    conda run -n "$CONDA_ENV" python -u train.py \
        -s "$DATA_DIR/scene-${SCENE}" -m "$MODEL" \
        $COMMON --save_iterations $SAVE_ITERS \
        --iterations 12000 --test_iterations 12000 \
        2>&1 | tee "$LOG" || true
    make_gp_scene "$SCENE"
    echo "[DONE] $JOB"
done
