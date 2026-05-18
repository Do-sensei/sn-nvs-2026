#!/bin/bash
# Step 3: render challenge views with the per-scene best schedule
# and zip into a Codabench-compatible submission archive.
#
# Per-scene best configuration:
#   scene 1: 70K / 45K  (long)
#   scene 2: 50K / 27K  (short)
#   scene 3: 70K / 45K  (long)
#   scene 4: 50K / 27K  (short)
#   scene 5: 60K / 40K  (medium)   — 70K/45K OOMs on this scene
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SN_NVS_DIR="$REPO_ROOT/submodules/sn-nvs"

: "${GS_DIR:?set GS_DIR to the gaussian-splatting submodule path}"
: "${CHALLENGE_DATA_DIR:?set CHALLENGE_DATA_DIR to all_scenes_challenge_views}"
: "${OUT_DIR:?set OUT_DIR to the experiment output root used by training}"
: "${SUBMIT_DIR:?set SUBMIT_DIR to where to write the submission}"
: "${CONDA_ENV:=gaussian_splatting}"

export PYTHONPATH="$GS_DIR:${PYTHONPATH:-}"

# Per-scene (schedule subdir, iteration) chosen from the sweep
declare -A SCHED=(
    [1]="train_70K_45K"  [2]="train_50K_27K"
    [3]="train_70K_45K"  [4]="train_50K_27K"
    [5]="train_60K_40K"
)
declare -A ITER=(
    [1]=70000  [2]=50000
    [3]=70000  [4]=50000
    [5]=60000
)

mkdir -p "$SUBMIT_DIR"
cd "$SN_NVS_DIR"           # render_challenge.py lives in the sn-nvs submodule

echo "=== Render challenge views (per-scene best) ==="
for SCENE in 1 2 3 4 5; do
    MODEL="$OUT_DIR/${SCHED[$SCENE]}/scene-${SCENE}"
    ITER_S=${ITER[$SCENE]}
    echo "[RENDER] scene-${SCENE}  (${SCHED[$SCENE]}, iter=${ITER_S})"
    conda run -n "$CONDA_ENV" python render_challenge.py 3dgs \
        -s "$CHALLENGE_DATA_DIR/scene-${SCENE}-challenge" \
        -m "$MODEL" \
        -i images_2 -r 1 --antialiasing \
        --iteration "$ITER_S" --depths ""
done

echo "=== Packaging ==="
for SCENE in 1 2 3 4 5; do
    SRC="$OUT_DIR/${SCHED[$SCENE]}/scene-${SCENE}/challenge/ours_${ITER[$SCENE]}/renders"
    DST="$SUBMIT_DIR/scene-${SCENE}-challenge/renders"
    mkdir -p "$DST"
    cp "$SRC"/*.png "$DST/"
    echo "scene-${SCENE}: $(ls "$DST"/*.png | wc -l) images"
done

cd "$SUBMIT_DIR"
ZIP_NAME="submission.zip"
rm -f "$ZIP_NAME"
# Use `find ... | zip -D` to avoid directory entries (Codabench requirement).
find scene-*-challenge -type f -name "*.png" | sort | zip -D "$ZIP_NAME" -@ > /dev/null
echo "=== DONE: $SUBMIT_DIR/$ZIP_NAME ==="
