#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Define which checkpoint folder to process (e.g., checkpoint-2000, checkpoint-3000, …)
CHECKPOINT_DIR="checkpoint-200"

# Source files required for fix step
SRC_CONFIG="config.json"
SRC_INDEX="diffusion_pytorch_model.safetensors.index.json"

# --- Step 1: Convert checkpoints ---
for dir in */; do
  seq="${dir%/}"
  [ -d "$seq/$CHECKPOINT_DIR" ] || { echo "Skipping ${seq}: no $CHECKPOINT_DIR"; continue; }

  echo "Converting in ${seq}…"
  (
    cd "$seq"
    python "$CHECKPOINT_DIR/zero_to_fp32.py" \
      --safe_serialization \
      "$CHECKPOINT_DIR/" \
      "my_${CHECKPOINT_DIR}_transformer/"
  )
done

# --- Step 2: Fix converted checkpoints ---
# Verify source files exist
for src in "$SRC_CONFIG" "$SRC_INDEX"; do
  if [[ ! -f "$src" ]]; then
    echo "Error: '$src' not found in $(pwd)"
    exit 1
  fi
done

find . -type d -name "my_${CHECKPOINT_DIR}_transformer" -print0 | \
while IFS= read -r -d '' ckpt_dir; do
  echo "Fixing $ckpt_dir …"

  cp "$SRC_CONFIG" "$ckpt_dir/"
  cp "$SRC_INDEX"  "$ckpt_dir/"

  for old in "$ckpt_dir"/model-*.safetensors; do
    [[ -e "$old" ]] || continue
    filename="${old##*/}"
    suffix="${filename#model-}"
    new="diffusion_pytorch_model-${suffix}"
    mv "$old" "$ckpt_dir/$new"
  done

  rm -f "$ckpt_dir/model.safetensors.index.json"
done

# --- Step 3: Delete old checkpoints ---
find . -type d -name "$CHECKPOINT_DIR" -prune -print0 | \
while IFS= read -r -d '' dir; do
  echo "Deleting $dir …"
  rm -rf "$dir"
done

echo "All steps finished."
