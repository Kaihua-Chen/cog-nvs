#!/bin/bash
set -e

BASE_PATH="./cognvs_eval_results/"
SCRIPT="eval_metrics.py"


################################
# kubric_4d
################################
DATASET="kubric_4d"
METHODS=("cognvs" "gcd" "trajcrafter" "gt_render")

for METHOD in "${METHODS[@]}"; do
    echo "Running ${DATASET} | ${METHOD}"
    python ${SCRIPT} \
        --base_path ${BASE_PATH} \
        --dataset ${DATASET} \
        --pred_method ${METHOD}
done

################################
# pardom_4d
################################
DATASET="pardom_4d"
METHODS=("cognvs" "gcd" "trajcrafter" "gt_render")

for METHOD in "${METHODS[@]}"; do
    echo "Running ${DATASET} | ${METHOD}"
    python ${SCRIPT} \
        --base_path ${BASE_PATH} \
        --dataset ${DATASET} \
        --pred_method ${METHOD}
done

################################
# dycheck (360p + 720p)
################################
DATASET="dycheck"
METHODS=("cognvs" "trajcrafter" "megasam_render" "shape_of_motion" "mosca")
RESOLUTIONS=("360p" "720p")

for METHOD in "${METHODS[@]}"; do
    for RES in "${RESOLUTIONS[@]}"; do
        echo "Running ${DATASET} | ${METHOD} | ${RES}"
        python ${SCRIPT} \
            --base_path ${BASE_PATH} \
            --dataset ${DATASET} \
            --pred_method ${METHOD} \
            --dycheck_resolution ${RES}
    done
done

echo "All evaluations finished."
