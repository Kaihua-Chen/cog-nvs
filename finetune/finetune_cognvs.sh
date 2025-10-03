# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "../checkpoints/CogVideoX-5b-I2V"
    --transformer_id "../checkpoints/cognvs_ckpt_inpaint"
    --model_name "cogvideox-v2v"
    --model_type "v2v"
    --training_type "sft"
)

# Output Configuration (per‐seq)
OUTPUT_ARGS=(
    --output_dir "../checkpoints/cognvs_ckpt_finetuned_davis_bear"
    --report_to "wandb"
)

# Data Configuration (per‐seq)
DATA_ARGS=(
    --json_file ""
    --base_dir_input "../demo_data/davis_bear"
    --base_dir_target ""
    --train_resolution "49x480x720"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 200
    --seed 42
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 200
    --checkpointing_limit 5
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation False
    --validation_dir ""
    --validation_steps 200
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# launch training
accelerate launch --main_process_port 29501  --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
