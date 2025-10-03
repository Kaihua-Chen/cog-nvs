# Reconstruct, Inpaint, Finetune: Dynamic Novel-view Synthesis from Monocular Videos

[*Kaihua Chen<sup>*</sup>*](https://www.linkedin.com/in/kaihuac/), [*Tarasha Khurana<sup>*</sup>*](https://www.cs.cmu.edu/~tkhurana/), [*Deva Ramanan*](https://www.cs.cmu.edu/~deva/)

This repository contains the official implementation of **CogNVS**.

![Teaser animation](assets/cognvs.gif)

## TODO

- [x] Release CogNVS inference pipeline and checkpoints

- [x] Release self-supervised data generation code

- [x] Release CogNVS test-time finetuning code

## 1. Getting Started

### 1.1 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Kaihua-Chen/cog-nvs
cd cog-nvs
conda create --name cognvs python=3.11
conda activate cognvs
pip install -r cognvs_requirements.txt
```

### 1.2 Download Checkpoints

1. **CogVideoX base model**
    Download the original CogVideoX-5b-I2V checkpoints from:
    https://huggingface.co/zai-org/CogVideoX-5b-I2V

2. **CogNVS inpainting checkpoint**
    We provide CogNVS inpainting checkpoints, which can be used for further test-time finetuning on the sequences you want:

   ```bash
   mkdir checkpoints
   cd checkpoints
   git lfs install
   git clone https://huggingface.co/kaihuac/cognvs_ckpt_inpaint
   cd ..
   ```

3. **Test-time finetuned checkpoints**
    Please refer to Step 3 "Self-supervised Data Pair Generation" to generate training pairs and then follow Step 4 "Test-time Finetuning" to finetune our inpainting checkpoints on your target sequence.

   We also provide checkpoints already finetuned on our `demo_data`. If you want to skip test-time finetuning, download them (~20GB each) from:

   ```bash
   git clone https://huggingface.co/kaihuac/cognvs_ckpt_test_time_finetuned
   ```

## 2. Inference

You can run inference in three ways:

- Use the CogNVS inpainting checkpoint directly (not recommended; only for quick test, quality is usually lower)
- Download and use our provided test-time finetuned checkpoints
- Perform your own test-time finetuning (following instructions in later sections) and run inference afterward

Example using a provided test-time finetuned checkpoint:

```bash
python demo.py \
    --model_path "checkpoints/CogVideoX-5b-I2V" \
    --cognvs_ckpt_path "checkpoints/cognvs_ckpt_finetuned_bear" \
    --data_path "demo_data/bear" \
    --mp4_name "example_eval_render.mp4"
```

The output will be saved to:

```
demo_data/bear/outputs/sample_eval_render_out.mp4
```

## 3. Self-supervised Data Pair Generation

1. Sequence folder structure

```
sequence_name/
├─ gt_rgb.mp4
└─ cam_info/
   └─ megasam_depth.npy
   └─ megasam_intrinsics.npy (optional)
   └─ megasam_c2ws.npy (optional)
```

2. Generate training pairs

```bash
python data_gen.py \
    --device "cuda:0" \
    --data_path "demo_data/bear" \
    --mode "train" \
    --intrinsics_file "cam_info/megasam_intrinsics.npy" \
    --extrinsics_file "cam_info/megasam_c2ws.npy"
```

(`intrinsics_file` and `extrinsics_file` are optional. The pipeline still works if you only provide the depth file from MegaSAM, DepthCrafter, etc.)

3. Generate evaluation pairs

```bash
python data_gen.py \
    --device "cuda:0" \
    --data_path "demo_data/bear" \
    --mode "eval"
```

Evaluation renders will be created from predefined trajectories in the `trajs/` folder. You can customize trajectories by editing those `.txt` files.

4. Convert MegaSAM output

After running **[MegaSAM](https://github.com/mega-sam/mega-sam)**, you will get `{seq}_sgd_cvd_hr.npz`. To convert it into the `cam_info` folder format, run:

```bash
python toolbox/convert_megasam_outputs.py
```

## 4. Test-time Finetuning

After generating training pairs, edit the config files and run test-time finetuning:

1. Edit `finetune/finetune_cognvs.sh`:

   - `model_path`: path to CogVideoX-5b-I2V checkpoint
   - `transformer_id`: path to our CogNVS inpainting checkpoint
   - `output_dir`: path to save the finetuned checkpoint
   - `base_dir_input`: sequence folder with training pairs

   Optional parameters:

   - `train_epochs`: number of epochs
   - `checkpointing_steps`: steps to save checkpoints
   - `checkpointing_limit`: max number of checkpoints to keep
   - `do_validation`: set `True` to enable validation (slower)
   - `validation_steps`: steps to run validation

2. Edit `finetune/accelerate_config.yaml`:

   - `gpu_ids`: GPU ids for training
   - `num_processes`: must match number of GPU ids

3. Start finetuning:

```bash
cd finetune
sh finetune_cognvs.sh
```

⚠️ Note: We adopt DeepSpeed ZeRO-2 for finetuning, so it can fit into A6000 GPUs (48 GB), but you need ≥ 5 GPUs. For reference, 200 steps of finetuning take ~70 minutes on 8 A6000 Ada GPUs.

4. Process finetuned checkpoints

Place the following files from the `toolbox/` folder into the `checkpoints/` directory:

- `config.json`
- `diffusion_pytorch_model.safetensors.index.json`
- `process_ckpts.sh`

The structure should be:

```
checkpoints/
├── config.json
├── diffusion_pytorch_model.safetensors.index.json
├── process_ckpts.sh
└── cognvs_ckpt_finetuned_bear/
    └── checkpoint-200/
```

Edit `process_ckpts.sh` to match your checkpoint step:

```bash
CHECKPOINT_DIR="checkpoint-200"
```

Then run:

```bash
cd checkpoints
sh process_ckpts.sh
```

## Acknowledgements

Our work builds on **[CogVideoX](https://huggingface.co/zai-org/CogVideoX-5b-I2V)** and uses **[DeepSpeed ZeRO-2](https://www.deepspeed.ai/tutorials/zero/)** for memory-efficient finetuning. Video depth estimation adopts **[MegaSAM](https://github.com/mega-sam/mega-sam)** or **[DepthCrafter](https://github.com/Tencent/DepthCrafter)**. Concurrent research includes [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/), [CAT4D](https://cat-4d.github.io/), [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter), and [ReCamMaster](https://jianhongbai.github.io/ReCamMaster/). We thank the authors for their contributions.

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{chen2025cognvs,
  title     = {Reconstruct, Inpaint, Finetune: Dynamic Novel-view Synthesis from Monocular Videos},
  author    = {Chen, Kaihua and Khurana, Tarasha and Ramanan, Deva},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```