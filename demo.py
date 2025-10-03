import os
import sys

import torch.utils.checkpoint
from torch.utils.data import SequentialSampler
import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

import argparse
import logging
from typing import Literal, Optional


import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel
)


from diffusers.utils import export_to_video, load_image, load_video

from pipeline.cognvs_pipeline import CogNVSPipeline

import cv2
import numpy as np
import torch
from PIL import Image
import glob



def main(args):

    hyperparams = {
        "num_frames": 49,
        "height": 480,
        "width": 720,
        "guidance_scale": 6
    }

    model_id = args.model_path
    transformer_id = args.cognvs_ckpt_path # we recommend using the test-time fine-tuned ckpt for better results

    transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_id, torch_dtype=torch.bfloat16)
    pipe = CogNVSPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    mp4_paths =  sorted(glob.glob(os.path.join(args.data_path, args.mp4_name)))  # you can change to process multiple mp4s, e.g., "*.mp4"
    output_paths = os.path.join(args.data_path, "outputs")
    
    if not os.path.exists(output_paths):
        os.makedirs(output_paths)
    

    for idx, mp4_path in enumerate(mp4_paths):

        print(f"Processing {mp4_path}...")
        
        # --- read first 49 frames and get original H, W ---
        cap = cv2.VideoCapture(mp4_path)
        ret, frame0 = cap.read()
        ori_h, ori_w = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = []
        for i in range(49):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()

        frames_np = np.stack(frames, axis=0)             # (49, 480, 720, 3)
        input_pixels = (
            torch.from_numpy(frames_np)
                .permute(0, 3, 1, 2)                    # → (49, 3, 480, 720)
                .unsqueeze(0)                           # → (1, 49, 3, 480, 720)
                .float()
                .div(127.5)
                .sub(1.0)
                .to("cuda")
        )

        # --- CogNVS inference ---
        video_frames = pipe(
                    prompt="",
                    images=input_pixels,
                    num_videos_per_prompt=1,
                    num_inference_steps=50,
                    num_frames=hyperparams["num_frames"],
                    guidance_scale=hyperparams["guidance_scale"],
                    generator=torch.Generator(device="cuda").manual_seed(42),
                    height=480,
                    width=720
                ).frames[0]

        video_frames = np.array(video_frames)
        video_frames = [cv2.resize(frame, (ori_w, ori_h)) for frame in video_frames]
        video_frames = [Image.fromarray(frame) for frame in video_frames]

        output_mp4_name = os.path.basename(mp4_path).replace(".mp4", "_out.mp4")
        export_to_video(video_frames,
                                os.path.join(output_paths, output_mp4_name),
                                10)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CogNVS demo script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory of CogVideoX-5b-I2V")
    parser.add_argument("--cognvs_ckpt_path", type=str, required=True, help="Path to the model directory of CogNVS")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data directory")
    parser.add_argument("--mp4_name", type=str, required=True, help="Path to the input data directory")
    
    args = parser.parse_args()

    main(args)
