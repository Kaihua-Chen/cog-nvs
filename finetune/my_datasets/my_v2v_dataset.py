import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

import json
import os
from PIL import Image
import random
import pycocotools.mask as maskUtils
import cv2
import numpy as np
import glob

from io import BytesIO
import imageio.v3 as iio
import torch.nn.functional as F

random.seed(1234)

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class my_cognvs_dataset(Dataset):
    def __init__(self, base_dir_input, base_dir_target, mode, frames, height, width, device, trainer, random_sub_sample=True, transform=None):

        self.base_dir_input = base_dir_input
        self.base_dir_target = base_dir_target
        self.mode = mode

        self.trainer = trainer
        if self.trainer is not None:
            self.device = device
            self.encode_video = trainer.encode_video
            self.encode_text = trainer.encode_text

        self.num_samples = frames
        self.H = height
        self.W = width

        self.random_sub_sample = random_sub_sample

        # Define resizing transform
        self.default_transform = transforms.Compose([
            transforms.Resize((self.H, self.W)),
            transforms.ToTensor()  # Convert image to PyTorch tensor
        ])
        self.transform = transform

    def __len__(self):
        """Return the total number of sequences."""
        return 1

    def __getitem__(self, idx):

        if self.random_sub_sample:
            cam_idx = random.randint(1, 8)
        
        if self.mode == "train":
            input_video_path = os.path.join(self.base_dir_input, f"train_render{cam_idx}.mp4")
            target_video_path = os.path.join(self.base_dir_input, "gt_rgb.mp4")
        else:
            input_video_path = os.path.join(self.base_dir_input, f"eval_render1.mp4")
            target_video_path = os.path.join(self.base_dir_input, "gt_rgb.mp4")

        input_video_frames = self.read_video_frames(input_video_path)
        target_video_frames = self.read_video_frames(target_video_path)

        num_frames = min(len(input_video_frames), len(target_video_frames))
        first_frame = input_video_frames[0]
        W_orig, H_orig = first_frame.size


        if self.random_sub_sample:
            if num_frames < self.num_samples:
                # Return all available indices + repeat the last index until length == num_samples
                sampled_indices = list(range(num_frames)) + [num_frames - 1] * (self.num_samples - num_frames)
            else:
                start_idx = random.randint(0, num_frames - self.num_samples)
                sampled_indices = list(range(start_idx, start_idx + self.num_samples))
        else:
            sampled_indices = list(range(num_frames))

        input_images = []
        target_images = []
        metadata = {}
        metadata["frame_indices"] = sampled_indices
        metadata["H_orig"] = H_orig
        metadata["W_orig"] = W_orig

        # Process the selected frames.
        for idx in sampled_indices:
            # Process input frame.
            input_img = input_video_frames[idx]
            input_img = self.default_transform(input_img)
            input_images.append(input_img)

            # Process target frame.
            target_img = target_video_frames[idx]
            target_img = self.default_transform(target_img)
            target_images.append(target_img)

        input_images = torch.stack(input_images, dim=0)  # Shape: (num_samples, C, H, W)
        input_images = input_images * 2 - 1

        target_images = torch.stack(target_images, dim=0)  # Shape: (num_samples, C, H, W)
        target_images = target_images * 2 - 1

        return {
            "input_images": input_images,
            "target_images": target_images,
            "metadata": metadata,
        }

    def read_video_frames(self, video_path):
        """Reads all frames from a video file using OpenCV and converts them to PIL Images."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        cap.release()
        return frames


    @staticmethod
    def load_image(image_path):
        """Load an image from the given path."""
        return Image.open(image_path).convert("RGB")





