import copy
import os
import cv2
import numpy as np
from diffusers.utils import export_to_gif
import json
from safetensors import safe_open


from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)

from pipeline.cognvs_pipeline import CogNVSPipeline

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register


class CogVideoXV2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()

        model_path = str(self.args.model_path)
        transformer_id = str(self.args.transformer_id)

        components.pipeline_cls = CogNVSPipeline
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        components.transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_id)
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        return components

    @override
    def initialize_pipeline(self) -> CogNVSPipeline:
        pipe = CogNVSPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        vae.to(self.accelerator.device)
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        self.components.text_encoder.to(self.accelerator.device)
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    # @override
    # def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     ret = {"encoded_videos": [], "prompt_embedding": [], "images": []}
    #
    #     for sample in samples:
    #         encoded_video = sample["encoded_video"]
    #         prompt_embedding = sample["prompt_embedding"]
    #         image = sample["image"]
    #
    #         ret["encoded_videos"].append(encoded_video)
    #         ret["prompt_embedding"].append(prompt_embedding)
    #         ret["images"].append(image)
    #
    #     ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
    #     ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
    #     ret["images"] = torch.stack(ret["images"])
    #
    #     return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        input_images = batch["input_images"]
        target_images = batch["target_images"]


        batch_size = input_images.shape[0]
        prompt = [""] * batch_size
        prompt_embedding = self.encode_text(prompt)

        target_images = target_images.permute(0, 2, 1, 3, 4).contiguous()
        latent = self.encode_video(target_images)


        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add noise to input_images
        input_images = input_images.permute(0, 2, 1, 3, 4).contiguous()
        input_video_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        input_video_noise_sigma = torch.exp(input_video_noise_sigma).to(dtype=input_images.dtype)
        noisy_input_images = input_images + torch.randn_like(input_images) * input_video_noise_sigma[:, None, None, None, None]

        input_images_latent_dist = self.components.vae.encode(noisy_input_images.to(dtype=self.components.vae.dtype)).latent_dist
        input_images_latents = input_images_latent_dist.sample() * self.components.vae.config.scaling_factor

        # print("input_images_latents:", input_images_latents.shape, input_images_latents.max(), input_images_latents.min())
        # print("***********************************")

        if patch_size_t is not None:
            ncopy = input_images_latents.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = input_images_latents[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            input_images_latents = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), input_images_latents], dim=2)
            assert input_images_latents.shape[2] % patch_size_t == 0


        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = input_images_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        # padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        # latent_padding = image_latents.new_zeros(padding_shape)
        # image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)


        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        # print("timesteps:", timesteps, "; loss:", loss)

        # latent_raw_dtype = latent.dtype
        # latent_pred_raw_dtype = latent_pred.dtype
        #
        # latent_pred_bf16 = latent_pred.to(dtype=torch.bfloat16)[:,1:,...]
        # latent_bf16 = latent.to(dtype=torch.bfloat16)[:,1:,...]
        #
        # latent_pred_bf16 = 1 / 0.7 * latent_pred_bf16.permute(0, 2, 1, 3, 4)
        # latent_bf16 = 1 / 0.7 * latent_bf16.permute(0, 2, 1, 3, 4)
        #
        # train_verify_data = {
        #     "latent_pred_bf16": latent_pred_bf16,
        #     "latent_bf16": latent_bf16,
        # }


        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogNVSPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        # prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        input_pixels, target_pixels, step, global_step = eval_data["input_pixels"], eval_data["target_pixels"], eval_data["step"], eval_data["global_step"]

        # input_pixels = target_pixels

        # print("during validation step...")
        # print("input_pixels:", input_pixels.shape, input_pixels.dtype, input_pixels.min(), input_pixels.max())
        # print("target_pixels:", target_pixels.shape, target_pixels.dtype, target_pixels.min(), target_pixels.max())

        ori_h, ori_w = int(eval_data['ori_h']), int(eval_data['ori_w']) 

        video_frames = pipe(
            prompt="",
            # image=input_pixels[:,0,...],
            images=input_pixels,
            num_videos_per_prompt=1,
            num_inference_steps=50,  # 50
            num_frames=self.state.train_frames,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
            height=480,
            width=720
        ).frames[0]

        val_save_dir = os.path.join(
            self.args.output_dir, "validation_images")
        os.makedirs(val_save_dir, exist_ok=True)


        out_file = os.path.join(
            val_save_dir,
            f"step_{global_step}_val_img_{step}.gif",
                                )

        for i in range(self.state.train_frames):
            img = video_frames[i]
            video_frames[i] = cv2.resize(np.array(img), (ori_w, ori_h))
            # video_frames[i] = np.array(img)

        video_frames = [Image.fromarray(frame) if isinstance(
            frame, np.ndarray) else frame for frame in video_frames]
        export_to_gif(video_frames, out_file, 8)

        target_pixels = (target_pixels[0] + 1) / 2 * 255
        target_pixels = target_pixels.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        target_pixels = [cv2.resize(frame, (ori_w, ori_h)) for frame in target_pixels]
        target_pixels = [Image.fromarray(frame) for frame in target_pixels]
        export_to_gif(target_pixels,
                      os.path.join(val_save_dir, f"step_{global_step}_val_img_{step}_gt.gif"),
                      8)

        input_pixels = (input_pixels[0] + 1) / 2 * 255
        input_pixels = input_pixels.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        input_pixels = [cv2.resize(frame, (ori_w, ori_h)) for frame in input_pixels]
        input_pixels = [Image.fromarray(frame) for frame in input_pixels]
        export_to_gif(input_pixels,
                      os.path.join(val_save_dir, f"step_{global_step}_val_img_{step}_input.gif"),
                      8)

        return [("video", video_frames)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-v2v", "lora", CogVideoXV2VLoraTrainer)
