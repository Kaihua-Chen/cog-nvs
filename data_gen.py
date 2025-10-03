import argparse
import numpy as np
import os
import glob
import cv2
import torch
import json
from skimage.transform import resize
from tqdm import tqdm
import random
random.seed(42)

from utils import Warper, generate_traj_txt



def nvs_render():
    funwarp = Warper(device=args.device)

    mp4_path = os.path.join(args.data_path, "gt_rgb.mp4")
    cap = cv2.VideoCapture(mp4_path)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frames.append(frame)
    

    cap.release()

    vid_len = len(video_frames)
    intended_shape = video_frames[0].shape[:2]

    video_frames = np.stack(video_frames, axis=0)  # (N, H, W, 3)
    video_tensor = torch.from_numpy(video_frames) 
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    video_tensor = video_tensor.float().to(args.device) / 255.0
    video_frames = video_tensor * 2.0 - 1.0 

    depths = np.load(os.path.join(args.data_path, args.depth_file))  # (N, H, W)
    megasam_shape = depths.shape[1:]
    depths = np.array([
            resize(depth, intended_shape, anti_aliasing=True, mode='reflect')
            for depth in depths
        ])
    depths = torch.from_numpy(depths).unsqueeze(1).float().to(args.device)

    radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * args.radius_scale
        )
    # radius = min(radius, 5)

    if args.intrinsics_file is not None:
        intrinsics = np.load(os.path.join(args.data_path, args.intrinsics_file))

        factor = np.sqrt((megasam_shape[0] / intended_shape[0]) * (
                megasam_shape[1] / intended_shape[1]))
        factor_x = megasam_shape[1] / intended_shape[1]
        factor_y = megasam_shape[0] / intended_shape[0]
        intrinsics[0, 0] /= factor
        intrinsics[1, 1] /= factor
        intrinsics[0, 2] /= factor_x
        intrinsics[1, 2] /= factor_y

        intrinsics = torch.from_numpy(intrinsics).float().to(args.device)
        intrinsics = intrinsics[None].repeat([depths.shape[0], 1, 1])
    else:
        focal = 500.0
        intrinsics = (
                torch.tensor([[focal, 0.0, depths.shape[-1]//2], [0.0, focal, depths.shape[-2]//2], [0.0, 0.0, 1.0]])
                .repeat(vid_len, 1, 1)
                .to(args.device)
            )
    
    if args.extrinsics_file is not None:
        extrinsics = np.load(os.path.join(args.data_path, args.extrinsics_file))
        extrinsics = torch.from_numpy(extrinsics).float().to(args.device)
    else:
        extrinsics = (
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                .repeat(vid_len, 1, 1)
                .to(args.device)
            )

    if args.mode == "train":
        
        extrinsics_tgt_list = []
        for idx in range(1, args.num_train_render+1):
            phi_sample = [random.uniform(-15.0, 15.0) for _ in range(2)]
            theta_sample = [random.uniform(-10.0, 10.0) for _ in range(2)]
            r_sample = [random.uniform(-0.1, 0.1) for _ in range(2)]

            phi_sample.sort()
            theta_sample.sort()
            r_sample.sort()

            phi = phi_sample + [random.uniform(phi_sample[0], phi_sample[1]) for _ in range(2)]
            theta = theta_sample + [random.uniform(theta_sample[0], theta_sample[1]) for _ in range(2)]
            r = r_sample + [random.uniform(r_sample[0], r_sample[1]) * radius for _ in range(2)]

            phi.sort()
            theta.sort()
            r.sort()

            extrinsics_tgt = generate_traj_txt(extrinsics, phi, theta, r, vid_len, args.device)
            extrinsics_tgt[:, 2, 3] = extrinsics_tgt[:, 2, 3] + radius
            extrinsics_tgt_list.append(extrinsics_tgt)

        if_twice = True


    elif args.mode == "eval":

        txt_files = sorted(glob.glob("trajs/*.txt"))

        extrinsics_tgt_list = []
        for idx, txt_file in enumerate(txt_files):
            with open(txt_file, 'r') as file:
                lines = file.readlines()
                phi = [float(i) for i in lines[0].split()]
                theta = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]

                extrinsics_tgt = generate_traj_txt(extrinsics, phi, theta, r, vid_len, args.device)
                extrinsics_tgt[:, 2, 3] = extrinsics_tgt[:, 2, 3] + radius
                extrinsics_tgt_list.append(extrinsics_tgt)

        if_twice = False

    else:
        raise ValueError("mode should be train or eval")

    extrinsics[:, 2, 3] = extrinsics[:,2 ,3] + radius
    
    for idx, extrinsics_tgt in enumerate(extrinsics_tgt_list):
        video_frames_tgt = []

        for i in tqdm(range(vid_len)):

            warped_frame, _, _, _ = funwarp.forward_warp(
                    video_frames[i:i+1], 
                    None,
                    depths[i:i+1], 
                    extrinsics[i:i+1], 
                    extrinsics_tgt[i:i+1],
                    intrinsics[i:i+1], 
                    None,
                    True, # whether to clean up noisy points on objects' boundaries
                    twice=if_twice,
                )
            video_frames_tgt.append(warped_frame)

        video_frames_tgt = (torch.cat(video_frames_tgt) + 1.0) / 2.0
        video_frames_tgt = video_frames_tgt.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)
        video_frames_tgt = (video_frames_tgt * 255).astype(np.uint8)


        output_mp4_name = f"{args.mode}_render{idx+1}.mp4"
        output_path = os.path.join(args.data_path, output_mp4_name)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (intended_shape[1], intended_shape[0]))
        for frame in video_frames_tgt:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Rendered video saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogNVS data_gen script")
    parser.add_argument("--device", type=str, default="cuda:0", help="which gpu to use")
    parser.add_argument("--data_path", required=True, type=str, help="Path to the input data directory")
    parser.add_argument("--mode", type=str, required=True, help="train or eval")
    parser.add_argument("--depth_file", type=str, default="cam_info/megasam_depth.npy", help="Path to the depth .npy file")
    parser.add_argument("--intrinsics_file", type=str, default=None, help="Path to the intrinsics .npy file")
    parser.add_argument("--extrinsics_file", type=str, default=None, help="Path to the extrinsics .npy file")
    parser.add_argument("--num_train_render", type=int, default=8, help="number of renderings in")
    parser.add_argument("--radius_scale", type=float, default=1.0, help="Scale factor for the spherical radius")

    args = parser.parse_args()

    args.save_dir = args.data_path
    
    nvs_render()