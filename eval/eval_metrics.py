import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from tomlkit import datetime
import torch
from cleanfid import fid
import glob
from tqdm import tqdm
import argparse
import shutil
from datetime import datetime


DATASETS = {
    "kubric_4d": {
        "scenes": [f"scn029{str(i).zfill(2)}" for i in range(20)],
        "pred_methods": ["cognvs", "gcd", "trajcrafter", "gt_render"],
        "gt_key": "gt",
    },
    "pardom_4d": {
        "scenes": [
            "scene_000048", "scene_000072", "scene_000074", "scene_000106", "scene_000165",
            "scene_000167", "scene_000173", "scene_000181", "scene_000294", "scene_000298",
            "scene_000310", "scene_000315", "scene_000337", "scene_000362", "scene_000365",
            "scene_000444", "scene_000581", "scene_000626", "scene_000651", "scene_000663",
        ],
        "pred_methods": ["cognvs", "gcd", "trajcrafter", "gt_render"],
        "gt_key": "gt",
    },
    "dycheck": {
        "scenes": ["apple", "block", "paper-windmill", "spin", "teddy"],
        "pred_methods": ["cognvs", "trajcrafter", "megasam_render", "shape_of_motion", "mosca"],
        "gt_key": "gt",
    },
}


def to_lpips_tensor(img):
    return torch.tensor(img).unsqueeze(0).to(torch.float32) * 2 - 1

def read_video(path, dataset, dycheck_resolution=None):
    if dataset == "dycheck":
        if dycheck_resolution == "360p":
            default_shape = (480, 360)
        elif dycheck_resolution == "720p":
            default_shape = (960, 720)
        else:
            raise ValueError(f"Invalid resolution '{dycheck_resolution}' for dataset '{dataset}'")
    elif dataset in ["kubric_4d", "pardom_4d"]:
        default_shape = (384, 576)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame.shape[:2] != default_shape:
            frame = cv2.resize(frame, (default_shape[1], default_shape[0]))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_norm = frame_rgb.astype(np.float32) / 255.0
        frames.append(frame_norm.transpose(2, 0, 1))  # (3, H, W)
    cap.release()
    return np.stack(frames, axis=0)


def get_video_pairs(base_path, dataset, pred_method, dycheck_resolution=None):
    cfg = DATASETS[dataset]
    scenes = cfg["scenes"]

    pred_dir = os.path.join(base_path, dataset, pred_method)
    if dataset == "dycheck":
        if dycheck_resolution == "360p":
            gt_dir = os.path.join(base_path, dataset, f"target_{cfg['gt_key']}")
        elif dycheck_resolution == "720p":
            gt_dir = os.path.join(base_path, dataset, f"target_{cfg['gt_key']}_1x")
        else:
            raise ValueError(f"Invalid resolution '{dycheck_resolution}' for dataset '{dataset}'")
    else:
        gt_dir = os.path.join(base_path, dataset, f"target_{cfg['gt_key']}")


    pairs = []

    for scene in scenes:
        if dataset in ["kubric_4d", "pardom_4d"]:
            pred_path = glob.glob(os.path.join(pred_dir, f"*{scene}_{pred_method}.mp4"))
            gt_path = glob.glob(os.path.join(gt_dir, f"*{scene}_{cfg['gt_key']}.mp4"))
            if pred_path and gt_path:
                pairs.append((scene, pred_path[0], gt_path[0]))

        elif dataset == "dycheck":
            if pred_method == "megasam_render":
                pred_paths = sorted(glob.glob(os.path.join(pred_dir, f"*{scene}_{pred_method}_stack_cam*.mp4")))
            else:   
                pred_paths = sorted(glob.glob(os.path.join(pred_dir, f"*{scene}_{pred_method}_cam*.mp4")))
            gt_paths   = sorted(glob.glob(os.path.join(gt_dir,  f"*{scene}_{cfg['gt_key']}_cam*.mp4")))
            for p, g in zip(pred_paths, gt_paths):
                pairs.append((scene, p, g))

    return pairs


def load_correspondence(base_path, dataset, scene, pred_method, camera_id):
    """Load frame correspondence for dycheck dataset"""
    if pred_method not in ["cognvs", "trajcrafter", "megasam_render"]:
        return None, None
    
    corr_file = os.path.join(base_path, dataset, f"dycheck_{scene}_source_target_correspondence.txt")
    
    if not os.path.exists(corr_file):
        return None, None
    
    source_indices = []
    target_indices = []
    target_prefix = str(camera_id)
    
    with open(corr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract prefix and index from format "X_XXXXX.png"
            parts = line.replace('.png', '').split('_')
            prefix = parts[0]
            frame_idx = int(parts[1])
            
            if prefix == '0':
                source_indices.append(frame_idx)
            elif prefix == target_prefix:
                target_indices.append(frame_idx)
    
    return source_indices, target_indices


def evaluate_dataset(base_path, dataset, pred_method, dycheck_resolution=None):

    if dataset == "dycheck":
        pairs = get_video_pairs(base_path, dataset, pred_method, dycheck_resolution)
    else:
        pairs = get_video_pairs(base_path, dataset, pred_method)

    print("we are evaluating dataset: ", dataset, "using method: ", pred_method)
    print(f"Found {len(pairs)} video pairs.")

    # Prepare log file path
    log_path = os.path.join(f"eval_log.log")

    pred_dir = os.path.join(base_path, dataset, pred_method)

    temp_gt_dir = os.path.join(pred_dir, "tmp_gt")
    temp_pred_dir = os.path.join(pred_dir, "tmp_pred")
    shutil.rmtree(temp_gt_dir, ignore_errors=True)
    shutil.rmtree(temp_pred_dir, ignore_errors=True)
    os.makedirs(temp_gt_dir)
    os.makedirs(temp_pred_dir)

    lpips_fn = lpips.LPIPS(net='alex')

    psnr_list, ssim_list, lpips_list = [], [], []
    frame_index = 0 

    print("evaluating psnr, ssim, lpips ...")

    for scene, pred_path, gt_path in tqdm(pairs, ncols=0): 
        if dataset == "dycheck":

            pred_rgb = read_video(pred_path, dataset, dycheck_resolution)
            gt_rgb = read_video(gt_path, dataset, dycheck_resolution)
 
            camera_id = int(gt_path.split('_cam')[-1].split('.')[0])
            source_indices, target_indices = load_correspondence(base_path, dataset, scene, pred_method, camera_id)

            if target_indices is not None:
                pred_rgb = pred_rgb[target_indices]

        else:
            pred_rgb = read_video(pred_path, dataset)
            gt_rgb = read_video(gt_path, dataset)

        assert pred_rgb.shape[0] == gt_rgb.shape[0], f"Different frame numbers for pred and gt in scene {scene}"
        T = pred_rgb.shape[0]

        for t in range(T):
            pred_frame = pred_rgb[t]
            gt_frame = gt_rgb[t]

            ### PSNR / SSIM
            psnr = peak_signal_noise_ratio(pred_frame, gt_frame, data_range=1.0)
            ssim = structural_similarity(pred_frame, gt_frame, data_range=1.0, channel_axis=0)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # LPIPS
            gt_tensor = to_lpips_tensor(gt_frame)
            pred_tensor = to_lpips_tensor(pred_frame)

            lpips_val = lpips_fn(pred_tensor, gt_tensor).item()
            lpips_list.append(lpips_val)

            gt_png = (gt_frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            pred_png = (pred_frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(temp_gt_dir, f"{frame_index:05d}.png"), cv2.cvtColor(gt_png, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(temp_pred_dir, f"{frame_index:05d}.png"), cv2.cvtColor(pred_png, cv2.COLOR_RGB2BGR))
            frame_index += 1

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    mean_lpips = np.mean(lpips_list)

    print(f"PSNR: {mean_psnr:.5f}")
    print(f"SSIM: {mean_ssim:.5f}")
    print(f"LPIPS: {mean_lpips:.5f}")

    print("evaluating fid and kid ...")
    fid_value = fid.compute_fid(temp_gt_dir, temp_pred_dir, mode="clean")
    kid_value = fid.compute_kid(temp_gt_dir, temp_pred_dir, mode="clean")

    print(f"FID: {fid_value:.5f}")
    print(f"KID: {kid_value:.5f}")

    print("cleaning up temporary files ...")
    shutil.rmtree(temp_gt_dir)
    shutil.rmtree(temp_pred_dir)

    # Append metrics to log file
    resolution_suffix = f"_{dycheck_resolution}" if dataset == "dycheck" and dycheck_resolution else ""
    log_entry = f"Dataset: {dataset}{resolution_suffix}, Method: {pred_method}\n"
    log_entry += f"PSNR: {mean_psnr:.5f}\n"
    log_entry += f"SSIM: {mean_ssim:.5f}\n"
    log_entry += f"LPIPS: {mean_lpips:.5f}\n"
    log_entry += f"FID: {fid_value:.5f}\n"
    log_entry += f"KID: {kid_value:.5f}\n"
    log_entry += "-" * 50 + "\n"
    
    with open(log_path, 'a') as f:
        f.write(log_entry)
    
    print(f"Results appended to {log_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pred_method", type=str, default="cognvs")
    parser.add_argument("--dycheck_resolution", type=str, default="720p", choices=["360p", "720p"]) 

    args = parser.parse_args()

    if args.pred_method not in DATASETS[args.dataset]["pred_methods"]:
        raise ValueError(f"Method '{args.pred_method}' unavailable for dataset {args.dataset}")


    evaluate_dataset(
        base_path=args.base_path,
        dataset=args.dataset,
        pred_method=args.pred_method,
        dycheck_resolution=args.dycheck_resolution
    )


if __name__ == "__main__":
    main()
