import os
import copy
import argparse
from datetime import datetime

import numpy as np
import torch
import cv2
import glob
import imageio
import torchvision
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from scipy.interpolate import interp1d, UnivariateSpline
from torchvision.transforms import CenterCrop, Compose, Resize
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from tqdm import tqdm

np.random.seed(123)


# Helper functions from pvd_utils
def save_video(data, images_path, my_fps=30, folder=None):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [np.array(Image.open(os.path.join(folder_name, path))) for folder_name, path in zip(folder, data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=my_fps, video_codec='h264', options={'crf': '10'})


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def visualizer_frame(camera_poses, highlight_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    z_values = [pose[:3, 3][2] for pose in camera_poses]
    z_min, z_max = min(z_values), max(z_values)
    
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["#00008B", "#ADD8E6"])
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    for i, pose in enumerate(camera_poses):
        camera_positions = pose[:3, 3]
        size = 100 if i == highlight_index else 25
        color = sm.to_rgba(camera_positions[2])
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
            s=size,
        )
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(90 + 30, -90)
    
    plt.ylim(-0.1, 0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close()
    return img


def setup_renderer(cameras, image_size, radius=0.01):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=10,
        bin_size=0
    )
    
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    
    render_setup = {'cameras': cameras, 'raster_settings': raster_settings, 'renderer': renderer}
    return render_setup


def my_world_point_to_kth(poses, ref_poses, points, k, device):
    kth_pose = ref_poses[k]
    inv_kth_pose = torch.inverse(kth_pose)
    new_poses = torch.bmm(inv_kth_pose.unsqueeze(0).expand_as(poses), poses)
    
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homogeneous_points = torch.cat([points, torch.ones(N, W * H, 1).to(device)], dim=-1)
    new_points = inv_kth_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1) @ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[..., :3].view(N, W, H, _)
    
    return new_poses, new_points


def my_world_point_to_obj(poses, ref_poses, points, k, r, elevation, device):
    poses, points = my_world_point_to_kth(poses, ref_poses, points, k, device)
    
    elevation_rad = torch.deg2rad(torch.tensor(180 - elevation)).to(device)
    sin_value_x = torch.sin(elevation_rad)
    cos_value_x = torch.cos(elevation_rad)
    R = torch.tensor([[1, 0, 0],
                      [0, cos_value_x, sin_value_x],
                      [0, -sin_value_x, cos_value_x]]).to(device)
    
    t = torch.tensor([0, 0, r]).to(device)
    pose_obj = torch.eye(4).to(device)
    pose_obj[:3, :3] = R
    pose_obj[:3, 3] = t
    
    inv_obj_pose = torch.inverse(pose_obj)
    new_poses = torch.bmm(inv_obj_pose.unsqueeze(0).expand_as(poses), poses)
    
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homogeneous_points = torch.cat([points, torch.ones(N, W * H, 1).to(device)], dim=-1)
    new_points = inv_obj_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1) @ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[..., :3].view(N, W, H, _)
    
    return new_poses, new_points


def my_preset_traj(c2ws_anchor, H, W, fs, c, device, viz_traj=False, save_dir=None):

    c2ws = c2ws_anchor
    
    if viz_traj:
        poses = c2ws.cpu().numpy()
        frames = [visualizer_frame(poses, i) for i in range(len(poses))]
        save_video(np.array(frames) / 255., os.path.join(save_dir, 'viz_traj.mp4'))
    
    num_views = c2ws.shape[0]
    R, T = c2ws[:, :3, :3], c2ws[:, :3, 3:]
    
    # Convert from RDF to LUF coordinate system
    R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)
    new_c2w = torch.cat([R, T], 2)
    
    w2c = torch.linalg.inv(
        torch.cat((new_c2w, torch.Tensor([[[0, 0, 0, 1]]]).to(device).repeat(new_c2w.shape[0], 1, 1)), 1))
    
    R_new, T_new = w2c[:, :3, :3].permute(0, 2, 1), w2c[:, :3, 3]
    image_size = ((H, W),)
    
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new,
                                 T=T_new, device=device)
    
    return cameras, num_views



def load_correspondence(base_path, scene, camera_id):
   
    corr_file = os.path.join(base_path, f"dycheck_{scene}_source_target_correspondence.txt")
    
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


class PCD_Render:
    def __init__(self, opts, gradio=False):
        self.opts = opts
        self.device = opts.device


    def render_pcd(self, raw_pts3d, raw_imgs, masks, views, renderer, cameras, device, motion_prob, nbv=False):

        masks = None

        static_masks = (motion_prob > 0.98).astype(np.bool)

        raw_pts3d = raw_pts3d.detach().cpu().numpy()

        static_pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(raw_pts3d, static_masks)], axis=0)).to(device)
        static_col = torch.from_numpy(np.concatenate([img[m] for img, m in zip(raw_imgs, static_masks)], axis=0)).to(device)


        N = static_pts.shape[0]
        k = max(1, int(N * 0.01))            # at least 1 sample
        idx = torch.randperm(N, device=device)[:k]

        static_pts = static_pts[idx]
        static_col = static_col[idx]

        res_imgs = []

        if masks is None:
            masks = [None] * len(raw_pts3d)

        for i, (p, img, mask, static_mask) in enumerate(
                        tqdm(zip(raw_pts3d, raw_imgs, masks, static_masks),
                            total=len(raw_pts3d)),
                        start=0):

            if mask is None:
                pts = torch.from_numpy(p).view(-1, 3).to(device)
                col = torch.from_numpy(img).view(-1, 3).to(device)
            else:
                print("Warning: mask is not None")
                return

            pts = torch.cat((pts, static_pts), dim=0)
            col = torch.cat((col, static_col), dim=0)


            current_points = pts
            current_features = col

            # Get the corresponding cameras for this single point cloud.
            current_camera = cameras[i]

            if current_points.shape[0] == 0:
                # Add a dummy point far away
                current_points = torch.tensor([[1e6, 1e6, 1e6]], dtype=torch.float32).to(current_features.device)
                current_features = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).to(current_features.device)

            # Create the point cloud (wrap in a list to make it a batch of 1).
            point_cloud = Pointclouds(points=[current_points], features=[current_features])

            # Render the full point cloud with the first renderer.
            image = renderer(point_cloud, cameras=current_camera)
            res_imgs.append(image)

        res_imgs = torch.vstack(res_imgs)
        return res_imgs
    

    def run_render(self, pcd, imgs, masks, H, W, camera_traj, camera_traj2, num_views, motion_prob, nbv=False):

        render_setup = setup_renderer(camera_traj, image_size=(H, W))
        renderer = render_setup['renderer']
        cameras = render_setup['cameras']

        render_results = self.render_pcd(pcd, imgs, masks, num_views, renderer,
                                                                           cameras, self.device, motion_prob,
                                                                           nbv=False)

        render_meta_data = {
            'render_results': render_results,
        }
        
        return render_meta_data

    def eval_render_tto_cam(self, cam_folder_path, cam, gradio=False):        

        _, test_indices = load_correspondence(
            base_path=cam_folder_path,
            scene=self.opts.seq,
            camera_id=cam
        )


        print("test_indices:", test_indices)


        ref_rgbs = []

        ref_video_file = os.path.join(cam_folder_path, f"dycheck_{self.opts.seq}_gt_cam{cam}.mp4")
        cap = cv2.VideoCapture(ref_video_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            ref_rgbs.append(frame)
        cap.release()

        ref_rgbs = np.array(ref_rgbs)
        ref_rgbs = torch.tensor(ref_rgbs, dtype=torch.float32).to(self.device)

        print("ref_rgbs:", ref_rgbs.shape, ref_rgbs.max(), ref_rgbs.min())

        gt_depth_npy_file = os.path.join(cam_folder_path, "gt_depth.npy")
        gt_depths = np.load(gt_depth_npy_file)

        
        pred_depth_npy_file = os.path.join(cam_folder_path, f"megasam_depth_aligned.npy")
        pred_pts3d_npy_file = os.path.join(cam_folder_path, f"megasam_pts3d_aligned.npy")
       
        # motion_prob_npy_file = os.path.join(cam_folder_path, f"motion_prob.npy")
        motion_prob_npy_file = os.path.join(cam_folder_path, f"static_masks_som.npy")
        
        pred_depths = np.load(pred_depth_npy_file)
        pred_pts3d = np.load(pred_pts3d_npy_file)
        motion_prob = np.load(motion_prob_npy_file)

        print("motion_prob:", motion_prob.shape, motion_prob.max(), motion_prob.min())


        motion_prob_list = list(motion_prob)
        for i in range(len(motion_prob)):
            motion_prob_list[i] = cv2.resize(motion_prob[i], 
                                    (360, 480),
                                    interpolation=cv2.INTER_LINEAR
                                    )
        motion_prob = np.array(motion_prob_list)

        pred_pcd = [torch.tensor(frame, dtype=torch.float32).to(self.device) for frame in pred_pts3d]

        print("motion_prob:", motion_prob.shape)
        
        masks = None

        imgs = []
        img_mp4_file = os.path.join(cam_folder_path, f"dycheck_{self.opts.seq}_gt_source.mp4")
        cap = cv2.VideoCapture(img_mp4_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            imgs.append(frame)
        cap.release()
        imgs = np.array(imgs)

        shape = [[imgs[0].shape[0], imgs[0].shape[1]]]
        im_shape = imgs[0].shape
        H, W = int(shape[0][0]), int(shape[0][1])


        intrinsics = np.load(os.path.join(cam_folder_path, "gt_intrinsics.npy"))
        ori_c2ws = np.load(os.path.join(cam_folder_path, "gt_c2ws.npy"))
        c2ws = np.load(os.path.join(cam_folder_path, f"gt_c2ws_cam{cam}.npy"))
        c2ws = np.repeat(c2ws[:1], len(ori_c2ws), axis=0)

        intrinsics = torch.tensor(intrinsics, dtype=torch.float32).to(self.device)
        ori_c2ws = torch.tensor(ori_c2ws, dtype=torch.float32).to(self.device)
        c2ws = torch.tensor(c2ws, dtype=torch.float32).to(self.device)
        focals = ((intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2).mean().view(1, 1)
        principal_points = torch.tensor([[intrinsics[0, 0, 2], intrinsics[0, 1, 2]]]).to(self.device)
        print("principal_points:", principal_points)
        print("focals:", focals)

        depth_avg = pred_depths[0][H // 2, W // 2] 
        radius = depth_avg * self.opts.center_scale

        pcd = pred_pcd
        ori_pcd = copy.deepcopy(pred_pcd)

        print("transforming coords...")

        c2ws, pcd = my_world_point_to_obj(poses=c2ws, ref_poses=ori_c2ws, points=torch.stack(pcd), k=0, r=radius,
                                          elevation=self.opts.elevation, device=self.device)

        ori_c2ws, _ = my_world_point_to_obj(poses=ori_c2ws, ref_poses=ori_c2ws, points=torch.stack(ori_pcd), k=0,
                                            r=radius,
                                            elevation=self.opts.elevation, device=self.device)

        ori_camera_traj, num_views = my_preset_traj(ori_c2ws, H, W, focals, principal_points,
                                                    self.device, viz_traj=False,
                                                    )
        
    
        masks = ~((gt_depths < 100) & (gt_depths > 0.1))
        pcd[masks] = 0
        masks = None
        
        c2w_single = torch.nn.Parameter(c2ws[0].clone().detach(), requires_grad=True)
        optimizer = torch.optim.Adam([c2w_single], lr=0.003)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_psnr = -float('inf')
        best_c2ws = None

        tmp_pcd = pcd[test_indices]
        tmp_imgs = imgs[test_indices]
        tmp_motion_prob = motion_prob[test_indices]

        ref_rgbs = ref_rgbs[::4]
        tmp_pcd = tmp_pcd[::4]
        tmp_imgs = tmp_imgs[::4]
        tmp_motion_prob = tmp_motion_prob[::4]

        for _ in tqdm(range(30), ncols=0, desc="TTO cam..."):
            optimizer.zero_grad()

            c2ws = c2w_single.unsqueeze(0).repeat(len(ori_c2ws), 1, 1)

            camera_traj, num_views = my_preset_traj(
                c2ws, H, W, focals, principal_points,
                self.device, viz_traj=False,
            )
            
            # Render using the current c2ws parameters.
            render_meta_data = self.run_render(tmp_pcd, tmp_imgs, masks, H, W, camera_traj,
                                            ori_camera_traj, num_views, tmp_motion_prob)
            render_results = render_meta_data['render_results']
            
            # Compute the rendered mask and MSE only on rendered regions.
            rendered_mask = (render_results > 0).any(dim=-1)
            mse = ((render_results - ref_rgbs) ** 2)[rendered_mask].mean()
            
            # Compute PSNR and define the loss.
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            ori_psnr = copy.deepcopy(psnr.item())
            print("psnr:", psnr.item())

            rgb_loss = -psnr
            
            # Backpropagate.
            rgb_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update the best PSNR and best c2ws if current PSNR is higher.
            if psnr.item() > best_psnr:
                best_psnr = psnr.item()
                best_c2ws = c2ws.clone().detach()  # clone to avoid further modifications

        # Zero gradients before final render.
        optimizer.zero_grad()

        # Use the best c2ws for final render.
        final_camera_traj, num_views = my_preset_traj(
            best_c2ws, H, W, focals, principal_points,
            self.device, viz_traj=False,
        )

        final_render_meta_data = self.run_render(pcd, imgs, masks, H, W, final_camera_traj,
                                                ori_camera_traj, num_views, motion_prob)
        render_results = final_render_meta_data['render_results']

        print("Original PSNR:", ori_psnr, "; Final best PSNR:", best_psnr)

        save_dir = cam_folder_path + "/render_results"
        os.makedirs(save_dir, exist_ok=True)
        save_video(render_results, os.path.join(save_dir, f'render_cam{cam}_stack_tto_cam.mp4'))

        return None



def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--dycheck_4d_base_path', type=str, required=True)
    parser.add_argument('--eval_seq_names', type=str, nargs='+', default=["paper-windmill"], help='Names of the evaluation sequences')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to use')
    parser.add_argument('--center_scale',  type=float, default=1., help='Scale factor for the spherical radius')
    parser.add_argument('--elevation',  type=float, default=5., help='Elevation angle of the input image in degrees')

    return parser


if __name__ == "__main__":
    parser = get_parser() 
    opts = parser.parse_args()

    dycheck_4d_base_path = opts.dycheck_4d_base_path
    eval_seq_names = opts.eval_seq_names

    for seq in eval_seq_names:
        for cam in [1, 2]:
            print("preprocessing seq:", seq)

            folder_path = os.path.join(dycheck_4d_base_path, seq)
            opts.seq = seq
            cam_folder_path = folder_path
            pvd = PCD_Render(opts)

            pvd.eval_render_tto_cam(cam_folder_path=cam_folder_path, cam=cam)

