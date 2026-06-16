import os

import torch
import torch.multiprocessing as mp

from mapping.scene.scaffold_model import GaussianModel
from mapping import MoonSplatBackEnd
from utils import Camera, update_viewpoints_from_poses, create_slam_video
from tracking.SLAM_wrapper import SLAMWrapper
from dataloader import load_dataset
from tracking.config import load_config, config
from munch import munchify
from mapping.mapping_utils import focal2fov, getProjectionMatrix2
from tqdm import tqdm
from tracking.tracking import MoonSplatTracking

import shutil
import warnings
warnings.filterwarnings("ignore")

class MoonSplat:
    def __init__(self, dataset):
        data_config = config["data_configs"]
        method_config = config["method_configs"]

        mp.set_start_method("spawn")
        torch.backends.cuda.matmul.allow_tf32 = True

        self.dataset = dataset
        h, w = self.dataset.get_img_shape()[0]

        
        loop_closure = method_config['loop_closure']
        
        focal = data_config['frames_focal_normalized'] * w
        fixed_calib_params = {
            "width": w,
            "height": h,
            "calibration": [focal, focal, w * 0.5, h * 0.5]
        }
        
        with torch.no_grad():
            slam_wrapper = SLAMWrapper(h, w, fixed_calib_params=fixed_calib_params, apply_MAST3r_backend=loop_closure)
            slam_K = slam_wrapper.K

        self.save_dir = data_config["results_path"]
        model_params = munchify({
            "sh_degree": 0,
            "source_path": "",
            "model_path": "",
            "resolution": -1,
            "white_background": False,
            "data_device": "cuda",
        })
        opt_params = munchify({
            "iterations": 30000,
            "position_lr_init": 0.0016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "position_lr_max_steps": 30000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.001,
            "rotation_lr": 0.001,
            "percent_dense": 0.01,
            "lambda_dssim": 0.2,
            "densification_interval": 100,
            "opacity_reset_interval": 3000,
            "densify_from_iter": 500,
            "densify_until_iter": 15000,
            "densify_grad_threshold": 0.0002,
        })
        pipeline_params = munchify({
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
        })
        self.model_params, self.opt_params, self.pipeline_params = model_params, opt_params, pipeline_params

        self.dataset.fx = slam_K[0,0]
        self.dataset.fy = slam_K[1,1]
        self.dataset.cx = slam_K[0,2]
        self.dataset.cy = slam_K[1,2]
        self.dataset.width = w
        self.dataset.height = h
        self.dataset.fovx = focal2fov(self.dataset.fx, self.dataset.width)
        self.dataset.fovy = focal2fov(self.dataset.fy, self.dataset.height)
        self.dataset.device = "cuda"
        intrinsics = torch.tensor([slam_K[0,0], slam_K[1,1], slam_K[0,2], slam_K[1,2]], device='cuda')

        
        self.feat_dim = 32
        self.n_offsets = 24
        self.voxel_size_lis = [0.05, 0.1, 0.25, 1, 5]
        self.distance_lis = [20.0, 40.0, 80.0, 160.0]
        self.appearance_dim = 32

        
        method_config = config["method_configs"]
        cfg_dict = {
            "Training": {
                "mapping_itr_num": method_config['mapping_itr_num'],
                "window_size": method_config['window_size'],
                "rgb_boundary_threshold": 0.01,
                "edge_threshold": 1.1,
            },
            "SLAM": {
                "color_refinement_iter": method_config['color_refinement_iter'],
                "viz": method_config['viz'],
            },
            "Hierarchical": {
                "rendering_width": 512,
                "upsampling_method": "bicubic",
                "point_ratio": 26,
            },
        }

        self.gaussians = GaussianModel(
            self.feat_dim, 
            self.n_offsets, 
            self.voxel_size_lis, 
            self.distance_lis,
            update_depth=3, 
            update_init_factor=16, 
            update_hierachy_factor=4, 
            use_feat_bank=False, 
            appearance_dim=self.appearance_dim, 
            ratio=1, 
            add_opacity_dist=False, 
            add_cov_dist=False, 
            add_color_dist=False,
            config=cfg_dict, 
            intrinsics=intrinsics
        )
        self.gaussians.init_lr(6.0)

        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.backend = MoonSplatBackEnd(self.dataset, config)

        self.device = "cuda:0"
        self.cameras = dict()
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        self.projection_matrix = projection_matrix.to(device=self.device)

        self.viz = method_config['viz']
        self.moonsplat_tracking = MoonSplatTracking(slam_wrapper, config)

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.set_hyperparams()

    def add_new_keyframe(self, key_frame_id):
        self.kf_indices.append(key_frame_id)
        viewpoint = self.cameras[key_frame_id]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > 0.01)[None]
        initial_depth = torch.tensor(viewpoint.pred_depth, device='cuda').unsqueeze(0)
        initial_depth[~valid_rgb] = 0
        return initial_depth[0].to(torch.float32)

    def initialize(self, key_frame_id, viewpoint):
        self.kf_indices = []
        self.current_window = []
        depth_map = self.add_new_keyframe(key_frame_id)
        self.request_init(key_frame_id, viewpoint, depth_map)

    def add_to_window(self, key_frame_id, window):
        window = [key_frame_id] + window
        removed_frame = None
        method_config = config["method_configs"]
        if len(window) > method_config['window_size']:
            removed_frame = window[-1]
            window.remove(removed_frame)
        return window, removed_frame

    def request_keyframe(self, key_frame_id, viewpoint, training_keyframe_id_set, depthmap):
        msg = ["keyframe", key_frame_id, viewpoint, training_keyframe_id_set, depthmap]
        self.backend.update(msg)

    def request_init(self, key_frame_id, viewpoint, depth_map):
        msg = ["init", key_frame_id, viewpoint, depth_map]
        self.backend.update(msg)

    def request_loop_update(self, key_frame_id, poses_old, poses_update, scales_old, scales_update, pred_depth_update, kf_indices):
        msg = ["loop_update", key_frame_id, poses_old, poses_update, scales_old, scales_update, pred_depth_update, kf_indices]
        self.backend.update(msg)

    def request_color_refinement(self):
        data = self.backend.sync_backend_msg
        if data is not None:
            self.gaussians = data[1]
        self.backend.update(["color_refinement"])
        data = self.backend.sync_backend_msg
        if data is not None:
            self.gaussians = data[1]

    def update(self, timestamp, img):
        data = self.backend.sync_backend_msg
        if data is not None:
            self.gaussians = data[1]

        viewpoint = Camera(
            timestamp,
            img,
            None,
            self.projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

        add_new_kf, pred_depth, loop_flag = self.moonsplat_tracking.update(viewpoint.mast3r_image, self.moonsplat_tracking.key_frame_id)

        if not add_new_kf:
            return

        R, T = self.moonsplat_tracking.get_pose()
        viewpoint.update_RT(R, T)
        viewpoint.pred_depth = pred_depth.detach()
        
        cfg_dict_for_grad = {
            "Training": {
                "edge_threshold": 1.1,
            },
        }
        viewpoint.compute_grad_mask(cfg_dict_for_grad)

        method_config = config["method_configs"]
        if loop_flag and method_config['mapping_itr_num'] > 0:
            poses_old, poses_update, scales_old, scales_update, pred_depth_update = self.moonsplat_tracking.get_loop_update()
            poses_update = torch.tensor(poses_update)[1:] 
            poses_old = torch.tensor(poses_old)[1:]
            scales_update = torch.tensor(scales_update)[1:]
            scales_old = torch.tensor(scales_old)[1:]
            pred_depth_update = pred_depth_update[1:]
            
            update_viewpoints_from_poses(self.cameras, poses_update, pred_depth_update)
            self.request_loop_update(self.moonsplat_tracking.key_frame_id, poses_old, poses_update, scales_old, scales_update, pred_depth_update, self.kf_indices)

        self.cameras[self.moonsplat_tracking.key_frame_id] = viewpoint

        method_config = config["method_configs"]
        if method_config['mapping_itr_num'] > 0:
            if self.moonsplat_tracking.key_frame_id == 0:
                self.initialize(self.moonsplat_tracking.key_frame_id, viewpoint)
                self.current_window.append(self.moonsplat_tracking.key_frame_id)
                self.moonsplat_tracking.key_frame_id += 1
                return viewpoint

            self.current_window, _ = self.add_to_window(
                self.moonsplat_tracking.key_frame_id,
                self.current_window,
            )
            training_keyframe_id_set = self.current_window

            depth_map = self.add_new_keyframe(self.moonsplat_tracking.key_frame_id)
            self.request_keyframe(self.moonsplat_tracking.key_frame_id, viewpoint, training_keyframe_id_set, depth_map)

        method_config = config["method_configs"]
        if method_config['mapping_itr_num'] > 0:
            cur_kf_idx = len(self.backend.viewpoints)
        else:
            cur_kf_idx = self.moonsplat_tracking.key_frame_id
            self.kf_indices = list(range(0, cur_kf_idx))

        self.moonsplat_tracking.key_frame_id += 1

if __name__ == "__main__":
    load_config("config.yaml")
    
    data_config = config["data_configs"]
    results_path = data_config["results_path"]
    
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f'{results_path}/TPV', exist_ok=True)
    os.makedirs(f'{results_path}/BEV', exist_ok=True)

    dataset = load_dataset(data_config["frames_path"])
    slam = MoonSplat(dataset)

    for i in tqdm(range(len(dataset))):
        timestamp, img = dataset[i]
        slam.update(timestamp, img)

    slam.moonsplat_tracking.slam_wrapper.shutdown()

    method_config = config["method_configs"]
    if method_config['color_refinement_iter'] > 0:
        slam.request_color_refinement()

    if slam.viz:
        create_slam_video(results_path)

    slam.moonsplat_tracking.slam_wrapper.save_results(f"{results_path}/slam_ply")
    slam.gaussians.save_ply(f"{results_path}/scaffold_model.ply")
