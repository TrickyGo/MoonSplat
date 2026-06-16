import random

import torch
from tqdm import tqdm

from mapping.gaussian_renderer import render
from mapping.mapping_utils import l1_loss, ssim
from utils import Camera, update_viewpoints_from_poses
from mapping.mapping_utils import (
    clone_obj, get_pose, get_loss_mapping, 
    update_point_cloud_in_batches_SIM3,
    anchor_in_frustum, tensor_to_image, depth_to_colormap
)
import numpy as np
import cv2

import copy

class MoonSplatBackEnd:
    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.config = config

        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.device = "cuda"
        self.dtype = torch.float32
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

        self.pose_opt = True

        self.width = float(self.dataset.width)
        self.height = float(self.dataset.height)
        self.fx = float(self.dataset.fx)
        self.fy = float(self.dataset.fy)
        self.cx = float(self.dataset.cx)
        self.cy = float(self.dataset.cy)

        self.intrinsics = torch.tensor([
                [self.fx,   0,          self.cx ],
                [0,         self.fy,    self.cy ],
                [0,         0,          1       ]
            ], device='cuda')
        
        self.trajectory = 255 + np.zeros((700, 700, 3), dtype=np.uint8)

        
        method_config = config['method_configs']
        data_config = config['data_configs']
        self.color_refinement_iter = method_config['color_refinement_iter']
        self.save_dir = data_config['results_path']

        self.apply_SIM3_gaussian_update = True

        self.max_depth = 5
        self.sync_backend_msg = None
        
        self.viz = method_config['viz']

    def set_hyperparams(self):
        
        method_config = self.config['method_configs']
        self.mapping_itr_num = method_config['mapping_itr_num']
        self.init_itr_num = self.mapping_itr_num
        self.window_size = method_config['window_size']

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, rgb = None):
        self.gaussians.extend_gaussian(
            viewpoint, anchor_index = frame_idx,  init=init, depthmap=depth_map, rgb = rgb
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                        intrinsics=self.intrinsics, 
                                        pose=get_pose(viewpoint), 
                                        cam_center=viewpoint.camera_center,
                                        distance_lis = self.gaussians.distance_lis,
                                        levels=self.gaussians.get_level, 
                                        h=self.height, 
                                        w=self.width)
            opt_mask.bitwise_or_(m)
            
            render_pkg = render(
                viewpoint, 
                self.gaussians, 
                self.pipeline_params, 
                self.background, 
                visible_mask=opt_mask
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        return render_pkg

    def map(self, training_keyframe_id_set, prune=False, iters=1, time_return = False):
        if len(training_keyframe_id_set) == 0:
            return

        for mapping_iter in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0

            tmp_training_keyframe_id_set = self.sample_historical_keyframes(training_keyframe_id_set, n_samples=4)
            if random.random() < 0.4:  
                tmp_training_keyframe_id_set += list(training_keyframe_id_set)

            tmp_viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in tmp_training_keyframe_id_set]
  
            for cam_idx in range(len(tmp_training_keyframe_id_set)):
                kf_idx = tmp_training_keyframe_id_set[cam_idx]
                viewpoint = tmp_viewpoint_stack[cam_idx]

                opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                            intrinsics=self.intrinsics, 
                                            pose=get_pose(viewpoint), 
                                            cam_center=viewpoint.camera_center,
                                            distance_lis = self.gaussians.distance_lis,
                                            levels=self.gaussians.get_level, 
                                            h=self.height, 
                                            w=self.width)
                opt_mask.bitwise_or_(m)

                
                render_pkg = render(
                    viewpoint, 
                    self.gaussians, 
                    self.pipeline_params, 
                    self.background, 
                    visible_mask=opt_mask
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )

                

                
                
                if mapping_iter == (iters - 1):
                    cur_kf_idx = training_keyframe_id_set[0]
                    with torch.no_grad():
                        viewpoint = self.viewpoints[training_keyframe_id_set[0]]

                        opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                        m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                                    intrinsics=self.intrinsics, 
                                                    pose=get_pose(viewpoint), 
                                                    cam_center=viewpoint.camera_center,
                                                    distance_lis = self.gaussians.distance_lis,
                                                    levels=self.gaussians.get_level, 
                                                    h=self.height, 
                                                    w=self.width)
                        opt_mask.bitwise_or_(m)

                        render_pkg = render(
                            viewpoint, 
                            self.gaussians, 
                            self.pipeline_params, 
                            self.background, 
                            visible_mask=opt_mask
                        )
                        (
                            image,
                            viewspace_point_tensor,
                            visibility_filter,
                            radii,
                            depth,
                            opacity,
                            n_touched,
                        ) = (
                            render_pkg["render"],
                            render_pkg["viewspace_points"],
                            render_pkg["visibility_filter"],
                            render_pkg["radii"],
                            render_pkg["depth"],
                            render_pkg["opacity"],
                            render_pkg["n_touched"],
                        )

                        gt_image = viewpoint.original_image.cuda()
                        pred_depth = viewpoint.pred_depth
                        render_img = tensor_to_image(image)
                        gt_img = tensor_to_image(gt_image)
                        depth_colored = depth_to_colormap(depth, max_depth=self.max_depth)
                        mono_depth_colored = depth_to_colormap(pred_depth, max_depth=self.max_depth)
                        
                        height, width = render_img.shape[:2]
                        images = [
                            cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR),
                            mono_depth_colored,
                            cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR),
                            depth_colored,
                        ]
                        grid = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
                        grid[:height, :width] = images[0] 
                        grid[:height, width:2*width] = images[1]  
                        grid[height:2*height, :width] = images[2]  
                        grid[height:2*height, width:2*width] = images[3] 
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        cv2.imwrite(f"{self.save_dir}/curkf-{len(self.viewpoints)}-trainkf-{cur_kf_idx}-iter-{mapping_iter}-FPV.png", grid)

                        
                        third_person_viewpoint = self.create_third_person_viewpoint(viewpoint)
                        third_person_image = self.render_novel_view(third_person_viewpoint)
                        cv2.imwrite(f"{self.save_dir}/TPV/curkf-{len(self.viewpoints)}-trainkf-{cur_kf_idx}-iter-{mapping_iter}-TPV.png", 
                                cv2.cvtColor(third_person_image, cv2.COLOR_RGB2BGR))
                        
                        
                        
                        
                        
                        
                        

                

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()

                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

    def color_refinement(self):

        iteration_total = self.color_refinement_iter * len(self.viewpoints.keys())
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )

            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                        intrinsics=self.intrinsics, 
                                        pose=get_pose(viewpoint_cam), 
                                        cam_center=viewpoint_cam.camera_center,
                                        distance_lis = self.gaussians.distance_lis,
                                        levels=self.gaussians.get_level, 
                                        h=self.height, 
                                        w=self.width)
            opt_mask.bitwise_or_(m)
            
            render_pkg = render(
                viewpoint_cam, 
                self.gaussians, 
                self.pipeline_params, 
                self.background, 
                visible_mask=opt_mask
            )
            image, visibility_filter, radii, scaling = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["scaling"],
            )

            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)

            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * ssim_loss + 0.05*scaling_reg

            loss.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

            

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
  
        if tag is None:
            tag = "sync_backend"
        
        msg = [tag, clone_obj(self.gaussians)]
        self.sync_backend_msg = msg

    def update(self, data):
        if data[0] == "color_refinement":
            self.color_refinement()
            self.push_to_frontend()

        elif data[0] == "init":
            cur_frame_idx = data[1]
            viewpoint = data[2]
            depth_map = data[3]
            self.reset()

            self.viewpoints[cur_frame_idx] = viewpoint
            self.orig_viewpoint = viewpoint
            self.add_next_kf(
                cur_frame_idx, viewpoint, depth_map=depth_map, init=True, rgb=viewpoint.original_image
            )
            self.initialize_map(cur_frame_idx, viewpoint)
            self.push_to_frontend("init")

        elif data[0] == "loop_update":
            cur_frame_idx = data[1]
            poses_old = data[2]
            poses_update = data[3]
            scales_old = data[4]
            scales_update = data[5]
            pred_depth_update = data[6]
            kf_indices = data[7]

            update_viewpoints_from_poses(self.viewpoints, torch.tensor(poses_update), pred_depth_update)

            updated_pcd = update_point_cloud_in_batches_SIM3(self.gaussians.get_anchor,
                                                        poses_old, 
                                                        poses_update, 
                                                        scales_old,
                                                        scales_update,
                                                        self.gaussians.get_anchor_index,
                                                        kf_indices)

                
            self.gaussians.update_anchor_loop(updated_pcd.cuda())

        elif data[0] == "keyframe":
            cur_frame_idx = data[1]
            viewpoint = data[2]
            training_keyframe_id_set = data[3]
            depth_map = data[4]

            self.viewpoints[cur_frame_idx] = viewpoint
            
            

            self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, rgb=viewpoint.original_image)

            iter_per_kf = self.mapping_itr_num

            self.map(training_keyframe_id_set, iters=iter_per_kf, time_return = True)

            self.push_to_frontend("keyframe")

    

    def create_third_person_viewpoint(self, viewpoint, move_dist=-0.5):
        third_person_viewpoint = copy.deepcopy(viewpoint)

        R = third_person_viewpoint.R
        T = third_person_viewpoint.T - torch.tensor([0,0,move_dist]).cuda()

        third_person_viewpoint.update_RT(R, T)
        
        return third_person_viewpoint

    def render_novel_view(self, novel_viewpoint):
        viewpoint = novel_viewpoint
        opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
        m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                    intrinsics=self.intrinsics, 
                                    pose=get_pose(viewpoint), 
                                    cam_center=viewpoint.camera_center,
                                    distance_lis = self.gaussians.distance_lis,
                                    levels=self.gaussians.get_level, 
                                    h=self.height, 
                                    w=self.width)
        opt_mask.bitwise_or_(m)
        
        render_pkg = render(
            viewpoint, 
            self.gaussians, 
            self.pipeline_params, 
            self.background, 
            visible_mask=opt_mask
        )
        
        
        image = render_pkg["render"]
        
        
        img_np = tensor_to_image(image)
        
        return img_np

    
    def sample_historical_keyframes(self, current_keyframe_id_set, n_samples=2):
        current_set = set(current_keyframe_id_set)
        all_keyframe_ids = set(self.viewpoints.keys())
        historical_keyframe_ids = all_keyframe_ids - current_set

        if len(historical_keyframe_ids) == 0:
            return []
        if len(historical_keyframe_ids) <= n_samples:
            return list(historical_keyframe_ids)

        return random.sample(list(historical_keyframe_ids), n_samples)

