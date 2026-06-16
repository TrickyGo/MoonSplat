import numpy as np
import torch
from tracking.lietorch_utils import as_SE3
from tracking.geometry import constrain_points_to_ray

USE_CALIB = True

class MoonSplatTracking:
    def __init__(self, slam_wrapper, config):
        self.w2c = np.zeros((4, 4))
        self.key_frame_id = 0

        
        N_MAX = 5000
        self.pose_history = np.repeat(np.eye(4)[np.newaxis, :, :], N_MAX, axis=0)
        self.pose_history_old = np.repeat(np.eye(4)[np.newaxis, :, :], N_MAX, axis=0)
        self.scale_history = np.repeat(np.ones(1), N_MAX, axis=0)
        self.scale_history_old = np.repeat(np.ones(1), N_MAX, axis=0)
        self.pred_depth_history = []

        self.slam_wrapper = slam_wrapper
        self.config = config

    def get_pose(self):
        return torch.from_numpy(self.w2c[:3, :3]), torch.from_numpy(self.w2c[:3, 3:]).squeeze()
    
    def get_loop_update(self):
        return (self.pose_history_old[:self.key_frame_id+1, :, :], 
                self.pose_history[:self.key_frame_id+1, :, :], 
                self.scale_history_old[:self.key_frame_id+1], 
                self.scale_history[:self.key_frame_id+1], 
                self.pred_depth_history[:self.key_frame_id+1])

    def update(self, frame, frame_id):
        with torch.no_grad():
            kf_count, T_WC_SIM3, status, add_new_kf = self.slam_wrapper.update(frame)
        
        c2w = as_SE3(T_WC_SIM3).matrix().squeeze(0).cpu().detach().clone().numpy()
        self.w2c = np.linalg.inv(c2w)

        cur_kf_idx = len(self.slam_wrapper.keyframes) - 1 
        pred_depth = None
        loop_flag = False

        if add_new_kf:
            cur_kf = self.slam_wrapper.keyframes[cur_kf_idx]
            rgb = cur_kf.uimg.permute(2, 0, 1).float().cuda().detach()
            _, h, w = rgb.shape

            X_canon = constrain_points_to_ray(
                cur_kf.img_shape.flatten()[:2], cur_kf.X_canon[None], cur_kf.K
            )
            cur_kf.X_canon = X_canon.squeeze(0)

            scale_factor = cur_kf.T_WC.data[0, -1]
            X_canon_scaled = cur_kf.X_canon * scale_factor
            pred_depth = X_canon_scaled[..., 2].reshape(h, w).cuda().detach()

            T_SE3_c2w = as_SE3(cur_kf.T_WC).matrix().squeeze(0).cpu().detach().clone().numpy()
            self.pose_history[cur_kf_idx] = T_SE3_c2w
            self.scale_history[cur_kf_idx] = scale_factor.cpu().detach().numpy()

        method_config = self.config['method_configs']
        loop_closure_stride = method_config['loop_closure_stride']
        if (add_new_kf and 
            self.key_frame_id > loop_closure_stride and 
            (self.key_frame_id % loop_closure_stride == 0)):
            
            self.pose_history_old = self.pose_history.copy()
            self.scale_history_old = self.scale_history.copy()
            self.pred_depth_history = []

            for kf_idx in range(len(self.slam_wrapper.keyframes)):
                cur_kf = self.slam_wrapper.keyframes[kf_idx]
                T_SE3_c2w = as_SE3(cur_kf.T_WC).matrix().squeeze(0).cpu().detach().clone().numpy()

                X_canon = constrain_points_to_ray(
                    cur_kf.img_shape.flatten()[:2], cur_kf.X_canon[None], cur_kf.K
                )
                cur_kf.X_canon = X_canon.squeeze(0)

                scale_factor = cur_kf.T_WC.data[0, -1]
                X_canon_scaled = cur_kf.X_canon * scale_factor
                self.pred_depth_history.append(X_canon_scaled[..., 2].reshape(h, w).cuda().detach())
                self.pose_history[kf_idx, :, :] = T_SE3_c2w
                self.scale_history[kf_idx] = scale_factor.cpu().detach().numpy()

            loop_flag = True

        return add_new_kf, pred_depth, loop_flag
    
