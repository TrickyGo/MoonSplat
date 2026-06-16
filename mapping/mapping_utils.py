

import math
import os
import random
import sys
from datetime import datetime
from errno import EEXIST
from os import makedirs, path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class LearningRateFunction:
    def __init__(self, lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps

    def __call__(self, step):
        if step < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            return 0.0
        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    return LearningRateFunction(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps)

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()

def gaussian(window_size, sigma):
    from math import exp
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

import copy
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj

def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = 0.9
    rgb_boundary_threshold = config.get("Training", {}).get("rgb_boundary_threshold", 0.01)

    gt_image = viewpoint.original_image.cuda()
    pred_depth = viewpoint.pred_depth

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (pred_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - pred_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

def update_point_cloud_vectorized(
    pcd: torch.Tensor,
    poses_old,
    poses_update,
    index_pose: torch.Tensor,
    keyframe_indices: list[int]
) -> torch.Tensor:
    device = pcd.device

    if isinstance(poses_old, np.ndarray):
        poses_old = torch.from_numpy(poses_old)
    if isinstance(poses_update, np.ndarray):
        poses_update = torch.from_numpy(poses_update)

    poses_old = poses_old.float().to(device)
    poses_update = poses_update.float().to(device)

    kf_map = {kf_idx: i for i, kf_idx in enumerate(keyframe_indices)}
    pose_ids = torch.tensor([kf_map[int(i.item())] for i in index_pose], device=device)

    pose_old_batch = poses_old[pose_ids]
    pose_update_batch = poses_update[pose_ids]

    ones = torch.ones((pcd.shape[0], 1), device=device)
    pcd_h = torch.cat([pcd, ones], dim=1).unsqueeze(-1)

    pose_old_inv_batch = torch.linalg.inv(pose_old_batch)
    pcd_cam = torch.bmm(pose_old_inv_batch, pcd_h)
    pcd_updated_h = torch.bmm(pose_update_batch, pcd_cam)

    return pcd_updated_h[:, :3, 0]

def update_point_cloud_vectorized_SIM3(
    pcd: torch.Tensor,
    poses_old,
    poses_update,
    scales_old: torch.Tensor,
    scales_update: torch.Tensor,
    index_pose: torch.Tensor,
    keyframe_indices: list[int]
) -> torch.Tensor:
    device = pcd.device
    
    if isinstance(poses_old, np.ndarray):
        poses_old = torch.from_numpy(poses_old)
    if isinstance(poses_update, np.ndarray):
        poses_update = torch.from_numpy(poses_update)
    
    poses_old = poses_old.float().to(device)
    poses_update = poses_update.float().to(device)
    scales_old = scales_old.float().to(device)
    scales_update = scales_update.float().to(device)
    
    kf_map = {kf_idx: i for i, kf_idx in enumerate(keyframe_indices)}
    pose_ids = torch.tensor([kf_map[int(i.item())] for i in index_pose], device=device)
    
    pose_old_batch = poses_old[pose_ids]
    pose_update_batch = poses_update[pose_ids]
    scale_old_batch = scales_old[pose_ids]
    scale_update_batch = scales_update[pose_ids]
    
    ones = torch.ones((pcd.shape[0], 1), device=device)
    pcd_h = torch.cat([pcd, ones], dim=1).unsqueeze(-1)
    
    pose_old_inv_batch = torch.linalg.inv(pose_old_batch)
    pcd_cam_h = torch.bmm(pose_old_inv_batch, pcd_h)
    pcd_cam = pcd_cam_h[:, :3, 0]
    
    scale_ratio = scale_update_batch / scale_old_batch
    pcd_cam_scaled = pcd_cam * scale_ratio.unsqueeze(-1)
    
    pcd_cam_scaled_h = torch.cat([
        pcd_cam_scaled.unsqueeze(-1),
        torch.ones((pcd.shape[0], 1, 1), device=device)
    ], dim=1)
    
    pcd_updated_h = torch.bmm(pose_update_batch, pcd_cam_scaled_h)
    
    return pcd_updated_h[:, :3, 0]

def update_point_cloud_in_batches_SIM3(
    pcd: torch.Tensor,
    poses_old,
    poses_update,
    scales_old: torch.Tensor,
    scales_update: torch.Tensor,
    index_pose: torch.Tensor,
    keyframe_indices: list[int],
    batch_size: int = 500_000
) -> torch.Tensor:
    updated_pcd_list = []
    N = pcd.shape[0]
    
    for i in range(0, N, batch_size):
        pcd_batch = pcd[i:i + batch_size]
        index_batch = index_pose[i:i + batch_size]
        
        updated = update_point_cloud_vectorized_SIM3(
            pcd_batch, poses_old, poses_update,
            scales_old, scales_update,
            index_batch, keyframe_indices
        )
        updated_pcd_list.append(updated)
    
    return torch.cat(updated_pcd_list, dim=0)

@torch.no_grad()
def anchor_in_frustum(anchors, intrinsics, pose, h, w, cam_center=None, 
                      distance_lis=None, levels=None, level_index: int = None):
    anchors = anchors.float()
    intrinsics = intrinsics.float()
    pose = pose.float()
    
    if len(intrinsics.shape) == 1:
        fx, fy, cx, cy = intrinsics
        intrinsics = torch.tensor([[fx, .0, cx],
                        [.0, fy, cy],
                        [.0, .0, 1.0]], device=intrinsics.device)
    
    points_cam = torch.mm(pose[:3, :3], anchors.t()) + pose[:3, 3].unsqueeze(1)
    points_img = torch.mm(intrinsics, points_cam)
    points_img = points_img / (points_img[2, :] + 1e-6)
    zs = points_cam[2, :]

    visible_mask = torch.bitwise_and(points_img[0, :] >= 0, points_img[0, :] <= w)
    visible_mask.bitwise_and_(points_img[1, :] >= 0)
    visible_mask.bitwise_and_(points_img[1, :] <= h)
    visible_mask.bitwise_and_(zs > 0)

    mask = torch.zeros([anchors.shape[0],], dtype=torch.bool, device='cuda')

    if distance_lis is not None:
        max_level = len(distance_lis)
        point_dist = torch.norm(anchors - cam_center, dim=-1)

        for level in range(max_level + 1):
            if level_index is not None:
                level = level_index

            level_mask = torch.ones([anchors.shape[0],], dtype=torch.bool, device='cuda')

            if level == 0:
                level_mask.bitwise_and_(point_dist < distance_lis[0])
            elif level == max_level:
                level_mask.bitwise_and_(point_dist >= distance_lis[max_level - 1])
            else:
                level_mask.bitwise_and_(point_dist < distance_lis[level])
                level_mask.bitwise_and_(point_dist >= distance_lis[level - 1])

            level_mask.bitwise_and_(levels == level)
            level_mask.bitwise_and_(visible_mask)
            mask.bitwise_or_(level_mask)

            if level_index is not None:
                break
    else:
        mask.bitwise_or_(visible_mask)

    return mask

def tensor_to_image(tensor):
    img = torch.clamp(tensor, 0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

def depth_to_colormap(depth, max_depth=10):
    depth_np = depth.squeeze().cpu().numpy()
    depth_clipped = np.clip(depth_np, 0, max_depth)
    depth_norm = (depth_clipped / max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V_mat = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V_mat = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V_mat

def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_pose(camera):
    T_w2c = torch.eye(4, device='cuda')
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    if camera.cam_trans_delta is not None and camera.cam_rot_delta is not None:
        tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
        new_w2c = SE3_exp(tau) @ T_w2c
        return new_w2c
    else:
        return T_w2c
