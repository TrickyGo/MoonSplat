

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from mapping.mapping_utils import (
    inverse_sigmoid, get_expon_lr_func, mkdir_p,
    BasicPointCloud, strip_symmetric, build_scaling_rotation
)
from torch import nn
import os
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from mapping.scene.embedding import Embedding
import math

from scipy.spatial import KDTree
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace

class GaussianModel:

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = self.build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.anchor_dict = {}
        for i in range(len(self.voxel_size_lis)):
            self.anchor_dict[i] = AnchorDict()

    def __init__(self,
                 feat_dim: int = 32,
                 n_offsets: int = 5,
                 voxel_size_lis: dict = None,
                 distance_lis: dict = None,
                 update_depth: int = 3,
                 update_init_factor: int = 100,
                 update_hierachy_factor: int = 4,
                 use_feat_bank: bool = False,
                 appearance_dim: int = 32,
                 ratio: int = 1,
                 add_opacity_dist: bool = False,
                 add_cov_dist: bool = False,
                 add_color_dist: bool = False,
                 intrinsics=None,
                 config=None
                 ):
        self.intrinsics = intrinsics
        self.config = config
        self.opt = GaussiansOptParams()

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size_lis = voxel_size_lis
        self.distance_lis = distance_lis
        self.max_level = len(voxel_size_lis) - 1
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.appearance_dim = appearance_dim
        self.level_dim = 1
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0).cuda()
        self._anchor_index = torch.empty(0).cuda()
        self._offset = torch.empty(0).cuda()
        self._anchor_feat = torch.empty(0).cuda()
        self._anchor_color = torch.empty(0).cuda()
        self.opacity_accum = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self._level = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.offset_gradient_accum = torch.empty(0).cuda()
        self.offset_denom = torch.empty(0).cuda()
        self.anchor_demon = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.opacity_dist_dim + self.level_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim + self.level_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim + self.level_dim + self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * self.n_offsets),
            nn.Tanh()
        ).cuda()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_anchor_index(self):
        return self._anchor_index

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_level(self):
        return self._level

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, colors=None, voxel_size=0.01):
        if isinstance(data, np.ndarray):
            np.random.shuffle(data)
            voxel_indices = np.round(data / voxel_size).astype(np.int32)
            unique_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
            voxelized_data = unique_indices * voxel_size
            if colors is not None:
                voxelized_colors = np.zeros((len(unique_indices), 3))
                for i in range(len(unique_indices)):
                    mask = inverse_indices == i
                    voxelized_colors[i] = np.mean(colors[mask], axis=0)
                return voxelized_data, voxelized_colors
            else:
                return voxelized_data
        elif isinstance(data, torch.Tensor):
            voxel_indices = torch.round(data / voxel_size).long()
            unique_indices, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
            voxelized_data = unique_indices.float() * voxel_size
            if colors is not None:
                voxelized_colors = torch.zeros((len(unique_indices), 3), device=data.device)
                for i in range(len(unique_indices)):
                    mask = inverse_indices == i
                    voxelized_colors[i] = torch.mean(colors[mask], dim=0)
                return voxelized_data, voxelized_colors
            else:
                return voxelized_data
        else:
            raise TypeError("Unsupported data type. Please provide a numpy array or a torch tensor.")

    def pcd_from_depth(self, depth, w2c, mask=None, rgb=None,
                       compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
        width, height = depth.shape[1], depth.shape[0]
        FX, FY, CX, CY = self.intrinsics

        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX) / FX
        yy = (y_grid - CY) / FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth.reshape(-1)

        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]

        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                scale_gaussian = depth_z / ((FX + FY) / 2)
                mean3_sq_dist = scale_gaussian ** 2
            else:
                raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

        if mask is not None:
            pts = pts[mask.reshape(-1)]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return pts, mean3_sq_dist
        else:
            if rgb is not None:
                rgb = rgb.permute(1, 2, 0)
                rgb = rgb.reshape(-1, 3)
                if mask is not None:
                    rgb = rgb[mask.reshape(-1)]
                return pts, rgb
            else:
                return pts

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def create_from_pcd(self, pcd, cam_pos, anchor_index: int, colors=None):
        points = pcd
        self.set_appearance(15120)

        for i in range(self.max_level + 1):
            if self.voxel_size_lis[i] <= 0:
                init_points = torch.tensor(points).float().cuda()
                init_dist = distCUDA2(init_points).float().cuda()
                median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
                self.voxel_size_lis[i] = median_dist.item()
                del init_dist
                del init_points
                torch.cuda.empty_cache()

            if colors is not None:
                fused_point_cloud, fused_colors = self.voxelize_sample(points, colors, voxel_size=self.voxel_size_lis[i])
            else:
                fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size_lis[i])
                fused_colors = torch.empty(0).cuda()

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if i == 0:
                mask = point_dist < self.distance_lis[0]
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]
            elif i == self.max_level:
                mask = point_dist >= self.distance_lis[i - 1]
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]
            else:
                mask = (self.distance_lis[i - 1] <= point_dist) & (point_dist < self.distance_lis[i])
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]

            if fused_point_cloud.shape[0] > 0:
                offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
                anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
                if fused_colors is not None:
                    anchor_colors = fused_colors.float().cuda()
                else:
                    anchor_colors = torch.ones((fused_point_cloud.shape[0], 3), device="cuda") * 0.5

                dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
                scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
                rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots[:, 0] = 1
                opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

                level_anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
                level_offset = nn.Parameter(offsets.requires_grad_(True))
                level_anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
                level_anchor_color = nn.Parameter(anchor_colors.requires_grad_(False))
                level_scaling = nn.Parameter(scales.requires_grad_(True))
                level_rotation = nn.Parameter(rots.requires_grad_(False))
                level_opacity = nn.Parameter(opacities.requires_grad_(False))
                level_level = torch.full((fused_point_cloud.shape[0],), i, device='cuda', dtype=torch.int)

                self._anchor = torch.cat([self._anchor.detach(), level_anchor.requires_grad_(True)], dim=0)
                self._offset = torch.cat([self._offset.detach(), level_offset.requires_grad_(True)], dim=0)
                self._anchor_feat = torch.cat([self._anchor_feat.detach(), level_anchor_feat.requires_grad_(True)], dim=0)
                self._anchor_color = torch.cat([self._anchor_color.detach(), level_anchor_color.requires_grad_(False)], dim=0)
                self._scaling = torch.cat([self._scaling.detach(), level_scaling.requires_grad_(True)], dim=0)
                self._rotation = torch.cat([self._rotation.detach(), level_rotation.requires_grad_(False)], dim=0)
                self._opacity = torch.cat([self._opacity.detach(), level_opacity.requires_grad_(False)], dim=0)
                self._level = torch.cat([self._level.detach(), level_level.requires_grad_(False)], dim=0)

                self._anchor = nn.Parameter(self._anchor.requires_grad_(True))
                self._offset = nn.Parameter(self._offset.requires_grad_(True))
                self._anchor_feat = nn.Parameter(self._anchor_feat.requires_grad_(True))
                self._anchor_color = nn.Parameter(self._anchor_color.requires_grad_(False))
                self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
                self._rotation = nn.Parameter(self._rotation.requires_grad_(False))
                self._opacity = nn.Parameter(self._opacity.requires_grad_(False))

                self.anchor_dict[i].set_hash_table(fused_point_cloud)

                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self.training_setup(self.opt)

    def add_pcd_level(self, point_voxelised, cam_pos, anchor_index: int, colors_voxelised=None):
        for level in range(self.max_level + 1):
            fused_point_cloud = point_voxelised[level]
            result = self.anchor_dict[level].hash_detect(fused_point_cloud)
            fused_point_cloud = fused_point_cloud[~result]

            if colors_voxelised is not None:
                fused_colors = colors_voxelised[level][~result]
            else:
                fused_colors = None

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if level == 0:
                mask = point_dist < self.distance_lis[0] * 1.05
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]
            elif level == self.max_level:
                mask = point_dist >= self.distance_lis[level - 1]
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]
            else:
                mask = (self.distance_lis[level - 1] <= point_dist * 1.05) & (point_dist < self.distance_lis[level] * 1.05)
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]

            if fused_point_cloud.shape[0] > 0:
                new_scaling = torch.ones_like(fused_point_cloud).repeat([1, 2]).float().cuda() * self.voxel_size_lis[level]
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([fused_point_cloud.shape[0], 4], device=fused_point_cloud.device).float()
                new_rotation[:, 0] = 1.0
                new_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
                new_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
                if fused_colors is not None:
                    new_colors = fused_colors.float().cuda()
                else:
                    new_colors = torch.ones((fused_point_cloud.shape[0], 3), device="cuda") * 0.5
                new_offsets = torch.zeros_like(fused_point_cloud).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": fused_point_cloud,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "anchor_color": new_colors,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._anchor_color = optimizable_tensors["anchor_color"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                add_level = torch.full((fused_point_cloud.shape[0],), level, device='cuda', dtype=torch.int)
                self._level = torch.cat([self._level.detach(), add_level.requires_grad_(False)], dim=0)

                self.anchor_dict[level].hash_table_add(fused_point_cloud)

                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

    def add_pcd(self, pcd, cam_pos, anchor_index: int, colors=None):
        points = pcd

        for i in range(self.max_level + 1):
            if colors is not None:
                fused_point_cloud, fused_colors = self.voxelize_sample(points, colors, voxel_size=self.voxel_size_lis[i])
            else:
                fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size_lis[i])
                fused_colors = None

            result = self.anchor_dict[i].hash_detect(fused_point_cloud)
            fused_point_cloud = fused_point_cloud[~result]
            if fused_colors is not None:
                fused_colors = fused_colors[~result]

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if i == 0:
                mask = point_dist < self.distance_lis[0] * 1.05
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]
            elif i == self.max_level:
                mask = point_dist >= self.distance_lis[i - 1]
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]
            else:
                mask = (self.distance_lis[i - 1] <= point_dist * 1.05) & (point_dist < self.distance_lis[i] * 1.05)
                fused_point_cloud = fused_point_cloud[mask]
                if fused_colors is not None:
                    fused_colors = fused_colors[mask]

            if fused_point_cloud.shape[0] > 0:
                new_scaling = torch.ones_like(fused_point_cloud).repeat([1, 2]).float().cuda() * self.voxel_size_lis[i]
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([fused_point_cloud.shape[0], 4], device=fused_point_cloud.device).float()
                new_rotation[:, 0] = 1.0
                new_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
                new_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
                if fused_colors is not None:
                    new_colors = fused_colors.float().cuda()
                else:
                    new_colors = torch.ones((fused_point_cloud.shape[0], 3), device="cuda") * 0.5
                new_offsets = torch.zeros_like(fused_point_cloud).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": fused_point_cloud,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "anchor_color": new_colors,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._anchor_color = optimizable_tensors["anchor_color"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                add_level = torch.full((fused_point_cloud.shape[0],), i, device='cuda', dtype=torch.int)
                self._level = torch.cat([self._level.detach(), add_level.requires_grad_(False)], dim=0)

                self.anchor_dict[i].hash_table_add(fused_point_cloud)

                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                                      dtype=torch.int32,
                                                      device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

    def extend_gaussian(self, camera, depthmap, anchor_index: int, rgb=None, init=False):
        T_w2c = torch.eye(4, device=camera.R.device)
        T_w2c[0:3, 0:3] = camera.R
        T_w2c[0:3, 3] = camera.T
        cam_center = camera.camera_center
        point_cloud, color = self.pcd_from_depth(depthmap,
                                                   T_w2c,
                                                   rgb=rgb,
                                                   mask=generate_mask(depthmap,
                                                                      point_ratio=self.config['Hierarchical']['point_ratio']))

        if init:
            self.create_from_pcd(point_cloud, cam_center, anchor_index, colors=color)
        else:
            self.add_pcd(point_cloud, cam_center, anchor_index, colors=color)

    def update_anchor_loop(self, update_point_cloud):
        updated_anchor = self._anchor.clone().detach()

        for level in range(self.max_level + 1):
            voxel_size = self.voxel_size_lis[level]
            level_mask = (self._level == level)
            level_indices = torch.nonzero(level_mask, as_tuple=False).squeeze(-1)

            if level_indices.numel() == 0:
                continue

            level_points = update_point_cloud[level_indices]
            voxelized_points = torch.round(level_points / voxel_size) * voxel_size

            if voxelized_points.shape[0] == 0:
                continue

            min_len = min(len(level_indices), len(voxelized_points))
            selected_indices = level_indices[:min_len]
            selected_voxelized = voxelized_points[:min_len]

            updated_anchor[selected_indices] = selected_voxelized
            self.anchor_dict[level].set_hash_table(selected_voxelized)

        self._anchor = nn.Parameter(updated_anchor.requires_grad_(True))
        self.replace_tensor_to_optimizer(self._anchor, 'anchor')

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._anchor_color], 'lr': 0.0, "name": "anchor_color"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._anchor_color], 'lr': 0.0, "name": "anchor_color"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._anchor_color], 'lr': 0.0, "name": "anchor_color"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.offset_lr_delay_mult,
                                                       max_steps=training_args.offset_lr_max_steps)
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                                    lr_final=training_args.mlp_featurebank_lr_final,
                                                                    lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                                    max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                               lr_final=training_args.appearance_lr_final,
                                                               lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                               max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        level_mask = (self._level == 0)

        anchor = self._anchor[level_mask].detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat[level_mask].detach().cpu().numpy()
        offset = self._offset[level_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[level_mask].detach().cpu().numpy()
        scale = self._scaling[level_mask].detach().cpu().numpy()
        rotation = self._rotation[level_mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or \
                    'conv' in group['name'] or \
                    'feat_base' in group['name'] or \
                    'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

def generate_mask(depth, point_ratio=26):
    h, w = depth.shape
    num_points = h * w
    num_true = num_points // point_ratio
    mask = torch.zeros(h, w, dtype=torch.bool)
    indices = torch.randperm(num_points)[:num_true]
    mask.view(-1)[indices] = True
    return mask

def generate_mask_grad(depth):
    h, w = depth.shape
    num_points = h * w
    num_true = num_points // 64

    depth = depth.unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(depth, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32), padding=1)
    grad_y = F.conv2d(depth, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32), padding=1)
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze()

    max_pool = F.max_pool2d(grad.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    nms_grad = grad * (grad == max_pool.squeeze())

    flat_indices = torch.argsort(nms_grad.view(-1), descending=True)
    selected_indices = flat_indices[:num_true]

    mask = torch.zeros(h, w, dtype=torch.bool)
    mask.view(-1)[selected_indices] = True
    return mask

class AnchorDict:
    MULTIPLIER = torch.tensor([1, 2654435761, 805459861])
    INCREMENT = 1
    MODULUS = 2 ** 63 - 1

    def __init__(self) -> None:
        self._hash_table = None

    def table_shape(self):
        print('shape of the hash table:', self._hash_table.shape)

    def _hash_func(self, in_tensor):
        in_tensor = in_tensor * self.MULTIPLIER.to(in_tensor.device)
        in_tensor = in_tensor.long()
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.MODULUS
        x += self.INCREMENT
        return x

    def set_hash_table(self, x: torch.tensor):
        hashed_tensor = self._hash_func(x)
        self._hash_table = torch.unique(hashed_tensor)

    def hash_detect(self, x: torch.tensor):
        if self._hash_table is None:
            return torch.zeros((x.shape[0]), device=x.device, dtype=torch.bool)
        x = self._hash_func(x)
        result = torch.isin(x, self._hash_table)
        return result

    def hash_table_add(self, x: torch.Tensor):
        if self._hash_table is None:
            self.set_hash_table(x)
            return
        x = self._hash_func(x)
        combined_tensor = torch.cat((self._hash_table, x))
        self._hash_table = torch.unique(combined_tensor)

class GaussiansOptParams:
    def __init__(self) -> None:
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
