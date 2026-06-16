import copy
import os
import re
import glob
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from mapping.mapping_utils import getProjectionMatrix2, getWorld2View2

def image_gradient(image):
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]

def image_gradient_mask(image, eps=0.01):
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)

def update_viewpoints_from_poses(viewpoints: dict, poses_update: torch.Tensor, pred_depth_update):
    assert len(viewpoints) == poses_update.shape[0], "Mismatched number of poses and viewpoints!"
    sorted_keys = sorted(viewpoints.keys())
    
    w2c_all = torch.linalg.inv(poses_update)
    R_all = w2c_all[:, :3, :3]
    T_all = w2c_all[:, :3, 3]

    for i, key in enumerate(sorted_keys):
        camera = viewpoints[key]
        camera.update_RT(R_all[i], T_all[i])
        camera.pred_depth = pred_depth_update[i]

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        gt_pose,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.device = device

        init_pose = torch.eye(4, device=device)
        self.R = init_pose[:3, :3]
        self.T = init_pose[:3, 3]
        if gt_pose is not None:
            self.R_gt = gt_pose[:3, :3]
            self.T_gt = gt_pose[:3, 3]
        else:
            self.R_gt = None
            self.T_gt = None          

        self.mast3r_image = color
        self.original_image = torch.from_numpy(self.mast3r_image["unnormalized_img"]).permute(2,0,1).cuda() / 255.0
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config.get("Training", {}).get("edge_threshold", 1.1)

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        median_img_grad_intensity = img_grad_intensity.median()
        self.grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None
        self.cam_rot_delta = None
        self.cam_trans_delta = None
        self.exposure_a = None
        self.exposure_b = None

def create_slam_video(results_path):
    fpv_pattern = os.path.join(results_path, "curkf-*-FPV.png")
    fpv_files = glob.glob(fpv_pattern)
    
    def extract_index(filepath):
        match = re.search(r'curkf-(\d+)', os.path.basename(filepath))
        return int(match.group(1)) if match else 0
    
    fpv_files.sort(key=extract_index)
    
    output_video_path = os.path.join(results_path, "slam_result_video.mp4")
    
    cols = 2
    spacing = 30
    margin_top = 100
    margin_bottom = 100
    label_img_gap = 20

    FONT_PATH = "/usr/share/fonts/truetype/CrimsonText/CrimsonText-Regular.ttf"
    LABEL_FONT_SIZE = 25
    FRAME_FONT_SIZE = 30

    LABEL_COLORS = {
        "fpv": (255, 255, 255),
        "tpv": (255, 255, 0),
        "frame": (255, 255, 255),
    }

    if not fpv_files:
        return None
    
    first_fpv_img = cv2.imread(fpv_files[0])
    if first_fpv_img is None:
        return None
    fpv_h, fpv_w = first_fpv_img.shape[:2]
    
    total_width = cols * fpv_w + (cols - 1) * spacing
    canvas_height = margin_top + fpv_h + margin_bottom
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (total_width, canvas_height))
    if not video_writer.isOpened():
        return None
    
    try:
        label_font = ImageFont.truetype(FONT_PATH, LABEL_FONT_SIZE)
        frame_font = ImageFont.truetype(FONT_PATH, FRAME_FONT_SIZE)
    except OSError:
        try:
            fallback_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            label_font = ImageFont.truetype(fallback_font, LABEL_FONT_SIZE)
            frame_font = ImageFont.truetype(fallback_font, FRAME_FONT_SIZE)
        except OSError:
            label_font = ImageFont.load_default()
            frame_font = ImageFont.load_default()
    
    written_frames = 0
    for fpv_file in fpv_files:
        index = extract_index(fpv_file)

        fpv_img = cv2.imread(fpv_file)
        if fpv_img is None:
            continue
        
        tpv_pattern = os.path.join(results_path, "TPV", f"curkf-{index}-trainkf-*-TPV.png")
        tpv_files = glob.glob(tpv_pattern)
        if tpv_files:
            tpv_img = cv2.imread(tpv_files[0])
        else:
            tpv_img = np.zeros((fpv_h, fpv_w, 3), dtype=np.uint8)

        fpv_img_resized = cv2.resize(fpv_img, (fpv_w, fpv_h))
        tpv_img_resized = cv2.resize(tpv_img, (fpv_w, fpv_h))

        canvas = np.zeros((canvas_height, total_width, 3), dtype=np.uint8)

        current_y = margin_top
        canvas[current_y:current_y + fpv_h, 0:fpv_w] = fpv_img_resized
        canvas[current_y:current_y + fpv_h, fpv_w + spacing:2 * fpv_w + spacing] = tpv_img_resized

        canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas_pil)
        labels = [
            ("Top: Input Image | Predicted Depth, Bottom: Render Image | Render Depth", 0, LABEL_COLORS["fpv"]),
            ("Third Person View Render", fpv_w + spacing, LABEL_COLORS["tpv"]),
        ]
        for label, x_start, color in labels:
            text_bbox = draw.textbbox((0, 0), label, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x_start + (fpv_w - text_width) // 2
            text_y = current_y - text_height - label_img_gap
            draw.text((text_x, text_y), label, font=label_font, fill=color)

        text = f"KeyFrame: {index}"
        text_bbox = draw.textbbox((0, 0), text, font=frame_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (total_width - text_width) // 2
        text_y = canvas_height - margin_bottom + (margin_bottom - text_height) // 2
        draw.text((text_x, text_y), text, font=frame_font, fill=LABEL_COLORS["frame"])

        canvas = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

        video_writer.write(canvas)
        written_frames += 1

        output_path = os.path.join(results_path, f"concat_frame_{index:04d}.png")
        cv2.imwrite(output_path, canvas)

    video_writer.release()

    if written_frames == 0:
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        return None

    return output_video_path
