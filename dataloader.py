import glob
import cv2
import numpy as np
import torch
import os

from tracking.mast3r_utils import resize_img

DATASET_IMG_SIZE = 512
DATASET_CENTER_PRINCIPLE_POINT = False
USE_CALIB = True

class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, dtype=np.float32):
        self.dtype = dtype
        self.rgb_files = []
        self.timestamps = []
        self.img_size = DATASET_IMG_SIZE
        self.camera_intrinsics = None
        self.use_calibration = USE_CALIB
        self.save_results = True
        
        self.input_folder = dataset_path

        # Scannetv2 Frames pattern, change for other datasets
        self.rgb_files = sorted(glob.glob(f"{self.input_folder}/frame-*.color.jpg"))
        
        self.timestamps = np.arange(0, len(self.rgb_files))

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        timestamp = self.timestamps[idx]
        return timestamp, img

    def read_img(self, idx):
        img = cv2.imread(str(self.rgb_files[idx]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self, idx):
        img = self.read_img(idx)
        img = img.astype(self.dtype) / 255.0
        img = resize_img(img, self.img_size)
        return img

    def get_img_shape(self):
        img = self.read_img(0)
        raw_img_shape = img.shape
        img, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(img, self.img_size, return_transformation=True)
        return img["img"][0].shape[1:], raw_img_shape[:2]

    def has_calib(self):
        return self.camera_intrinsics is not None

def load_dataset(dataset_path):
    return MonocularDataset(dataset_path)

class Intrinsics:
    def __init__(self, img_size, W, H, K_orig, K, distortion, mapx, mapy):
        self.img_size = img_size
        self.W, self.H = W, H
        self.K_orig = K_orig
        self.K = K
        self.distortion = distortion
        self.mapx = mapx
        self.mapy = mapy
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            np.zeros((H, W, 3)), self.img_size, return_transformation=True
        )
        self.K_frame = self.K.copy()
        self.K_frame[0, 0] = self.K[0, 0] / scale_w
        self.K_frame[1, 1] = self.K[1, 1] / scale_h
        self.K_frame[0, 2] = self.K[0, 2] / scale_w - half_crop_w
        self.K_frame[1, 2] = self.K[1, 2] / scale_h - half_crop_h

    @staticmethod
    def from_calib(img_size, W, H, calib, always_undistort=False):
        fx, fy, cx, cy = calib[:4]
        distortion = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_opt = K.copy()
        mapx, mapy = None, None
        center = DATASET_CENTER_PRINCIPLE_POINT
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (W, H), 0, (W, H), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, K_opt, (W, H), cv2.CV_32FC1
        )

        return Intrinsics(img_size, W, H, K, K_opt, distortion, mapx, mapy)
