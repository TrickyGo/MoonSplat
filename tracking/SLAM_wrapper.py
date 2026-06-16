import multiprocessing as mp
import pathlib
import time
import lietorch
import numpy as np
import torch
from tracking.global_opt import FactorGraph
from tracking.config import config, set_global_config
from tracking.frame import Mode, SharedKeyframes, SharedStates, create_frame
from tracking.geometry import constrain_points_to_ray
from tracking.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from tracking.tracker import FrameTracker
from dataloader import Intrinsics

from plyfile import PlyData, PlyElement

USE_CALIB = True
RETRIEVAL_K = 3
RETRIEVAL_MIN_THRESH = 5e-3
LOCAL_OPT_MIN_MATCH_FRAC = 0.1
DATASET_IMG_SIZE = 512

@torch.no_grad()
def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue

        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        
        kf_idx = []
        
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=RETRIEVAL_K,
            min_thresh=RETRIEVAL_MIN_THRESH,
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)

        kf_idx = set(kf_idx)  
        kf_idx.discard(idx)  
        kf_idx = list(kf_idx)  
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, LOCAL_OPT_MIN_MATCH_FRAC
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        
        factor_graph.solve_GN_calib()

        
        torch.cuda.empty_cache()
        

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)

def _save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)

def _save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        
        X_canon = constrain_points_to_ray(
            keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
        )
        keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)
    _save_ply(savedir / filename, pointclouds, colors)

class SLAMWrapper:
    def __init__(self, h, w, fixed_calib_params=None, apply_MAST3r_backend=True):
        self.img_size = DATASET_IMG_SIZE  
        self.h, self.w = h, w
        self.apply_BA = apply_MAST3r_backend
        self.device = "cuda:0"
        
        self.manager = mp.Manager()
        self.keyframes = SharedKeyframes(self.manager, self.h, self.w)
        self.states = SharedStates(self.manager, self.h, self.w)
        
        self.K = self._load_calibration(fixed_calib_params)
        
        self.keyframes.set_intrinsics(self.K)
        
        self.model = load_mast3r(device=self.device)
        self.model.share_memory()
        
        if self.apply_BA:
            self._start_processes()
        
        self.current_frame_idx = 0
        self.tracker = FrameTracker(self.model, self.keyframes, self.device)
        self.states.set_mode(Mode.INIT)

    def _start_processes(self):
        self.backend = mp.Process(
            target=run_backend,
            args=(config, self.model, self.states, self.keyframes, self.K)
        )
        self.backend.daemon = True
        self.backend.start()

    def _load_calibration(self, calib_params) -> torch.Tensor:
        intrinsics = Intrinsics.from_calib(
            self.img_size,
            calib_params["width"],
            calib_params["height"],
            calib_params["calibration"]
        )
        return torch.from_numpy(intrinsics.K_frame).to(self.device, dtype=torch.float32)

    def update(self, img: np.ndarray, timestamp: float = None):
        T_WC = (self.states.get_frame().T_WC if self.current_frame_idx > 0 
                else lietorch.Sim3.Identity(1, device=self.device))
        
        frame = create_frame(
            self.current_frame_idx,
            img,
            T_WC,
            img_size=self.img_size,
            device=self.device
        )
        
        mode = self.states.get_mode()
        add_new_kf = False
        
        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X_init, C_init)
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self.states.set_mode(Mode.TRACKING)
            self.states.set_frame(frame)
        
        elif mode == Mode.TRACKING:
            add_new_kf, _, _ = self.tracker.track(frame)
            self.states.set_frame(frame)

        if add_new_kf:
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
        
        self.current_frame_idx += 1
        return len(self.keyframes), frame.T_WC, mode, add_new_kf

    def save_results(self, save_dir: str):
        if not pathlib.Path(save_dir).exists():
            pathlib.Path(save_dir).mkdir(parents=True)
            
        _save_reconstruction(
            save_dir,
            "reconstruction.ply",
            self.keyframes,
            c_conf_threshold=0.5
        )

    def shutdown(self):
        self.states.set_mode(Mode.TERMINATED)
        if self.apply_BA:
            self.backend.join()
