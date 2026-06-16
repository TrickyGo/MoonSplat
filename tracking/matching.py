import torch
import torch.nn.functional as F
import tracking_backends


def img_gradient(img):
    device = img.device
    dtype = img.dtype
    b, c, h, w = img.shape

    gx_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gx_kernel = gx_kernel.repeat(c, 1, 1, 1)

    gy_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gy_kernel = gy_kernel.repeat(c, 1, 1, 1)

    gx = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gx_kernel,
        groups=img.shape[1],
    )

    gy = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gy_kernel,
        groups=img.shape[1],
    )

    return gx, gy

MATCHING_CFG = {
    "max_iter": 10,
    "lambda_init": 1e-8,
    "convergence_thresh": 1e-6,
    "dist_thresh": 1e-1,
    "radius": 3,
    "dilation_max": 5,
}

def match(X11, X21, D11, D21, idx_1_to_2_init=None):
    idx_1_to_2, valid_match2 = match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init)
    return idx_1_to_2, valid_match2

def pixel_to_lin(p1, w):
    idx_1_to_2 = p1[..., 0] + (w * p1[..., 1])
    return idx_1_to_2

def lin_to_pixel(idx_1_to_2, w):
    u = idx_1_to_2 % w
    v = idx_1_to_2 // w
    p = torch.stack((u, v), dim=-1)
    return p

def prep_for_iter_proj(X11, X21, idx_1_to_2_init):
    b, h, w, _ = X11.shape
    device = X11.device

    
    rays_img = F.normalize(X11, dim=-1)
    rays_img = rays_img.permute(0, 3, 1, 2)  
    gx_img, gy_img = img_gradient(rays_img)
    rays_with_grad_img = torch.cat((rays_img, gx_img, gy_img), dim=1)
    rays_with_grad_img = rays_with_grad_img.permute(
        0, 2, 3, 1
    ).contiguous()  

    
    X21_vec = X21.view(b, -1, 3)
    pts3d_norm = F.normalize(X21_vec, dim=-1)

    
    if idx_1_to_2_init is None:
        
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    p_init = lin_to_pixel(idx_1_to_2_init, w)
    p_init = p_init.float()

    return rays_with_grad_img, pts3d_norm, p_init

def match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init=None):
    cfg = MATCHING_CFG
    b, h, w = X21.shape[:3]
    device = X11.device

    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(
        X11, X21, idx_1_to_2_init
    )
    p1, valid_proj2 = tracking_backends.iter_proj(
        rays_with_grad_img,
        pts3d_norm,
        p_init,
        cfg["max_iter"],
        cfg["lambda_init"],
        cfg["convergence_thresh"],
    )
    p1 = p1.long()

    
    batch_inds = torch.arange(b, device=device)[:, None].repeat(1, h * w)
    dists2 = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21, dim=-1
    )
    valid_dists2 = (dists2 < cfg["dist_thresh"]).view(b, -1)
    valid_proj2 = valid_proj2 & valid_dists2

    if cfg["radius"] > 0:
        (p1,) = tracking_backends.refine_matches(
            D11.half(),
            D21.view(b, h * w, -1).half(),
            p1,
            cfg["radius"],
            cfg["dilation_max"],
        )

    
    idx_1_to_2 = pixel_to_lin(p1, w)

    return idx_1_to_2, valid_proj2.unsqueeze(-1)
