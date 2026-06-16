

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from mapping.scene.scaffold_model import GaussianModel

from einops import repeat

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    level = pc.get_level[visible_mask].view(-1, 1)

    
    ob_view = anchor - viewpoint_camera.camera_center
    
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    
    ob_view = ob_view / ob_dist

    
    if pc.use_feat_bank:
        if pc.level_dim == 0:
            cat_view = torch.cat([ob_view], dim=1)
        else:
            cat_view = torch.cat([ob_view, level], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) 

        
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) 

    if pc.level_dim == 0:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) 
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) 
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) 
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) 
        
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        
        appearance = pc.get_appearance(camera_indicies)

    
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) 
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    
    opacity = neural_opacity[mask]

    
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color_input = torch.cat([cat_local_view, appearance], dim=1)
        else:
            color_input = torch.cat([cat_local_view_wodist, appearance], dim=1)
    else:
        if pc.add_color_dist:
            color_input = cat_local_view
        else:
            color_input = cat_local_view_wodist

    anchor_color = pc._anchor_color[visible_mask]
    color_residual = pc.get_color_mlp(color_input)
    color_residual = color_residual * 0.5

    base_color = anchor_color.unsqueeze(1).repeat(1, pc.n_offsets, 1).reshape(-1, 3)
    color = base_color + color_residual.reshape([anchor.shape[0] * pc.n_offsets, 3])
    color = torch.clamp(color, 0, 1)

    
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) 
    
    
    offsets = grid_offsets.view([-1, 3]) 
    
    
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) 
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot
    
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask = None):
    
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    

    
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points.requires_grad_(True)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    
    rendered_image, radii, depth, opacity, n_touched = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        
        theta=viewpoint_camera.cam_rot_delta,
        rho=viewpoint_camera.cam_trans_delta,
    )

    
    
    rendered_depth = depth.squeeze()

    
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "depth": rendered_depth,
                "opacity": opacity,
                "n_touched": n_touched,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": rendered_depth,
                "opacity": opacity,
                "n_touched": n_touched,
                }
