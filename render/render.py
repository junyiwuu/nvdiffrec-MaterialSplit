# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr
import logging

from . import util
from . import renderutils as ru
from . import light

# ==============================================================================================
#  Helper functions
# ==============================================================================================
# 用nvdiffrast根据rasterization的结果，将mesh的属性interpolate到each pixel
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), 
                          rast, 
                          attr_idx, 
                          rast_db=rast_db, 
                          diff_attrs=None 
                            if rast_db is None else 'all')

# ==============================================================================================
#  计算和光交互后的shaded, kd_grad, occlusion
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        lgt,
        material,
        bsdf,
        disable_occl
    ):
    # logging.debug("start shade")

    perturbed_nrm = None
    # kd_ks_normal是合成材质
    if 'kd_ks_normal' in material:
        # Combined texture, used for MLPs because lookups are expensive
        # torch.normal是生成gaussian noise
        # MLP代表的是材质场，所以必须输入3D位置，告诉他“我在哪”才能返回这个地方的材质
        all_tex_jitter = material['kd_ks_normal'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
        # 没有加noise的
        all_tex = material['kd_ks_normal'].sample(gb_pos)

        # 确认channel正确，应该是9或10
        assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"

        # 拆开，kd ks和perturbed normal
        kd, ks, perturbed_nrm = all_tex[..., :-6], all_tex[..., -6:-3], all_tex[..., -3:]

        # Compute albedo (kd) gradient, used for material regularizer
        # 算带jitter和不带jitter的差值
        kd_grad    = torch.sum(torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]), dim=-1, keepdim=True) / 3

    # separate ks as a mlp. still do kd grad
    elif 'kd_normal' in material and "ks" in material: # when using separate rough mlp
        kd_norm_tex_jitter = material['kd_normal'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
        kd_norm_tex = material['kd_normal'].sample(gb_pos)

        assert kd_norm_tex.shape[-1] == 6 or kd_norm_tex.shape[-1] == 7, "if separate ks, kd_normal must be 6 or 7 channels"

        kd, perturbed_nrm = kd_norm_tex[..., :-3], kd_norm_tex[..., -3: ]
        if disable_occl:
            ks2_tex_jitter = material['ks'].sample(gb_pos + torch.normal(mean=0, std=0.003, size=gb_pos.shape, device="cuda"))
            ks2 = material['ks'].sample(gb_pos)
            # mlp is two channel, add one more for occlusion

            occlusion = torch.zeros_like(ks2[..., :1])
            ks = torch.cat([occlusion, ks2], dim=-1)

            ks_grad    = torch.sum(torch.abs(ks2_tex_jitter[..., : ] - ks2[..., : ]), dim=-1, keepdim=True) / 3
        else:
            ks_tex_jitter = material['ks'].sample(gb_pos + torch.normal(mean=0, std=0.003, size=gb_pos.shape, device="cuda"))
            ks = material['ks'].sample(gb_pos)

            ks_grad    = torch.sum(torch.abs(ks_tex_jitter[..., : ] - ks[..., : ]), dim=-1, keepdim=True) / 2

        kd_grad = torch.sum(torch.abs(kd_norm_tex_jitter[..., :-3] - kd_norm_tex[..., :-3]) , dim=-1, keepdim=True)/3



    else:
        # Texture2D case
        # 如果不是MLP合成贴图，那就每个单个来采样
        kd_jitter  = material['kd'].sample(gb_texc + torch.normal(mean=0, std=0.005, size=gb_texc.shape, device="cuda"), gb_texc_deriv)
        # 在uv坐标采样albedo
        kd = material['kd'].sample(gb_texc, gb_texc_deriv)

        kd_grad    = torch.sum(torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True) / 3

        # 如果有normal的话采样normal
        if 'normal' in material:
            perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)

        # 用uv坐标采样specular，只取前三个通道
        # logging.info(f"what is kd size: {material['kd'].getMips()[0].shape}")
        # logging.info(f"what is ks size: {material['ks'].getMips()[0].shape}")
        ks_jitter  = material['ks'].sample(gb_texc + torch.normal(mean=0, std=0.00, size=gb_texc.shape, device="cuda"), gb_texc_deriv)
        # based on current material['ks] - which nn.parameter   -> sample on the uv cooridnate, and got result 
        if disable_occl:
            
            ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:2] # skip alpha
            occlusion = torch.zeros_like(ks[..., :1])         

            ks = torch.cat([occlusion, ks], dim=-1)    
            ks_jitter = torch.cat([occlusion, ks_jitter], dim=-1) 

        else:
            ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
        
        ks_grad    = torch.sum(torch.abs(ks_jitter[..., 0:3] - ks[..., 0:3]), dim=-1, keepdim=True) / 3


    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    kd = kd[..., 0:3]

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    if 'no_perturbed_nrm' in material and material['no_perturbed_nrm']:
        perturbed_nrm = None

    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, 
                                          perturbed_nrm, gb_normal, 
                                          gb_tangent, gb_geometric_normal, 
                                          two_sided_shading=True, opengl=True)

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert 'bsdf' in material or bsdf is not None, "Material must specify a BSDF type"
    bsdf = material['bsdf'] if bsdf is None else bsdf
    if bsdf == 'pbr':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'diffuse':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=False)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'normal':
        shaded_col = (gb_normal + 1.0)*0.5
    elif bsdf == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    elif bsdf == 'kd':
        shaded_col = kd
    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf
    
    # Return multiple buffers
    buffers = {
        'shaded'    : torch.cat((shaded_col, alpha), dim=-1),
        'kd_grad'   : torch.cat((kd_grad, alpha), dim=-1),
        'occlusion' : torch.cat((ks[..., :1], alpha), dim=-1),

        'ks_grad'   : torch.cat((ks_grad, alpha), dim=-1),
    }
    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf,
        disable_occl
    ):
    # logging.debug("start render layer")
    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position 得到每个像素对应的世界坐标
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0, dim=-1))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    # 面法线
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    ################################################################################
    # Shade
    ################################################################################

    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, lgt, mesh.material, bsdf, disable_occl=disable_occl)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
##############################
#####  渲染frame buffer  #####   最终输出,在这里面将所有的layer给混合成最终的屏幕上的图片
##############################
def render_mesh(
        ctx,
        mesh, 
        mtx_in,
        view_pos,
        lgt,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None, 
        bsdf        = None,
        disable_occl=False
    ):
    # logging.debug("start render mesh")
    # 处理shape
    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    # -------------------- world space -> clip space ---------------------------
    def composite_buffer(key, layers, background, antialias, ):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, lgt, resolution, spp, msaa, bsdf, disable_occl=disable_occl), rast)]

    # ----------------------Setup background----------------
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        # 全黑
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # -------------------------Composite layers front-to-back---------------------------
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded': # 只有shaded也就是最终图像结果要和背景合成，要用antialias，
            accum = composite_buffer(key, layers, background, True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    
    all_tex = mlp_texture.sample(gb_pos)
    # assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    # util.norm: 把perturbed normal normalized化，避免数值不规范导致shading的问题

    if all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10:
        return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], util.safe_normalize(perturbed_nrm)        
    if all_tex.shape[-1] == 6 or all_tex.shape[-1] == 7:
        return (rast[..., -1:] > 0).float(), all_tex[..., :-3], util.safe_normalize(perturbed_nrm)
    if all_tex.shape[-1] == 3:
        return (rast[..., -1:] > 0).float(), all_tex
    if all_tex.shape[-1] == 2: # ks only do roughness and metallic, occlusion is 0
        occlusion = torch.zeros_like(all_tex[... , :1])
        all_tex = torch.cat([occlusion, all_tex], dim=-1)
        return (rast[..., -1:] > 0).float(), all_tex



