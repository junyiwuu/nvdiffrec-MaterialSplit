# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import time
import argparse
import json
import logging

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas
from omegaconf import OmegaConf
from util.logging_util import(
    recursive_load_config,
    config_logging,
    tb_logger,
)
from datetime import datetime, timedelta

# Import data readers / generators
# 数据集，支持不同类型的数据
from dataset.dataset_mesh import DatasetMesh
from dataset.dataset_nerf import DatasetNERF
from dataset.dataset_llff import DatasetLLFF

# Import topology / geometry trainers
# 几种不同的几何优化方式
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh
from geometry.flexicubes_geo import FlexiCubesGeometry


import render.renderutils as ru
from render.renderutils.loss import(
    MetricTracker,
)
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render


import omegaconf
from omegaconf import OmegaConf

import faulthandler; faulthandler.enable(all_threads=True)

RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################
# 份额变化一flag决定用哪种mse
@torch.no_grad()
def createLoss(loss_type ):
    if loss_type == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif loss_type == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif loss_type == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif loss_type == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif loss_type == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    elif loss_type == "perceptual":
        return lambda img, ref: ru.image_loss(img, ref, loss='perceptual', tonemapper='log_srgb')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################
# 根据传入的bg_type来选择混合给什么背景
@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background

    # add the background to image
    target['img'] = torch.cat((
        torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]),  # torch.lerp(starts, ends, weight)
        target['img'][..., 3:4]), dim=-1)

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    # got uv based on the position
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    # 用uv, 其他信息构建新的mesh. the returned mesh is the one with uv information
    # 这一步还只有uv的mesh，没有贴图
    # 
    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    # 将uv从mesh中反推出来，bake回来，uv就是刚生成的uv
    if FLAGS.separate_rough:
        mask, kd, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_normal'])
        _, ks = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['ks'])
    else:
        mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
        
    # 多层材质
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    # normlaize
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################
# mat = initial_guess_material(geometry, True, FLAGS)
def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    # when bool mlp = True
    if mlp:
        if FLAGS.separate_rough: # if FLAGS set separating the roughness
            # cat kd min and norm min
            mlp_min = torch.cat((kd_min[0:3], nrm_min), dim=0)
            mlp_max = torch.cat((kd_max[0:3], nrm_max), dim=0)

            mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=6, min_max=[mlp_min, mlp_max])
            mlp_ks_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=3, min_max=[ks_min, ks_max])
            
            mat =  material.Material({
                'kd_normal' : mlp_map_opt,
                'ks' : mlp_ks_map_opt
                })

        else:

            mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
            mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
            mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
            mat =  material.Material({'kd_ks_normal' : mlp_map_opt})

    # 要么使用普通贴图
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        # 如果没有初始贴图
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            # 完全随机初始化kd
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # 如果有初始贴图    
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}

    #validation阶段，不需要gradient
    with torch.no_grad():
        lgt.build_mips() # build light mipmaps, accelerate render 

        # 如果要用相机坐标系下的光照，对环境光做坐标变化
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])  # 在light.py中有xfm function

        # 调用geometry的渲染，生成当前优化结果的输出，buffer中通常有多个buffer
        buffers = geometry.render(glctx, target, lgt, opt_material)

        # 将ref=目标图像和当前geometry+材质渲染出来的RGB做相同的色彩空间变化
        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0] # 做colorspace的变换
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1) # 两张图拼在一起给你看

        # 额外的显示设置

        if FLAGS.display is not None:
            white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.render(glctx, target, layer['relight'], opt_material)
                    result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
   
        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):

    # 对所有的validation dataset，跑一遍当前的model，把结果图像保存，算分数，最后输出平均结果
    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    '''
    glctx: context
    geometry: current geometry representation
    opt_material: current material parameters
    lgt: current lighting
    dataset_validate: validation dataset
    out_dir
    FLAGS    
    '''

    # validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS)


    # initialize the metrics 
    mse_values = []
    psnr_values = []

    # 按bacth=1读取验证机数据， 相当于每帧检查，不能多于1.一张图对应一张target图算loss
    # 如果输入只有mesh，会自己渲染出target image sequence, 在datamesh.py
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)
    os.makedirs(out_dir, exist_ok=True)



    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')
        # 记录每张图的validate结果

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            # 合成background，统一batch
            target = prepare_batch(target, FLAGS.background)

            # 得到当前渲染结果
            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
            #-------------------- Compute metrics------------------
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)
            # logging.info(opt.shape)

            

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))



            # ---------------------------------------------------------
            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            # 记录每张渲染图
            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

            # show in tensorboard, concate two images
            opt = opt.cpu().permute(2, 0, 1)
            ref = ref.cpu().permute(2, 0, 1)
            opt_conct_ref = torch.cat([opt, ref], dim=2)
            
            tb_logger.log_img(img=opt_conct_ref, global_step=it)

        # calculate average metrics
        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr, avg_mse

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, loss_dict: dict, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.loss_dict = loss_dict
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()
        # 所有可训练参数给到这里
        # ---------------collect parameters
        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

    def forward(self, target, it):
        if self.optimize_light:

            #如果要优化llight，也就是light参与计算的话，让light生成mipmap
            self.light.build_mips()

            # 如果是camera空间的话，要对light做一个变换
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])
        
        return self.geometry.tick(glctx, target, self.light, self.material, self.loss_dict, it)
            #返回的是img_loss, reg_loss, or maybe ks_loss



# ---------------------------------- optimize mesh -------------------------------------------
def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    optimize_light=True,
    optimize_geometry=True,
    ):

    # ------- initialize ---------------
    if FLAGS.separate_rough:
        train_metrics = MetricTracker(*["img_loss", "reg_loss", "iter_time", "ks_loss"])
    else:
        train_metrics = MetricTracker(*["img_loss", "reg_loss", "iter_time"])

    start_time = time.time()


    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    # set the learning rate for pos and mat. 
    # learning rate is tuple, first one for pos, second one for mat
    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate



    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

 
    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    loss_dict = {}
    if FLAGS.ref_textures:
        loss_dict["ks_loss_fn"] = createLoss(FLAGS.ks_loss)
        loss_dict['image_loss_fn'] = createLoss(FLAGS.loss)
        logging.debug(f"choose {FLAGS.ks_loss} as loss for ks, choose {FLAGS.loss} as loss for other parts")
    else: 
        loss_dict['image_loss_fn'] = createLoss(FLAGS.loss)
        logging.debug(f"choose {FLAGS.loss} as loss for all")

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, loss_dict, FLAGS)


    if FLAGS.isosurface == 'flexicubes':
        betas = (0.7, 0.9)
    else:
        betas = (0.9, 0.999)

    # # ----------------- multi GPU situation --------------------
    # if FLAGS.multi_gpu: 
    #     # Multi GPU training mode
    #     import apex
    #     from apex.parallel import DistributedDataParallel as DDP

    #     trainer = DDP(trainer_noddp)
    #     trainer.train()
    #     if optimize_geometry:
    #         optimizer_mesh = apex.optimizers.FusedAdam(trainer_noddp.geo_params, lr=learning_rate_pos, betas=betas)
    #         scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    #     optimizer = apex.optimizers.FusedAdam(trainer_noddp.params, lr=learning_rate_mat)
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    # -------------------- single GPU situation ---------------------------
    # else:
    trainer = trainer_noddp
    if optimize_geometry:
        optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos, betas=betas)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    # 两个用不同的优化器
    optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
    # initialize the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    # 创建train 和validate 的 dataloader
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    # 为validation dataset做迭代器
    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    # 看下面对应的地方，当需要的时候可以立刻next(v_it)拿一张图做validate
    v_it = cycle(dataloader_validate)

    # 
    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        # 用随机背景
        target = prepare_batch(target, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                result_image, result_dict = validate_itr(
                    glctx, 
                    prepare_batch(next(v_it), FLAGS.background), 
                    geometry, opt_material, lgt, FLAGS)
                np_result_image = result_image.detach().cpu().numpy()
                if display_image:
                    util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                if save_image:
                    util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)
                    img_cnt = img_cnt+1

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        # 相当于geometry.tick，得到loss
        if not FLAGS.separate_rough:
            img_loss, reg_loss = trainer(target, it)
        else:
            img_loss, reg_loss, ks_loss = trainer(target, it)
        

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        # img_loss_vec.append(img_loss.item())
        # reg_loss_vec.append(reg_loss.item())

        train_metrics.update("img_loss", img_loss.item())
        train_metrics.update("reg_loss", reg_loss.item()) 
        train_metrics.update("iter_time", time.time() - start_time)
        if FLAGS.separate_rough:
            train_metrics.update("ks_loss" , ks_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()  # calculate gradient
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64 #lgt.base.grad是刚刚backward出来的提督，乘以64为了让光照更新更快，将其放大
        if 'kd_ks_normal' in opt_material:
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0 # 和光照相反，缩小让其更慢

        optimizer.step()
        scheduler.step()

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()
        # if FLAGS.separate_rough:
        #     optimizer_ks.step()
        #     scheduler_ks.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range 保证参数合理
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================

        # if it % 10 == 0:

        tb_logger.log_dic(
            {
                f"{pass_name}/train/{k}": v
                for k, v in train_metrics.result().items()
            },
            global_step=it,
        )

        tb_logger.writer.add_scalar(
            f"{pass_name}/lr",
            scheduler.get_last_lr()[0],
            global_step=it        
        )

            # logging.info(f"currently is iter : {it}")

        train_metrics.reset()


        # if it % log_interval == 0 and FLAGS.local_rank == 0:
            # img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            # reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            # iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            # remaining_time = (FLAGS.iter-it)*iter_dur_avg
            # print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
            #     (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
            
            # -------------------- log to tensorboard -------------------------
            

    return geometry, opt_material

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = datetime.now()
    print(f"start at {t_start}")

    # in parser, using dash, and when use FLAGS.xxx, use underscore 会自动把dash转成下划线    
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default='configs/bob.yaml', help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    # add ref texture forlder
    parser.add_argument('-rtex', '--ref_textures', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--isosurface', default='dmtet', choices=['dmtet', 'flexicubes'])
    
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True
    FLAGS.add_datetime_prefix = False
    # add for roughness MLP
    FLAGS.separate_rough      = True
    FLAGS.ks_loss             = "perceptual"




    #   ------------------------ multi GPU -----------------------------------
    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # -------------------use config / logging------------------------------------
    # if FLAGS.config is not None:
    #     data = json.load(open(FLAGS.config, 'r'))
    #     for key in data:
    #         FLAGS.__dict__[key] = data[key]

    cfg = recursive_load_config(FLAGS.config)
    for k, v in OmegaConf.to_container(cfg, resolve=True).items():
        setattr(FLAGS, k, v)

    pure_job_name = os.path.basename(FLAGS.config).split(".")[0]
    # add time prefix
    if FLAGS.add_datetime_prefix:
        job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
    else:
        job_name = pure_job_name

    # create output directory
    if FLAGS.out_dir is not None:
        out_dir_job = os.path.join(FLAGS.out_dir, job_name)
    else:
        out_dir_job = os.path.join('./output', job_name)
    os.makedirs(out_dir_job, exist_ok=True)

    # tensorboard
    out_dir_tb = os.path.join(out_dir_job, "tensorboard", FLAGS.loss)
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb)

    # tensorboard setup  设置tensorboard的路径
    tb_logger.set_dir(out_dir_tb)

    out_dir_loss = os.path.join(out_dir_job, FLAGS.loss)
    if not os.path.exists(out_dir_loss):
        os.makedirs(out_dir_loss)

    # evaluation folder
    out_dir_eval = os.path.join(out_dir_loss, "evaluation")
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval)


    # logging settings
    config_logging(cfg.logging, out_dir=out_dir_loss)
    logging.debug(f"config: {cfg}") # print all cfg information, save as debug level

    # -----------------display / out_dir----------------------------------------
    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = f"out/cube_{FLAGS.train_res[0]}x{FLAGS.train_res[1]}"
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir

    # if FLAGS.local_rank == 0:
    #     print("Config / Flags:")
    #     print("---------")
    #     for key in FLAGS.__dict__.keys():
    #         print(key, FLAGS.__dict__[key])
    #     print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # glctx = dr.RasterizeGLContext()
    glctx = dr.RasterizeCudaContext()

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    if os.path.splitext(FLAGS.ref_mesh)[1] == '.obj':
        if not FLAGS.ref_textures:
            ref_mesh         = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
            dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False)
            dataset_validate = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=True)
        else: # using seperate textures, not mtl
            ref_mesh         = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override, textures_path=FLAGS.ref_textures)
            dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False)
            dataset_validate = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=True)

    # nerf situation
    elif os.path.isdir(FLAGS.ref_mesh):
        if os.path.isfile(os.path.join(FLAGS.ref_mesh, 'poses_bounds.npy')):
            dataset_train    = DatasetLLFF(FLAGS.ref_mesh, FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
            dataset_validate = DatasetLLFF(FLAGS.ref_mesh, FLAGS)
        elif os.path.isfile(os.path.join(FLAGS.ref_mesh, 'transforms_train.json')):
            dataset_train    = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_train.json'), FLAGS, examples=(FLAGS.iter+1)*FLAGS.batch)
            dataset_validate = DatasetNERF(os.path.join(FLAGS.ref_mesh, 'transforms_test.json'), FLAGS)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    if FLAGS.base_mesh is None:
        # ==============================================================================================
        #  If no initial guess, use DMtets to create geometry
        # ==============================================================================================
        logging.info("No initial mesh provided, use DMTets to create geometry")
        # Setup geometry for optimization

        if FLAGS.isosurface == 'flexicubes':
            logging.info("flexicubes selected")
            geometry = FlexiCubesGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        elif FLAGS.isosurface == 'dmtet':
            logging.info("DMTet selected")
            geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        else: 
            assert False, "Invalid isosurfacing %s" % FLAGS.isosurface

        # Setup textures, make initial guess from reference if possible
        mat = initial_guess_material(geometry, True, FLAGS)

        # Run optimization
        logging.info("pass 1 started ")
        geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, 
                        FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=FLAGS.learn_light)

      


        if FLAGS.local_rank == 0 and FLAGS.validate:
            logging.info("starting geometry evaluation")
            out_dir_eval_dmtet = os.path.join(out_dir_eval, "dmtet_validate")
            if not os.path.exists(out_dir_eval_dmtet):
                os.makedirs(out_dir_eval_dmtet)

            avg_psnr, avg_mse = validate(glctx, geometry, mat, lgt, dataset_validate, out_dir_eval_dmtet , FLAGS)
            # validate(glctx, geometry, mat, lgt, dataset_validate, out_dir_eval, FLAGS)
            tb_logger.writer.add_text(
                "val/dmtet", f"average psnr: {avg_psnr}\n\
                    average mse: {avg_mse}"
            )



        # Create textured mesh from result
        base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

        # Free temporaries / cached memory 
        torch.cuda.empty_cache()


        if FLAGS.separate_rough:
            mat['kd_normal'].cleanup()
            del mat['kd_normal']
            mat['ks'].cleanup()
            del mat['ks']
        else:
            mat['kd_ks_normal'].cleanup()
            del mat['kd_ks_normal']
           

        lgt = lgt.clone()
        geometry = DLMesh(base_mesh, FLAGS)

        

        if FLAGS.local_rank == 0:
            # Dump mesh for debugging.
            os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
            obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
            light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

        # ==============================================================================================
        #  Pass 2: Train with fixed topology (mesh)
        # ==============================================================================================
        logging.info("pass2 : Starting train with fixed topology (DLMesh)")
        geometry, mat = optimize_mesh(glctx, geometry, base_mesh.material, lgt, dataset_train, dataset_validate, FLAGS, 
                    pass_idx=1, pass_name="mesh_pass", warmup_iter=100, optimize_light=FLAGS.learn_light and not FLAGS.lock_light, 
                    optimize_geometry=not FLAGS.lock_pos)
    else:
        # ==============================================================================================
        #  Train with fixed topology (mesh)
        # ==============================================================================================
        logging.info("Initial mesh is provided, train with fixed mesh")
        # Load initial guess mesh from file
        base_mesh = mesh.load_mesh(FLAGS.base_mesh)
        geometry = DLMesh(base_mesh, FLAGS)
        
        mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)

        geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, FLAGS, pass_idx=0, pass_name="mesh_pass", 
                                        warmup_iter=100, optimize_light=not FLAGS.lock_light, optimize_geometry=not FLAGS.lock_pos)

    # ==============================================================================================
    #  Validate
    # ==============================================================================================
    if FLAGS.validate and FLAGS.local_rank == 0:
        logging.info("starting mesh evaluation")
        out_dir_eval_mesh = os.path.join(out_dir_eval, "mesh_validate")
        if not os.path.exists(out_dir_eval_mesh):
            os.makedirs(out_dir_eval_mesh)
        # validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "validate"), FLAGS)
        avg_psnr, avg_mse = validate(glctx, geometry, mat, lgt, dataset_validate, out_dir_eval_mesh, FLAGS)

        # validate(glctx, geometry, mat, lgt, dataset_validate, out_dir_eval, FLAGS)
        tb_logger.writer.add_text(
            "val/mesh", f"average psnr: {avg_psnr}\n\
                average mse: {avg_mse}"
        )
    

    # ==============================================================================================
    #  Dump output
    # ==============================================================================================
    if FLAGS.local_rank == 0:
        final_mesh = geometry.getMesh(mat)
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

#----------------------------------------------------------------------------
