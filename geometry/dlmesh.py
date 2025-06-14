# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import logging

from render import mesh
from render import render
from render import regularizer

###############################################################################
#  Geometry interface
###############################################################################

class DLMesh(torch.nn.Module):
    def __init__(self, initial_guess, FLAGS):
        super(DLMesh, self).__init__()

        self.FLAGS = FLAGS

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        
        # 所有的vertex are nn.parameter 可以被优化，可参与differentiable
        self.mesh.v_pos = torch.nn.Parameter(self.mesh.v_pos, requires_grad=True)
        self.register_parameter('vertex_pos', self.mesh.v_pos)

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        # 设置当前的material
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_dict, iteration):
        
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']

        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        loss_fn = loss_dict['image_loss_fn']
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])  # apply mask
        # img loss  is typical calculate the final render with target

        if self.FLAGS.separate_rough: 
            # create loss for ks optimization
            ks_loss_fn = loss_dict['ks_loss_fn']
            ks_loss = ks_loss_fn(buffers['shaded'][... , 0:3] * color_ref[... , 3:],
                                color_ref[... , 0:3] * color_ref[... , 3:]) 
            ks_loss = ks_loss * 0.5 + img_loss 

            # if iteration % 20==0:
            #     logging.info(f"perceptual loss: {ks_loss}")
            #     logging.info(f"img_loss: {torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])}" )



        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer. 
        if self.FLAGS.laplace == "absolute":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)
        elif self.FLAGS.laplace == "relative":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)                

        # Albedo (k_d) smoothnesss regularizer
        # [..., :-1]是实际像素点，乘以[..., -1: ]是apha，就是有效的部分x
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.005

        if self.FLAGS.separate_rough: 
            return img_loss, reg_loss, ks_loss
        else:
            return img_loss, reg_loss
