# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder_224 import MaskDecoder_224, MaskDecoder2_224
from .prompt_encoder import PromptEncoder
from .encoder_decoder import MyDecoder, MyEncoder
import cv2
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from segment_anything_Tea.modeling.Pvtv2 import *



class prompt_generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv3 = nn.Sequential(
                        nn.Conv2d( 768, 256, 1),
                        nn.Conv2d(256, 256, 3, padding=1))
        
        self.conv2 = nn.Sequential(
                        nn.Conv2d(768, 256, 1),
                        nn.Conv2d(256, 256, 3, padding=1))

        self.conv1 = nn.Sequential(
                        nn.Conv2d(768, 256, 1),
                        nn.Conv2d(256, 256, 3, padding=1))
       
        self.co3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),  # 分成 4 组
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())
        self.co2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),  # 分成 4 组
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())
        self.co1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),  # 分成 4 组
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())
        self.cout = nn.Sequential(nn.Conv2d(256, 1, 1),
                                  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
    
    def forward(self, inter_features, image_embeddings):
        
        x1, x2, x3 = inter_features[0], inter_features[1], inter_features[2]
        x3 = self.conv3(x3)
        x2 = self.conv2(x2)
        x1 = self.conv1(x1)

        mul34 = image_embeddings * x3
        x3_new = torch.abs(mul34 - x3)
        x4_new = torch.abs(mul34 - image_embeddings)
        add34 = self.co3(mul34 + x3_new + x4_new)

        mul23 = add34 * x2
        add34_new = torch.abs(mul23 - add34)
        x2_new = torch.abs(mul23 - x2)
        add23 = self.co2(mul23 + x2_new + add34_new)

        mul12 = add23 * x1
        add23_new = torch.abs(mul12 - add23)
        x1_new = torch.abs(mul12 - x1)
        add12 = self.co1(mul12 + x1_new + add23_new)

        out = self.cout(add12)
        
        return out


class ImplicitPriorEmbedder(nn.Module):
    def __init__(self, in_channels, latent_dim=128):
        super().__init__()
        # 坐标编码器
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim))
        
        # 隐式解码器
        self.implicit_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels))
        
        # 先验条件生成器
        self.prior_conditioner = nn.Sequential(
            nn.Conv2d(9, latent_dim, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, seg_prior):
        B, C, H, W = x.shape
        # 生成网格坐标
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        coords = torch.stack([grid_x, grid_y], dim=-1).to(x.device).repeat(B, 1, 1, 1)
        
        # 编码坐标
        coord_feat = self.coord_encoder(coords.view(B*H*W, 2))
        
        # 从分割先验生成条件向量
        condition = self.prior_conditioner(seg_prior).view(B, 1, -1)
        condition = condition.repeat(1, H*W, 1).view(B*H*W, -1)
        
        # 融合坐标特征和条件
        fused_feat = torch.cat([coord_feat, condition], dim=1)

        # 解码为特征调制参数
        mod_params = self.implicit_decoder(fused_feat).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # 应用调制
        return x * mod_params.sigmoid() + mod_params


class BCAF(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.d_head = dim // heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.weight1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.weight2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    ################  因果注意力掩码 ################################
    def linear_causal_attention(self, q, k, v):
        # Φ(x) = relu(x)  也可以换成 elu, exp
        q = F.relu(q)
        k = F.relu(k)

        B, H, N, D = q.shape

        # 累加和因果 attention 实现（prefix sum）
        # 用 cumsum 保证当前只能看到历史
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # [B, H, N, D, D]
        kv_cumsum = kv.cumsum(dim=2)           # 累加 [B, H, N, D, D]
        k_cumsum = k.cumsum(dim=2)             # 累加 [B, H, N, D]

        # output[i] = q[i] @ (sum_{j=0}^i k[j]^T v[j]) / (q[i] @ sum_{j=0}^i k[j])
        out = torch.einsum('bhnc,bhncd->bhnd', q, kv_cumsum)
        normalizer = torch.einsum('bhnc,bhnd->bhn', q, k_cumsum).unsqueeze(-1) + 1e-6
        out = out / normalizer

        return out  # [B, H, N, D]

    ##################  没有掩码  ################################
    # def linear_causal_attention(self, q, k, v):
    #     # Φ(x) = relu(x)  也可以换成 elu, exp
    #     q = F.relu(q)
    #     k = F.relu(k)

    #     B, H, N, D = q.shape

    #     kv = torch.einsum('bhnk,bhnv->bhkv', k, v)
    #     z = torch.einsum('bhnk,bhk->bhn', q, k.sum(dim=2)) + 1e-6
    #     out = torch.einsum('bhnk,bhkv->bhnv', q, kv) / z.unsqueeze(-1)

    #     return out  # [B, H, N, D]

    def forward(self, x3, x4):
        # x3, x4: [B, C, H, W]
        b3, c3, h3, w3 = x3.shape
        x3_flat = x3.reshape(b3, c3, -1).permute(0, 2, 1)  # [B, N3, C]
        x4_flat = x4.reshape(b3, c3, -1).permute(0, 2, 1)  # [B, N4, C]

        B, N3, C = x3_flat.shape
        N4 = x4_flat.shape[1]
        H = self.heads

        # Project
        q3 = self.q_proj(x3_flat).view(B, N3, H, self.d_head).transpose(1, 2)
        q4 = self.q_proj(x4_flat).view(B, N4, H, self.d_head).transpose(1, 2)

        k3, v3 = self.kv_proj(x3_flat).chunk(2, dim=-1)
        k3 = k3.view(B, N3, H, self.d_head).transpose(1, 2)
        v3 = v3.view(B, N3, H, self.d_head).transpose(1, 2)

        k4, v4 = self.kv_proj(x4_flat).chunk(2, dim=-1)
        k4 = k4.view(B, N4, H, self.d_head).transpose(1, 2)
        v4 = v4.view(B, N4, H, self.d_head).transpose(1, 2)

        # Cross Attention
        out3_cross = self.linear_causal_attention(q3, k4, v4)
        out4_cross = self.linear_causal_attention(q4, k3, v3)

        # Self Attention
        out3_self = self.linear_causal_attention(q3, k3, v3)
        out4_self = self.linear_causal_attention(q4, k4, v4)

        # Project back
        out3_cross = out3_cross.transpose(1, 2).reshape(B, N3, C)
        out4_cross = out4_cross.transpose(1, 2).reshape(B, N4, C)

        out3_self = out3_self.transpose(1, 2).reshape(B, N3, C)
        out4_self = out4_self.transpose(1, 2).reshape(B, N4, C)

        # 合并 Cross 和 Self 分支（融合方式可微调）
        out3 = self.dropout(self.out_proj(out3_cross + out3_self))
        out4 = self.dropout(self.out_proj(out4_cross + out4_self))

        # 相似度增强（你原本定义的融合方式）
        sim = out3 * out4
        out3 = nn.ReLU()(x3_flat - out3) + out3 + sim
        out4 = nn.ReLU()(x4_flat - out4) + out4 + sim

        # 恢复空间结构
        out3 = out3.reshape(b3, h3, w3, -1).permute(0, 3, 1, 2)
        out4 = out4.reshape(b3, h3, w3, -1).permute(0, 3, 1, 2)

        return out3, out4



class decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bcaf3 = BCAF(320)
        self.bcaf2 = BCAF(128)
        self.bcaf1 = BCAF(64)

        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512, bias=False),  # depthwise
            nn.Conv2d(512, 320, kernel_size=1, bias=False),  # pointwise,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, padding=1, groups=320, bias=False),  # depthwise
            nn.Conv2d(320, 128, kernel_size=1, bias=False),  # pointwise,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),  # depthwise
            nn.Conv2d(128, 64, kernel_size=1, bias=False),  # pointwise,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.co3 = nn.Sequential(
                                 nn.Conv2d(320, 320, 3, padding=1, groups=320),  # 分成 4 组
                                 nn.BatchNorm2d(320),
                                 nn.ReLU())
        self.co2 = nn.Sequential(
                                 nn.Conv2d(128, 128, 3, padding=1, groups=128),  # 分成 4 组
                                 nn.BatchNorm2d(128),
                                 nn.ReLU())
        self.co1 = nn.Sequential(
                                 nn.Conv2d(64, 64, 3, padding=1, groups=64),  # 分成 4 组
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())
        self.cout = nn.Sequential(nn.Conv2d(64, 9, 1),
                                  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))

    def forward(self, x1, x2, x3, x4):

        x4 = self.conv3(x4)
        x3, x4 = self.bcaf3(x3, x4)
        mul34 = x4 * x3
        x3_new = torch.abs(x3 - mul34)
        x4_new = torch.abs(x4 - mul34)
        add34 = self.co3(mul34 + x3_new + x4_new)

        add34 = self.conv2(add34)
        x2, add34 = self.bcaf2(x2, add34)
        mul23 = add34 * x2
        add34_new = torch.abs(add34 - mul23)
        x2_new = torch.abs(x2 - mul23)
        add23 = self.co2(mul23 + add34_new + x2_new)

        add23 = self.conv1(add23)
        x1, add23 = self.bcaf1(x1, add23)
        mul12 = add23 * x1
        add23_new = torch.abs(add23 - mul12)
        x1_new = torch.abs(x1 - mul12)
        add12 = self.co1(mul12 + x1_new + add23_new)

        out = self.cout(add12)

        return out



class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder_224,
        # mask_decoder2: MaskDecoder2_224,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375]
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()

        self.image_encoder = image_encoder
        # self.prompt_generator = prompt_generator()
        self.prompt_encoder = prompt_encoder
        # for param in self.prompt_encoder.parameters():
        #     param.requires_grad = False
        self.mask_decoder = mask_decoder

        # self.pvtv2_v2 = pvt_v2_b2()  # 64, 128, 320, 512
        # save_model_v2 = torch.load('./segment_anything/pvt_v2_b2.pth')
        # model_dict_v2 = self.pvtv2_v2.state_dict()
        # state_dict_v2 = {k: v for k, v in save_model_v2.items() if k in model_dict_v2.keys()}
        # model_dict_v2.update(state_dict_v2)
        # self.pvtv2_v2.load_state_dict(model_dict_v2)
        # self.decoder = decoder()
        # self.enhance = ImplicitPriorEmbedder(3)

        self.reduce_factor = 4
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)


    @property
    def device(self) -> Any:
        return self.pixel_mean.device


    def forward(self, batched_input, multimask_output, gt=None, mode='train'):
        sam_out = self.forward_train(batched_input, multimask_output, gt=gt, mode=mode)

        # input_enhance = self.enhance(batched_input, sam_out)
        # x1, x2, x3, x4 = self.pvtv2_v2(input_enhance)
        # out = self.decoder(x1, x2, x3, x4)
            
        return sam_out

    def forward_train(self, batched_input, multimask_output, gt, mode):

        input_images = self.preprocess(batched_input)
        inter_features, image_embeddings, low_image_embeddings = self.image_encoder(input_images)

        # mask_prompt = self.prompt_generator(inter_features, image_embeddings)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions, msk_feat, up_embed = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            gt=gt,
            mode=mode
        )

        return self.up4(low_res_masks)#, image_embeddings, inter_features


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
        
        

