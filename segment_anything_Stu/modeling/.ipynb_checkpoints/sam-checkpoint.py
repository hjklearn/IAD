# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder_224 import MaskDecoder_224, MaskDecoder2_224
# from .mask_decoder_512 import MaskDecoder_512

from .prompt_encoder import PromptEncoder
from .encoder_decoder import MyDecoder, MyEncoder
from .Pvtv2 import pvt_v2_b0, pvt_v2_b2, pvt_v2_b3
from .maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
import cv2
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .WTconv import *
from PIL import Image
import scipy.misc
import sys
sys.path.append('/root/autodl-tmp/H-SAM_new/')


class MPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, padd):
        super(MPPM, self).__init__()
        self.features_avg = []
        for pa in padd:
            self.features_avg.append(nn.Sequential(
                                nn.AdaptiveAvgPool2d(pa),
                                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                                nn.GELU(),))
        
        self.features_max = []
        for pa in padd:
            self.features_max.append(nn.Sequential(
                                nn.AdaptiveMaxPool2d(pa),
                                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                                nn.GELU(),))
            
        self.features_avg = nn.ModuleList(self.features_avg)
        self.features_max = nn.ModuleList(self.features_max)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding = 1, bias=False, groups = in_dim),
            nn.GELU(),
        )
        

    def forward(self, x):
        x_size = x.size()
        
        out_avg = []
        out_max = []
        out = [self.local_conv(x)]
        for f in self.features_avg:
            out_avg.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
            
        for f in self.features_max:
            out_max.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
            
        add1 = out_avg[0] + out_max[0]
        add2 = out_avg[1] + out_max[1]
        add3 = out_avg[2] + out_max[2]
        add4 = out_avg[3] + out_max[3]
        out.append(add1)
        out.append(add2)
        out.append(add3)
        out.append(add4)
            
        return torch.cat(out, 1)

class MFOM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.MPPM = MPPM(hidden_dim, hidden_dim //4,  [3,6,9,12])
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),
                                       nn.GELU())
        self.conv3 = nn.Conv2d(hidden_dim, in_dim, kernel_size=1)

    def forward(self, x):
        original = x

        x = self.conv1(x)
        x = self.gelu(x)

        x = self.MPPM(x).contiguous()
        x = self.conv2(x)
        x = self.conv3(x)
        
        return original + x
    

# class WTDaspp(nn.Module):
#     def __init__(self, in_planes):
#         super(WTDaspp, self).__init__()
        
#         self.wt_1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//2, kernel_size=1),
#                                   WTConv2d(in_channels=in_planes//2, out_channels=in_planes//2, kernel_size=3, dilation=1))
#         self.wt_3 = nn.Sequential(nn.Conv2d(in_planes, in_planes//2, kernel_size=1),
#                                   WTConv2d(in_channels=in_planes//2, out_channels=in_planes//2, kernel_size=3, dilation=3))
#         self.wt_6 = nn.Sequential(nn.Conv2d(in_planes, in_planes//2, kernel_size=1),
#                                   WTConv2d(in_channels=in_planes//2, out_channels=in_planes//2, kernel_size=3, dilation=6))
#         self.wt_9 = nn.Sequential(nn.Conv2d(in_planes, in_planes//2, kernel_size=1),
#                                   WTConv2d(in_channels=in_planes//2, out_channels=in_planes//2, kernel_size=3, dilation=9))
#         self.conv_out = nn.Sequential(
#                                 nn.Conv2d(in_planes*2, in_planes, kernel_size=1),
#                                 nn.BatchNorm2d(in_planes),
#                                 nn.ReLU())
        
#     def forward(self, coarse):

#         x_wt1 = self.wt_1(coarse)
#         x_wt3 = self.wt_3(coarse)
#         x_wt6 = self.wt_6(coarse)
#         x_wt9 = self.wt_9(coarse)
#         cat_x = torch.cat((x_wt1, x_wt3, x_wt6, x_wt9), dim=1)
#         out = self.conv_out(cat_x) + coarse

#         return out


class Deoder_MSM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.msm4 = MFOM(512, 256)
        self.conv4 = nn.Sequential(
                                   nn.Conv2d(512, 320, 1),
                                   nn.BatchNorm2d(320),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.msm3 = MFOM(320, 160)
        self.conv3 = nn.Sequential(
                                   nn.Conv2d(320, 128, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.msm2 = MFOM(128, 64)
        self.conv2 = nn.Sequential(
                                   nn.Conv2d(128, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.msm1 = MFOM(64, 32)
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(64, 9, 1))
        
        self.up4 = nn.Sequential(
                                   nn.Conv2d(512, 128, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.up81 = nn.Sequential(
                                   nn.Conv2d(512, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        self.up82 = nn.Sequential(
                                   nn.Conv2d(320, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
    
    def forward(self, x1, x2, x3, x4):
        
        m4 = self.msm4(x4)

        add3 = self.conv4(m4) + x3
        m3 = self.msm3(add3)

        add2 = self.conv3(m3) + x2 + self.up4(m4)
        m2 = self.msm2(add2)

        add1 =self.conv2(m2) + x1 + self.up81(m4) + self.up82(m3)
        m1 = self.msm1(add1)

        out = self.conv1(m1)

        return out, m1, m2, m3, m4


# class AttentionGate(nn.Module):
#     def __init__(self, in_c, size_n):
#         super().__init__()
#         self.conv = nn.Sequential(nn.Conv2d(9, in_c, 1),
#                                    nn.MaxPool2d(size_n, size_n))
#         self.spatial_att  = nn.Sequential(
#             WTConv2d(in_c, in_c, kernel_size=5, stride=1, bias=True, wt_levels=1,),
#             nn.ReLU(),
#             WTConv2d(in_c, in_c, kernel_size=5, stride=1, bias=True, wt_levels=1,),
#             nn.Sigmoid())
        
#     def forward(self, first_coarse):
#         first_coarse_conv = self.conv(first_coarse)
#         out = self.spatial_att(first_coarse_conv)

#         return out, first_coarse_conv



class Maskfusion(nn.Module):
    def __init__(self, size_n, channel):
        super().__init__()
        self.avg  = nn.AdaptiveAvgPool2d(size_n)
        self.max  = nn.AdaptiveMaxPool2d(size_n)
        self.sig = nn.Sigmoid()

        self.conv1 = nn.Sequential(nn.Conv2d(9, channel, 1),
                                   nn.AdaptiveMaxPool2d(size_n))
        self.spatial_att  = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.Sigmoid())

        
    def forward(self, image_embeddings, c0):
        
        c0_parts = [c0[:, i:i+1, :, :] for i in range(9)] 
        
        outputs = []
        for idx, c0_part in enumerate(c0_parts):
          
            c0_avg = self.avg(c0_part)
            c0_max = self.max(c0_part)
            
            add_feat = c0_avg + c0_max
            add_feat = self.sig(add_feat)
              
            mul_out = add_feat * image_embeddings
            outputs.append(mul_out)
        
        final_output = torch.stack(outputs).sum(dim=0)
        
        c0_conv = self.conv1(c0)
        out_spatial = self.spatial_att(c0_conv) * image_embeddings
        final_output = final_output + out_spatial
        
        return final_output


class Deoder_fine(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fusion4 = Maskfusion(7, 512)
        self.fusion4 = Maskfusion(16, 512)
        self.conv4 = nn.Sequential(
                                   nn.Conv2d(512, 320, 1),
                                   nn.BatchNorm2d(320),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        # self.fusion3 = Maskfusion(14, 320)
        self.fusion3 = Maskfusion(32, 320)
        self.conv3 = nn.Sequential(
                                   nn.Conv2d(320, 128, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        # self.fusion2 = Maskfusion(28, 128)
        self.fusion2 = Maskfusion(64, 128)
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(128, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        # self.fusion1 = Maskfusion(56, 64)
        self.fusion1 = Maskfusion(128, 64)
        self.conv1 = nn.Sequential(nn.Conv2d(64, 9, 1),
                                  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        

        self.up4 = nn.Sequential(
                                   nn.Conv2d(512, 128, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.up81 = nn.Sequential(
                                   nn.Conv2d(512, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        self.up82 = nn.Sequential(
                                   nn.Conv2d(320, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
    
    
    def forward(self, x1, x2, x3, x4, low_res_masks):
        
        fusion4 =self.fusion4(x4, low_res_masks)

        unet4 = self.conv4(fusion4) + x3
        fusion3 = self.fusion3(unet4, low_res_masks)

        unet3 = self.conv3(fusion3) + x2 + self.up4(fusion4)
        fusion2 = self.fusion2(unet3, low_res_masks)

        unt2 = self.conv2(fusion2) + x1 + self.up81(fusion4) + self.up82(fusion3)
        fusion1 = self.fusion1(unt2, low_res_masks)

        out = self.conv1(fusion1)

        return out, fusion1, fusion2, fusion3, fusion4


class Deoder_predict(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv4 = nn.Sequential(
                                   nn.Conv2d(512, 320, 1),
                                   nn.BatchNorm2d(320),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.conv3 = nn.Sequential(
                                   nn.Conv2d(640, 128, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
       
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(256, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.conv1 = nn.Sequential(nn.Conv2d(128, 9, 1),
                                  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
    
    def forward(self, m1, m2, m3, m4, f1, f2, f3, f4):
        
        add4 = m4 + f4
        add3 = m3 + f3
        add2 = m2 + f2
        add1 = m1 + f1

        add4_c = torch.cat((self.conv4(add4), add3), dim=1)
        add3_c = torch.cat((self.conv3(add4_c), add2), dim=1)
        add2_c = torch.cat((self.conv2(add3_c), add1), dim=1)
        out = self.conv1(add2_c)
        
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
        
        self.pvtv2_v2 = pvt_v2_b2()  # 64, 128, 320, 512
        save_model_v2 = torch.load('/root/autodl-tmp/H-SAM_new/segment_anything/pvt_v2_b2.pth')
        model_dict_v2 = self.pvtv2_v2.state_dict()
        state_dict_v2 = {k: v for k, v in save_model_v2.items() if k in model_dict_v2.keys()}
        model_dict_v2.update(state_dict_v2)
        self.pvtv2_v2.load_state_dict(model_dict_v2)


        self.alpha = nn.Parameter(torch.zeros(1))
        self.Sigmoid = nn.Sigmoid()
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(nn.Conv2d(512, 256, 1), 
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.decoder_msm = Deoder_MSM()
        self.decoder_fine = Deoder_fine()
        self.decoder_predict = Deoder_predict()

        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        self.mask_decoder = mask_decoder

        # self.prompt_encoder2 = prompt_encoder
        # for param in self.prompt_encoder2.parameters():
        #     param.requires_grad = False
        # self.mask_decoder2 = mask_decoder

        self.num_classes = 9
        self.reduce_factor = 4    
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)



    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def one_hot_encoder(self, input_tensor, n_classes):
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, batched_input, multimask_output, image_size, gt=None, mode='train'):
        # if isinstance(batched_input, list):
        #     outputs = self.forward_test(batched_input, multimask_output)
            
        # else:
        #     outputs = self.forward_train(batched_input, multimask_output, image_size
        outputs = self.forward_train(batched_input, multimask_output, image_size, gt=gt, mode=mode)
            
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size,input_points=None, gt=None, mode='train'):

        input_images = self.preprocess(batched_input)
        image_embeddings,low_image_embeddings = self.image_encoder(input_images)

        ####################### first stage ###########
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=input_points, boxes=None, masks=None
        )
        low_res_masks_1, iou_predictions, msk_feat, up_embed = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            gt=gt,
            mode=mode
        )

        ###### 第二阶段 ########################
        low_res_masks_b = self.up4(low_res_masks_1).sum(dim=1, keepdim=True)
        low_res_masks_b = (low_res_masks_b > 0.5).float()
        new_input = batched_input * low_res_masks_b + batched_input
        x1, x2, x3, x4 = self.pvtv2_v2(new_input)

        gate = self.Sigmoid(self.alpha)
        image_embeddings = image_embeddings * gate + (1 - gate) * self.conv(x4)
        
        ########## 解码1 #######
        predict1, m1, m2, m3, m4 = self.decoder_msm(x1, x2, x3, x4)
        predict1_binary = predict1.sum(dim=1, keepdim=True)
        predict1_binary = (predict1_binary > 0.5).float()

        sparse_embeddings2, dense_embeddings2 = self.prompt_encoder(
            points=input_points, boxes=None, masks=predict1_binary
        )
        low_res_masks_2, iou_predictions2, msk_feat2, up_embed2 = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings2,
            dense_prompt_embeddings=dense_embeddings2,
            multimask_output=multimask_output,
            gt=gt,
            mode=mode
        )

        predict2, f1, f2, f3, f4  = self.decoder_fine(x1, x2, x3, x4, low_res_masks_2)
        predict3 = self.decoder_predict(m1, m2, m3, m4, f1, f2, f3, f4)
      
        outputs1 = {
            'low_res_masks_1': low_res_masks_1,
            'low_res_masks_2': low_res_masks_2,
            'binary_predict1': predict1,
            'finally_predict2': predict2,
            'predict3': predict3
        }
        
        return outputs1

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions,_,_ = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                mode='test'
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

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
    
        
        

