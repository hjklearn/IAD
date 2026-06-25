# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from icecream import ic

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder_224, MaskDecoder2_224, \
                    MaskDecoder_512, MaskDecoder2_512, PromptEncoder, Sam, \
                    TwoWayTransformer, TwoWayTransformer2, mobilenet_v2, EfficientNet, tiny_vit_5m_224, \
                    MobileNetV3_Large, LeViT_128S



def build_sam_vit_h(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )





def build_sam_vit_l(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


def build_sam_vit_b(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )

build_sam = build_sam_vit_l


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    if image_size == 512:
        sam = Sam(
            image_encoder=EfficientNet.from_pretrained("efficientnet-b1", advprop=True),
            # image_encoder = mobilenet_v2(),
            # image_encoder = MobileNetV3_Large(),
            # image_encoder = tiny_vit_5m_224(pretrained=True),
            # image_encoder = LeViT_128S(fuse=True, pretrained=True),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder_224(
                num_multimask_outputs=num_classes,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            image_size=image_size
        )
    elif image_size == 256:
        sam = Sam(
            image_encoder=EfficientNet.from_pretrained("efficientnet-b1", advprop=True),
            # image_encoder = mobilenet_v2(),
            # image_encoder = MobileNetV3_Large(),
            # image_encoder = tiny_vit_5m_224(pretrained=True),
            # image_encoder = LeViT_128S(fuse=True, pretrained=True),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder_224(
                num_multimask_outputs=num_classes,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            image_size=image_size
        )
    elif image_size == 224:
        sam = Sam(
            image_encoder=EfficientNet.from_pretrained("efficientnet-b1", advprop=True),
            # image_encoder = mobilenet_v2(),
            # image_encoder = MobileNetV3_Large(),
            # image_encoder = tiny_vit_5m_224(pretrained=True),
            # image_encoder = LeViT_128S(fuse=True, pretrained=True),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder_224(
                num_multimask_outputs=num_classes,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            image_size=image_size
        )
    # sam.eval()
    sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

            # 过滤掉 except_keys 中的参数
            except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not any(ex_key in k for ex_key in except_keys)
            }

            load_result = sam.load_state_dict(filtered_state_dict, strict=False)
            print('loading mask_decoder pretrain!!!')

            # loaded_keys = set(state_dict.keys()) - set(load_result.unexpected_keys)
            # print("Successfully Loaded Keys:")
            # for key in loaded_keys:
            #     print("  ", key)

    return sam, image_embedding_size


