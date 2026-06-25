from segment_anything_Tea import build_sam, SamPredictor
from segment_anything_Stu import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything_Stu.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        # >>> model = ViT('B_16_imagenet1k')
        # >>> lora_model = LoRA_ViT(model, r=4)
        # >>> preds = lora_model(img)
        # >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0

        self.sam = sam_model


    def forward(self, batched_input, multimask_output=8, gt=None, mode='train'):
        return self.sam(batched_input, multimask_output, gt=gt, mode=mode)


if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    lora_sam = LoRA_Sam(sam, 4)
    lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
