import torch
import torch.nn.functional as F
from torch import nn


def _normalize(x, if_normalize=True, eps=1e-6):
    """Normalize tensor along channel dimension."""
    if if_normalize:
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + eps)
    return x


class CATLoss(nn.Module):
    """
    Class Activation Transfer (CAT) Loss
    用于特征蒸馏，将学生与教师模型的高层特征进行空间对齐和相似性约束。
    """

    def __init__(
        self,
        cam_resolution: int = 2,
        if_normalize: bool = True,
        loss_weight: float = 400.0,
    ):
        """
        Args:
            cam_resolution: CAM下采样分辨率（默认 2x2）
            if_normalize: 是否归一化特征
            loss_weight: 损失权重系数
        """
        super().__init__()
        self.cam_resolution = cam_resolution
        self.if_normalize = if_normalize
        self.loss_weight = loss_weight

    def forward(self, feat_student, feat_teacher):
        """
        Args:
            feat_student: 学生模型的特征 (B, C, H, W)
            feat_teacher: 教师模型的特征 (B, C, H, W)
        """
        # 下采样到固定CAM尺寸
        stu_cam = F.adaptive_avg_pool2d(feat_student, (self.cam_resolution, self.cam_resolution))
        tea_cam = F.adaptive_avg_pool2d(feat_teacher, (self.cam_resolution, self.cam_resolution))

        # 特征归一化
        stu_cam = _normalize(stu_cam, self.if_normalize)
        tea_cam = _normalize(tea_cam, self.if_normalize)

        # MSE蒸馏损失
        loss = F.mse_loss(stu_cam, tea_cam)

        return self.loss_weight * loss
