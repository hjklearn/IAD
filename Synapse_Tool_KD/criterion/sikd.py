import torch
import torch.nn as nn

class ShapeIntensityKD(nn.Module):
    """
    Shape-Intensity Knowledge Distillation Loss
    只计算 KD 部分: L_kd = MSE(f_student, f_teacher)
    """
    def __init__(self):
        super(ShapeIntensityKD, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: 学生网络特征 [B, C, H, W]
            teacher_feat: 教师网络特征 [B, C, H, W]
        Returns:
            kd_loss: 蒸馏损失值 (float)
        """
        # 教师特征不参与反传
        teacher_feat = teacher_feat.detach()

        # 计算特征层之间的均方误差
        kd_loss = self.mse(student_feat, teacher_feat)
        return kd_loss
