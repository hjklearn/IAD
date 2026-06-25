import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginReLU(nn.Module):
    """
    Margin ReLU σ_m(x) = max(x, m)
    m: channel-wise negative margin (learned or precomputed)
    """
    def __init__(self, margin: torch.Tensor):
        super(MarginReLU, self).__init__()
        # margin shape: [C], one value per channel
        self.register_buffer('margin', margin)

    def forward(self, x):
        # x shape: [N, C, H, W]
        m = self.margin.view(1, -1, 1, 1)
        return torch.max(x, m)


class FeatureRegressor(nn.Module):
    """
    Student transform T_s: 1x1 conv + BN
    Used to align student feature channel with teacher feature channel
    """
    def __init__(self, in_channels, out_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.regressor(x)


def partial_l2_distance(teacher_feat, student_feat):
    """
    Partial L2 distance dp(T, S)
    if S_i <= T_i <= 0: loss = 0
    else: loss = (T_i - S_i)^2
    """
    with torch.no_grad():
        mask = (student_feat <= teacher_feat) & (teacher_feat <= 0)
    diff = (teacher_feat - student_feat) ** 2
    diff = diff * (~mask)  # mask out positions where loss=0
    loss = diff.mean()
    return loss


class DistillationLoss(nn.Module):
    """
    Full distillation loss:
    L_distill = dp( σ_m(F_t), r(F_s) )
    Total loss = L_task + α * L_distill
    """
    def __init__(self, teacher_channels, student_channels, alpha=1.0, margin=None):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        # Initialize margin: if not provided, start small negative
        if margin is None:
            margin = torch.full((teacher_channels,), -1.0)
        self.margin_relu = MarginReLU(margin)
        self.regressor = FeatureRegressor(student_channels, teacher_channels)

    def forward(self, feat_s, feat_t, task_loss):
        # Teacher transform (margin ReLU)
        t_trans = self.margin_relu(feat_t)
        # Student transform (1x1 conv + BN)
        s_trans = self.regressor(feat_s)
        # Distillation loss
        l_distill = partial_l2_distance(t_trans, s_trans)
        # Total loss
        total_loss = task_loss + self.alpha * l_distill
        return total_loss