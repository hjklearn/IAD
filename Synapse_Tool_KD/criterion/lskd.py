# import torch
# import torch.nn.functional as F

# def z_score_standardize_2d(logits, tau=1.0, eps=1e-6):
#     """
#     对分割任务 logits 做样本级 Z-score 标准化
#     logits: [B, C, H, W]
#     tau: 基础温度
#     """
#     mean = logits.mean(dim=1, keepdim=True)               # [B,1,H,W]
#     std = logits.std(dim=1, keepdim=True) + eps           # [B,1,H,W]
#     z = (logits - mean) / (std * tau)
#     return z


# def logit_standardization_kd_loss(student_logits, teacher_logits, tau=1.0):
#     """
#     Logit Standardization KD (LS-KD) 损失 (仅KD部分)
#     Args:
#         student_logits: [B, C, H, W]
#         teacher_logits: [B, C, H, W]
#         tau: 基础温度 (Base Temperature)
#     Returns:
#         kd_loss: 蒸馏损失值 (scalar tensor)
#     """
#     # Step 1: 对每个样本每个像素位置做Z-score标准化
#     z_s = z_score_standardize_2d(student_logits, tau)
#     z_t = z_score_standardize_2d(teacher_logits, tau)

#     # Step 2: softmax 计算概率分布
#     q_t = F.softmax(z_t, dim=1)          # teacher 概率
#     q_s_log = F.log_softmax(z_s, dim=1)  # student log概率

#     # Step 3: 计算KL散度蒸馏损失
#     kd_loss = F.kl_div(q_s_log, q_t, reduction='batchmean') * (tau ** 2)

#     return kd_loss



import torch
import torch.nn.functional as F

def z_score_standardize_2d(logits, tau=2.0, eps=1e-3):
    """
    对分割任务 logits 做样本级 Z-score 标准化
    logits: [B, C, H, W]
    tau: 基础温度
    eps: 防止除零
    """
    mean = logits.mean(dim=1, keepdim=True)               # [B,1,H,W]
    std = logits.std(dim=1, keepdim=True) + eps           # [B,1,H,W]
    z = (logits - mean) / (std * tau)
    return z

def logit_standardization_kd_loss(student_logits, teacher_logits, tau=2.0):
    """
    Logit Standardization KD (LS-KD) 损失
    适用于分割任务 (2D logits)
    """
    # 1. Z-score 标准化
    z_s = z_score_standardize_2d(student_logits, tau)
    z_t = z_score_standardize_2d(teacher_logits, tau)

    # 2. softmax
    q_t = F.softmax(z_t, dim=1).clamp(min=1e-8, max=1.0)
    q_s_log = F.log_softmax(z_s, dim=1)

    # 3. KL 散度按像素平均，而不是 batchmean
    kd_pixel = F.kl_div(q_s_log, q_t, reduction='none')   # [B,C,H,W]
    kd_loss = kd_pixel.sum(dim=1).mean() * (tau ** 2)     # 先按类求和，再对所有像素平均

    return kd_loss

