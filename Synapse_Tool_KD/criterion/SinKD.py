import torch
import torch.nn.functional as F

# def sinkhorn_loss(t, s, lam=10.0, max_iter=50, p=2):
#     """
#     Sinkhorn蒸馏损失 (L_SD)
#     Args:
#         t: Teacher输出 [B, D]
#         s: Student输出 [B, D]
#         lam: 平滑系数 λ (越大 -> 越接近OT距离)
#         max_iter: Sinkhorn归一化迭代次数
#         p: 距离范数类型 (默认L2)
#     Returns:
#         L_SD: Sinkhorn蒸馏损失 (标量)
#     """
#     if t.dim() > 2:
#         t = t.flatten(1)  # [B, D]，把除了batch维外的全部展开
#         s = s.flatten(1)
#     B, D = t.shape

#     # === 1. 构建距离矩阵 D_ij ===
#     # [B, D] → [B, 1, D] 和 [1, B, D]
#     t_expand = t.unsqueeze(1)   # [B, 1, D]
#     s_expand = s.unsqueeze(0)   # [1, B, D]
#     # 距离矩阵 D_{ij} = ||t_i - s_j||_p
#     D_mat = torch.norm(t_expand - s_expand, p=p, dim=2)  # [B, B]

#     # === 2. 构建初始核矩阵 K = exp(-λ * D) ===
#     K = torch.exp(-lam * D_mat)
#     K = K / (K.sum() + 1e-8)  # 防止数值爆炸

#     # === 3. Sinkhorn迭代: 行列归一化 ===
#     for _ in range(max_iter):
#         K = K / (K.sum(dim=1, keepdim=True) + 1e-8)  # Normalize rows
#         K = K / (K.sum(dim=0, keepdim=True) + 1e-8)  # Normalize cols

#     # === 4. 计算Sinkhorn损失 L_SD = <K, D> ===
#     L_SD = torch.sum(K * D_mat)

#     return L_SD

def sinkhorn_loss(t, s, lam=10.0, max_iter=50, p=2):
    with torch.no_grad():  # 🚀 teacher部分不反传梯度
        if t.dim() > 2:
            t = t.flatten(1)
            s = s.flatten(1)
        B, D = t.shape

        # 距离矩阵
        D_mat = torch.cdist(t, s, p=p)  # ✅ 高效计算 [B, B]，内部自动优化
        K = torch.exp(-lam * D_mat)
        K = K / (K.sum() + 1e-8)

        for _ in range(max_iter):
            K = K / (K.sum(dim=1, keepdim=True) + 1e-8)
            K = K / (K.sum(dim=0, keepdim=True) + 1e-8)

        L_SD = torch.sum(K * D_mat)

    return L_SD


def reshape_feat(x):
    # x: [1, 9, 512, 512]
    x = x.permute(0, 2, 3, 1)   # [1, 512, 512, 9]
    x = x.reshape(-1, 9)        # [512*512, 9]
    return x


def pixel_info_nce(pre, pos, neg, temperature=0.1):
    """
    pre, pos, neg: [1, 9, 512, 512]
    """

    q = reshape_feat(pre)
    p = reshape_feat(pos)
    n = reshape_feat(neg)

    # L2 normalize
    q = F.normalize(q, dim=1)
    p = F.normalize(p, dim=1)
    n = F.normalize(n, dim=1)

    # cosine similarity
    pos_sim = torch.sum(q * p, dim=1) / temperature   # [N]
    neg_sim = torch.sum(q * n, dim=1) / temperature   # [N]

    loss = -torch.log(
        torch.exp(pos_sim) /
        (torch.exp(pos_sim) + torch.exp(neg_sim))
    )

    return loss.mean()
