import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CrossKDLoss(nn.Module):
    """
    Cross-head Knowledge Distillation loss.

    Args:
        teacher_head_layers: nn.ModuleList of teacher head conv layers [C1,...,Cn]
                             (should be the SAME ordering as in teacher)
        temp_cls: temperature for classification KD
        cls_loss: 'kl' or 'ce' for classification distillation (default 'kl' uses KL on soft logits)
        reg_loss: 'smooth_l1' or 'l1' for regression distillation
        reduction: 'mean' or 'sum'
    """
    def __init__(self,
                 teacher_head_layers: nn.ModuleList,
                 temp_cls: float = 1.0,
                 cls_loss: str = 'kl',
                 reg_loss: str = 'smooth_l1',
                 reduction: str = 'mean'):
        super().__init__()
        assert isinstance(teacher_head_layers, (list, nn.ModuleList)), "teacher_head_layers must be ModuleList or list"
        # keep a reference but do not register new params
        self.teacher_head_layers = teacher_head_layers
        self.temp_cls = float(temp_cls)
        assert cls_loss in ('kl', 'ce')
        assert reg_loss in ('smooth_l1', 'l1')
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        assert reduction in ('mean', 'sum')
        self.reduction = reduction

        # Freeze teacher head parameters (no updates), but keep them in graph so gradients can flow to inputs.
        for p in self.teacher_head_layers.parameters():
            p.requires_grad = False

    def _forward_through_teacher_from(self, feat: torch.Tensor, start_layer_idx: int):
        """
        Feed `feat` through teacher_head_layers[start_layer_idx:] sequentially to produce cross-head prediction.
        feat: tensor [B, C_in, H, W] (fits into C_{start_layer})
        start_layer_idx: 0-based index in teacher_head_layers; we will run from start_layer_idx to end-1.
        """
        x = feat
        for layer in list(self.teacher_head_layers)[start_layer_idx:]:
            x = layer(x)
        return x

    def _cls_kd_loss(self, p_hat: torch.Tensor, p_teacher: torch.Tensor):
        """
        p_hat, p_teacher: [B, num_classes, H, W]
        compute temperatured KL (student vs teacher), averaged over batch & spatial positions.
        """
        T = self.temp_cls
        # reshape to [B * H * W, C]
        B, C, H, W = p_hat.shape
        p_hat_flat = p_hat.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
        p_t_flat   = p_teacher.permute(0, 2, 3, 1).reshape(-1, C)

        if self.cls_loss == 'kl':
            # KLDiv expects log-probs for input
            log_p_hat = F.log_softmax(p_hat_flat / T, dim=1)
            p_t_soft  = F.softmax(p_t_flat / T, dim=1)
            # use batchmean to be consistent
            kl = F.kl_div(log_p_hat, p_t_soft, reduction='batchmean') * (T * T)
            return kl
        else:
            # 'ce' : use cross-entropy treating teacher soft targets as probabilities (soft cross-entropy)
            p_t_soft = F.softmax(p_t_flat / T, dim=1)
            # implement soft cross entropy: -sum p_t * log p_hat
            log_p_hat = F.log_softmax(p_hat_flat / T, dim=1)
            loss = -(p_t_soft * log_p_hat).sum(dim=1)  # [BHW]
            if self.reduction == 'mean':
                return loss.mean() * (T * T)
            else:
                return loss.sum() * (T * T)

    def _reg_loss(self, p_hat: torch.Tensor, p_teacher: torch.Tensor):
        """
        p_hat, p_teacher: [B, R, H, W] (e.g., R=4 for bbox regression)
        compute per-element smooth_l1 or l1 and average over spatial & channels
        """
        diff = p_hat - p_teacher
        if self.reg_loss == 'smooth_l1':
            loss = F.smooth_l1_loss(p_hat, p_teacher, reduction='none')
        else:
            loss = diff.abs()
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

    def forward(self,
                f_s_list: List[torch.Tensor],
                p_teacher: torch.Tensor,
                head_task: str = 'cls'):
        """
        Args:
            f_s_list: list of student intermediate features [f1_s, f2_s, ..., f_{n-1}_s],
                      where f_i corresponds to the input of C_{i+1} in teacher head.
                      Each tensor shape: [B, C_i, H, W]
            p_teacher: teacher original predictions, shape [B, C_out, H, W]
            head_task: 'cls' or 'reg' (choose loss type)
        Returns:
            scalar cross-head KD loss (averaged over provided features)
        """
        assert head_task in ('cls', 'reg')
        n_feat = len(f_s_list)
        if n_feat == 0:
            return torch.tensor(0.0, device=p_teacher.device, requires_grad=True)

        losses = []
        # For each student's intermediate feature fi_s, forward through teacher head from layer (i+1)
        # assume ordering: teacher_head_layers = [C1, C2, ..., Cn], and f_s_list[i] maps to input of C_{i+1}
        for i, f_s in enumerate(f_s_list):
            start_idx = i + 1  # start at C_{i+1}
            # forward through teacher head layers from start_idx to end
            # IMPORTANT: teacher params are frozen but we do NOT use torch.no_grad(), so gradients can flow back to f_s.
            p_hat_s = self._forward_through_teacher_from(f_s, start_idx)

            # now compute task specific loss between p_hat_s and p_teacher
            if head_task == 'cls':
                loss_i = self._cls_kd_loss(p_hat_s, p_teacher)
            else:
                loss_i = self._reg_loss(p_hat_s, p_teacher)
            losses.append(loss_i)

        # average across all cross-head predictions
        loss_all = sum(losses) / len(losses)
        return loss_all
