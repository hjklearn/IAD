import torch
import torch.nn.functional as F
from torch import nn


def comp_class_pro(label: torch.Tensor, feature: torch.Tensor, num_classes: int):
    """
    Args:
        label: (B, H, W) long tensor,
        feature: (B, C, H, W) float tensor,
        num_classes: int n
        
    Returns:
        prototypes: (B, num_classes, C) tensor
    """
    B, C, H, W = feature.shape
    prototypes = torch.zeros((B, num_classes, C), device=feature.device)
    
    for b in range(B):
        for cls in range(num_classes):
            mask = (label[b] == cls)  # shape: (H, W)
            if mask.sum() == 0:
                continue  
            
            # 提取该类所有像素对应的特征
            cls_feats = feature[b, :, mask]  # shape: (C, N)
            # 计算原型：对 N 个特征向量做均值
            prototype = cls_feats.mean(dim=1)  # shape: (C,)
            prototypes[b, cls] = prototype
    
    return prototypes  # shape: (B, num_classes, C)



class Pixel2TeacherProtoLoss(nn.Module):
    """
    学生像素 → 老师原型 蒸馏损失
    支持 L2 + Cosine 组合
    """
    def __init__(self, num_classes, alpha=1.0, beta=0.1):
        """
        Args:
            num_classes (int): 类别数
            alpha (float): Cosine 损失权重
            beta (float): L2 损失权重
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, stu_feat, tea_feat, label):
        """
        Args:
            stu_feat: (B, C_s, H, W) 学生特征
            tea_feat: (B, C_t, H, W) 老师特征
            label:    (B, H, W)
        """
        B, Cs, H, W = stu_feat.shape
        Ct = tea_feat.shape[1]

        loss = 0.0
        count = 0

        for b in range(B):
            for cls in range(self.num_classes):
                mask = (label[b] == cls)  # (H,W)
                if mask.sum() == 0:
                    continue

                # 学生该类的像素特征: (C_s, N)
                stu_cls_feat = stu_feat[b, :, mask]  # (C_s, N)

                # 老师该类的像素特征 → 原型: (C_t,)
                tea_cls_feat = tea_feat[b, :, mask]  # (C_t, N)
                tea_proto = tea_cls_feat.mean(dim=1, keepdim=True)  # (C_t,1)

                # L2 损失
                l2_loss = torch.norm(stu_cls_feat - tea_proto, p=2, dim=0).mean()

                # Cosine 损失
                stu_norm = F.normalize(stu_cls_feat, dim=0)  # (C,N)
                tea_norm = F.normalize(tea_proto, dim=0)     # (C,1)
                cos_sim = (stu_norm * tea_norm).sum(dim=0)   # (N,)
                cos_loss = (1 - cos_sim).mean()

                # 加权组合
                loss += self.alpha * cos_loss + self.beta * l2_loss
                count += 1

        if count > 0:
            loss = loss / count
        else:
            loss = torch.tensor(0.0, device=stu_feat.device)

        return loss





import torch
from torch import nn
import torch.nn.functional as F


class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None,
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        # print("self.shuffle", self.shuffle_config)
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self, x, gt):
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self, x_student, x_teacher, n_iter):
        interval = self.shuffle_config['interval']
        print(interval, "1")
        B, C, W, H = x_student.shape
        if n_iter % interval == 0:
            print("2")
            idx = torch.randperm(C)
            x_student = x_student[:, idx, :, :].contiguous()
            x_teacher = x_teacher[:, idx, :, :].contiguous()
        print("3")
        return x_student, x_teacher

    def transform(self, x):
        B, C, W, H = x.shape
        loss_type = self.transform_config['loss_type']
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = self.transform_config['group_size']
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def warmup(self, n_iter):
        # print("war")
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return
        else:
            if mode == 'linear':
                self.alpha = self.alpha_0 * (n_iter / warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter / warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self, n_iter):
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear':
                self.alpha = self.alpha_0 * ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

    def forward(self, x_student, x_teacher):
        # print("start kld")
        # if self.warmup_config:
        #     print("warm")
        #     self.warmup(n_iter)
        # if self.earlydecay_config:
        #     print("decay")
        #     self.earlydecay(n_iter)

        # if self.resize_config:
        #     print("resize(")
        #     x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        # if self.shuffle_config:
        #     print("shuffle")
        #     x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        # if self.transform_config:
        #     print("transform")
        #     x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)
        # # print("hhh")

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)
        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        # print("self.alpha", self.alpha)
        loss = self.alpha * loss
        return loss


class OFD(nn.Module):
    '''
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	'''

    def __init__(self, in_channels, out_channels):
        super(OFD, self).__init__()
        self.connector = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        fm_s = self.connector(fm_s)

        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t) ** 2 * mask)

        return loss

    def get_margin(self, fm, eps=1e-6):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask

        margin = masked_fm.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True) + eps)

        return margin


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')

    def _resize(self,x,x_t):
        x = F.interpolate(
            input=x,
            size=x_t.shape[2:],
            mode='bilinear',align_corners=False)
        return x

    def forward(self, x_student, x_teacher):
        # x_student = self._resize(x_student,x_teacher)
        loss_AT = self.mse(x_student.mean(dim=1), x_teacher.mean(dim=1))

        x_student = F.log_softmax(x_student,dim=1)
        x_teacher = F.softmax(x_teacher,dim=1)

        loss_PD = self.KLD(x_student, x_teacher)/(x_student.numel()/x_student.shape[1])
        loss = loss_AT + loss_PD
        # loss = loss_PD
        return loss

class Similarity(nn.Module):
    ##Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author##
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        # print('f_s', f_s.shape)
        bsz = f_s.shape[0]
        # print('bsz', bsz)
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)
        # print('f_s', f_s.shape)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)
        # print('G_t', G_t.shape)
        G_diff = G_t - G_s
        # loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        loss = (G_diff * G_diff).view(-1, 1).sum(0)
        # print('(G_diff * G_diff).view(-1, 1)', (G_diff * G_diff).view(-1, 1).shape)
        return loss

class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

        # self.connector = nn.Sequential(*[
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(out_channels)
        # ])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        avgpool = nn.AvgPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(avgpool(feat_S), avgpool(feat_T))
        return loss

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

a = torch.randn(2, 3, 224, 224)
b = torch.randn(2, 3, 224, 224)
model = Similarity()
result = model(a, b)




import torch.nn as nn


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1., gamma=1.):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss




class contrastive_loss(nn.Module):
    def  __init__(self):
        super(contrastive_loss, self).__init__()

    def mag(self, x):
        x = x ** 2
        s = x.sum()
        s = s ** (1 / 2)
        return s

    def cosine_similarity(self, x, y):
        S = (x * y).sum()
        S = S / (self.mag(x) * self.mag(y))
        return S

    def forward(self, pre, pos, neg, t=1):
        cos = self.cosine_similarity(pos, pre)
        N = torch.exp(cos / t)
        cos = self.cosine_similarity(pre, neg)
        D = torch.exp(cos / t)
        loss = - torch.log(N / (N + D))
        # N, D = convert(N), convert(D)
        return loss



##### VL2Lite KD ########
def pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    """
    计算样本之间的欧氏距离矩阵 d(i,j) = ||x_i - x_j||
    输入: x -> (N, D)
    输出: d -> (N, N)
    """
    return torch.cdist(x, x, p=2)


# 平滑 L1 损失函数
def smooth_l1(delta: torch.Tensor) -> torch.Tensor:
    """
    L1_smooth(δ) = 0.5 * δ^2, if |δ| < 1
                 = |δ| - 0.5, otherwise
    """
    abs_delta = torch.abs(delta)
    mask = (abs_delta < 1.0).float()
    return mask * 0.5 * (delta ** 2) + (1 - mask) * (abs_delta - 0.5)


class VL2Lite(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feat_vlm: torch.Tensor, feat_lw: torch.Tensor) -> torch.Tensor:
        
        feat_vlm = feat_vlm.flatten(1)
        feat_lw = feat_lw.flatten(1)

        assert feat_vlm.size(0) == feat_lw.size(0), "Batch size must match"
        N = feat_lw.size(0)

        # 计算成对距离
        d_vlm = pairwise_distance(feat_vlm)
        d_lw = pairwise_distance(feat_lw)

        # 差值并应用平滑 L1
        diff = d_lw - d_vlm
        loss_mat = smooth_l1(diff)

        # 计算均值损失
        loss = loss_mat.sum() / (N * N)
        return loss

