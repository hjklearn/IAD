import sys
sys.path.append('/media/lai/data_h/KD_model/')
from torch.autograd import Variable
import argparse
from datetime import datetime
from utils_data.dataloader import BaseSegmentationExperiment
from utils_data.utils import AvgMeter
import numpy as np
import torch.nn as nn
import os
import torch
from importlib import import_module
from torch.nn import functional as F
from segment_anything_Tea import sam_model_registry_tea
from segment_anything_Stu import sam_model_registry
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from distillation_loss import *


def pearson_distance1(x, y, eps=1e-8):
    B = x.shape[0]
    x = x.view(B, -1)
    y = y.view(B, -1)

    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)

    corr_num = (x_centered * y_centered).sum(dim=1)
    corr_den = torch.sqrt((x_centered**2).sum(dim=1) * (y_centered**2).sum(dim=1) + eps)

    corr = corr_num / corr_den
    return (1 - corr).mean()   # Pearson 距离


def resize_to_32x32(x):
    return F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)


def comp_class_pro(label: torch.Tensor, feature: torch.Tensor, num_classes: int):
    
    B, C, H, W = feature.shape
    label = label.squeeze(1)  # (B, H, W)

    prototypes = torch.zeros((B, C), device=feature.device)
    
    for b in range(B):
        mask = (label[b] == 1)  # 前景 mask, shape (H, W)
        if mask.sum() == 0:
            continue
        
        # 提取前景像素对应的特征
        cls_feats = feature[b, :, mask]  # (C, N)
        prototype = cls_feats.mean(dim=1)  # (C,)
        prototypes[b] = prototype
    
    return prototypes  # shape: (B, num_classes, C)


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, net_techer, net_student, optimizer, optimizer_teacher, epoch, best_dice, step, scheduler):
    # net_techer.eval()
    net_student.train()

    loss_record1, loss_record2, loss_record3 = AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        # with torch.no_grad():
        #     tea_out, tea_ima_embed, inter_features = net_techer(images)
        out, stu_local_fea, stu_global = net_student(images)

        # ---- loss function ----
        loss1 = structure_loss(out, gts)
        loss = loss1
        
        # ---- KIS distillation --- 
        # loss1 = structure_loss(out, gts)
        # loss_l2 = nn.MSELoss()(out, tea_out)
        # teacher_eve_cla = comp_class_pro(gts, tea_out, num_classes=2)
        # student_eve_cla = comp_class_pro(gts, out, num_classes=2)
        # proloss_l2 = nn.MSELoss()(student_eve_cla, teacher_eve_cla)
        # loss = 1 * loss1 + o.1 * (0.1 * loss_l2 + 1 * proloss_l2)

        ### KAS distillation ###
        # stu_global_fea = nn.Conv2d(256, 1, kernel_size=1).to(device=stu_global.device)(stu_global)
        # stu_global_fea = resize_to_32x32(stu_global_fea)

        # loss1 = structure_loss(out, gts)

        # loss_l2 = nn.MSELoss()(stu_global_fea, out)
        # stu_global_fea_s = torch.sigmoid(stu_global_fea)
        # outputs1_s = torch.sigmoid(out)
        # inter = (stu_global_fea_s * outputs1_s).sum(axis=(2, 3))
        # unior = (stu_global_fea_s + outputs1_s).sum(axis=(2, 3))
        # self_dice = (2 * inter + 1e-6) / (unior + 1e-6)
        # self_loss2 = 1 - self_dice.mean()

        # neg_out = 1 - outputs1_s
        # neg_f = 1 - stu_global_fea_s
        # neg_out = neg_out + neg_f

        # c_loss = contrastive_loss()(out, stu_global_fea, neg_out)

        # loss = loss1 + 0.01 *(1e-2 * self_loss2 + 1 * c_loss + 1e-2 * loss_l2)


        # ---- backward ----
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step(epoch + epoch/opt.epoch)

        # ---- recording loss ----
        loss_record1.update(loss1.data, opt.batchsize)
        # loss_record2.update(loss2.data, opt.batchsize)

        
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.show()))

    save_path = '{}/{}/'.format(opt.train_save, opt.train_data_type)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        step+=1
        meandice = test(net_student, opt, save_path)
        if meandice > best_dice:
            print('new best dice: ', meandice)
            best_dice = meandice
            torch.save(net_student.state_dict(), save_path + 'best.pth')
            # net_student.save_lora_parameters(save_path + 'best.pth')
            print('[Saving Snapshot:]', save_path + 'best.pth')
        torch.save(net_student.state_dict(), save_path + 'last.pth')
        # net_student.save_lora_parameters(save_path + 'last.pth')
        print('[best_dice: {:.4f}]'.format(best_dice))
    return best_dice


def test(model, opt, log_path):
    log_path = log_path + 'test_log.txt'

    model.eval()

    test_loader = BaseSegmentationExperiment(opt).test_loader

    DSC = 0.0
    JACARD = 0.0
    preds = []
    gts = []
    num1 = len(test_loader)

    for i, pack in enumerate(test_loader, start=1):
        image, gt = pack
        image = Variable(image).cuda()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        
        with torch.no_grad():
            out, _, _ = model(image)
        res = out

        res = res.sigmoid().data.cpu().numpy().squeeze()

        input = np.where(res >= 0.5, 1, 0)
        target = np.where(np.array(gt) >= 0.5, 1, 0)
        
        preds.append(input)
        gts.append(gt)
        
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        union = input_flat + target_flat - intersection
        
        jacard = ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))
        jacard = '{:.4f}'.format(jacard)
        jacard = float(jacard)
        JACARD += jacard
        
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC += dice
        
    print('*****************************************************')
    print('Dice Score: ' + str(DSC/num1))
    print('Jacard Score: ' + str(JACARD/num1))
    print('*****************************************************')

    mean_dsc = DSC / num1
    mean_jacard = JACARD / num1

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{time_now}] Epoch {epoch if epoch is not None else '-'} | DSC: {mean_dsc:.4f} | Jaccard: {mean_jacard:.4f}\n"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(log_line)

    return DSC/num1 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/media/lai/data_h/KD_model/dataset/BioMedicalDataset')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['PolypSegData', 'DSB2018', 'ISIC2018', 'COVID19', 'BUSI'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['DSB2018', 'MonuSeg2018', 'ISIC2018', 'PH2', 'COVID19', 'COVID19_2', 'BUSI', 'STU',
                                                                               'CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB'])
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')

    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=6, help='training batch size')   # batchsize=6
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_save', type=str, default='/media/lai/data_h/KD_model/Polyp_Tool/Pth_fintune_DIST_KAS')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--vit_name', type=str, default='vit_l', help='select one vit model')
    parser.add_argument('--ckpt', type=str, default='/media/lai/data_h/KD_model/segment_anything_Stu/sam_vit_l_0b3195.pth', help='Pretrained checkpoint')
    parser.add_argument('--load_pth', type=str, default='/media/lai/data_h/KD_model/Pth/Pth_DIST_KAS/Synapse_512_pretrain_vit_l_epo300_bs16_lr0.0001_s2345/epoch_299.pth', help='Pretrained checkpoint')
    parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_tea')

    opt = parser.parse_args()
    

    #----------------- tracher model -----------------#
    # sam_tea, img_embedding_size = sam_model_registry_tea[opt.vit_name](image_size=opt.img_size,
    #                                                             num_classes=0,
    #                                                             checkpoint=opt.ckpt, pixel_mean=[0, 0, 0],
    #                                                             pixel_std=[1, 1, 1])
    
    # pkg = import_module(opt.module)
    # net_techer = pkg.LoRA_Sam(sam_tea, opt.rank).cuda()
    # net_techer.load_lora_parameters('/media/lai/data_h/Polyp_Tool/Pth_tea/PolypSegData/best.pth')
    # print('teacher pretrain loading!!!')
    net_techer = None

    # net = PVT_net(sam_model).cuda()

    #----------------- student model -----------------#
    sam, img_embedding_size = sam_model_registry[opt.vit_name](image_size=opt.img_size,
                                                                num_classes=0,
                                                                checkpoint=opt.load_pth, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module('sam_lora_image_encoder_stu')
    net_student = pkg.LoRA_Sam(sam, opt.rank).cuda()
    new_state_dict = torch.load(opt.load_pth)
    # net_student.load_state_dict(new_state_dict)
    model_dict = net_student.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    net_student.load_state_dict(model_dict)
    print('student pretrain loading!!!')

    params = net_student.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    optimizer_teacher = None


    opt.train_dataset_dir = os.path.join(opt.data_path, opt.train_data_type)
    opt.test_dataset_dir = os.path.join(opt.data_path, opt.test_data_type)

    train_loader = BaseSegmentationExperiment(opt).train_loader
    # total_step = len(train_loader)
    
    print("#"*20, "Start Training", "#"*20)
    step = 0
    best_dice = 0.0
    for epoch in range(1, opt.epoch + 1):

        best_dice = train(train_loader, net_techer, net_student, optimizer, optimizer_teacher, epoch, best_dice, step, scheduler)
