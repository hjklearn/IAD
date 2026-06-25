import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic
from PIL import Image
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


def create_feature_align_modules(t_feats, s_feats):
    
    assert len(t_feats) == len(s_feats), "教师和学生特征数量必须一致"
    
    align_modules = nn.ModuleList()
    
    for t_feat, s_feat in zip(t_feats, s_feats):
        _, C_t, H_t, W_t = t_feat.shape
        _, C_s, H_s, W_s = s_feat.shape
        
        align_module = nn.Sequential()
        
        # 1. 通道对齐：1x1 卷积（仅当通道数不同时添加）
        if C_s != C_t:
            align_module.add_module(
                'conv1x1', 
                nn.Conv2d(C_s, C_t, kernel_size=1, bias=False)
            )
        
        # 2. 空间对齐：插值到教师特征的空间尺寸（仅当 H/W 不同时添加）
        if (H_s, W_s) != (H_t, W_t):
            align_module.add_module(
                'interpolate',
                nn.Upsample(size=(H_t, W_t), mode='bilinear', align_corners=False)
            )
        
        align_modules.append(align_module)
    
    return align_modules



def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]


def calc_loss(tea_out, outputs, low_res_label_batch, label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    loss_ce = ce_loss(outputs, low_res_label_batch[:].long())
    loss_dice = dice_loss(outputs, low_res_label_batch, softmax=True)
    loss1 = ((1 - dice_weight) * loss_ce + dice_weight * loss_dice)

    loss_l2 = nn.MSELoss()(outputs, tea_out)

    weights = torch.ones(9, device=outputs.device)
    weights[8] = 4.0
    weights[4] = 2.0
    weights = weights.view(1, 9, 1)
    teacher_eve_cla = comp_class_pro(low_res_label_batch, tea_out, num_classes=9)
    student_eve_cla = comp_class_pro(low_res_label_batch, outputs, num_classes=9)
    mse = F.mse_loss(student_eve_cla, teacher_eve_cla, reduction='none')
    proloss_l2 = (mse * weights).mean()

    loss2 = loss_l2 + proloss_l2
    
    return loss_ce, loss_dice, loss1, loss2



def OFD_loss(source, target):

    # target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


def resize_to_32x32(x):
    return F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

def trainer_synapse(args, model, snapshot_path, multimask_output, net_tea, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split=args.split,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
                                   ]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        # model = model.module
    model.train()
    net_tea.eval()
    
    # 不统计层的参数
    excluded_keywords = ['head', 'norm_head', '_fc', '_dropout', '_avg_pooling']

    def is_excluded(name):
        return any(ex in name for ex in excluded_keywords)

    model_total_params = sum(
        p.numel() for name, p in model.named_parameters()
        if not is_excluded(name)
    )

    model_grad_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and not is_excluded(name)
    )

    print(f'model_grad_params (exclude head): {model_grad_params / 1e6:.2f} M')
    print(f'model_total_params (exclude head): {model_total_params / 1e6:.2f} M')

    ce_loss = CrossEntropyLoss()
    cos_loss = CosineEmbeddingLoss()
    l1_loss = nn.L1Loss()
    dice_loss = DiceLoss(num_classes + 1)
    BCE = torch.nn.BCEWithLogitsLoss()
    # dice_loss = Focal_loss(num_classes=num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
   
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, low_image, label_batch = sampled_batch['image_512'], sampled_batch['low_image'], sampled_batch['label_512']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            binary_label_batch = sampled_batch['low_binary_label']
            image_batch, low_image, label_batch = image_batch.cuda(), low_image.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            binary_label_batch = binary_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                tea_out, tea_ima_embed, inter_features = net_tea(image_batch, multimask_output)

            outputs1, stu_local_fea, stu_global = model(image_batch, multimask_output)

            # loss_ce1, loss_dice1, loss1, loss2 = calc_loss(tea_out, outputs1, low_res_label_batch, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)

            ### KAS distillation ###
            # stu_global_fea = nn.Conv2d(256, 9, kernel_size=1).to(device=stu_global.device)(stu_global)
            # stu_global_fea = resize_to_32x32(stu_global_fea)
            # loss_ce1 = ce_loss(outputs1, label_batch[:].long())
            # loss_dice1 = dice_loss(outputs1, label_batch, softmax=True)
            # loss1 = ((1 - 0.9) * loss_ce1 + 0.9 * loss_dice1)

            # loss_l2 = nn.MSELoss()(stu_global_fea, outputs1)
            # stu_global_fea_s = torch.softmax(stu_global_fea, dim=1)
            # outputs1_s = torch.softmax(outputs1, dim=1)
            # inter = (stu_global_fea_s * outputs1_s).sum(axis=(2, 3))
            # unior = (stu_global_fea_s + outputs1_s).sum(axis=(2, 3))
            # self_dice = (2 * inter + 1e-6) / (unior + 1e-6)
            # self_loss2 = 1 - self_dice.mean()

            # neg_out = 1 - outputs1_s
            # neg_f = 1 - stu_global_fea_s
            # neg_out = nn.Conv2d(9, 9, kernel_size=3, padding=1).to(device=stu_global.device)(neg_out + neg_f)
            # c_loss = contrastive_loss()(outputs1, stu_global_fea, neg_out)
            # loss = loss1 + 0.1 * (self_loss2 + c_loss + loss_l2)
   

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Peak GPU Memory: {peak_mem:.2f} MB")

            torch.cuda.synchronize()
            end = time.time()
            print(f"Time per iteration: {(end - start)*1000:.2f} ms")

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            # writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
            # writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
            # writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)
            # writer.add_scalar('info/loss_self2', loss_self, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            # logging.info('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item()))
            # logging.info('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f, loss_ce2: %f, loss_dice2: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item(), loss_ce2.item(), loss_dice2.item()))


        save_interval = 25 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                torch.save(model.state_dict(), save_mode_path)
                # model.save_lora_parameters(save_mode_path)
            except:
                # model.module.save_lora_parameters(save_mode_path)
                torch.save(model.module.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                torch.save(model.state_dict(), save_mode_path)
                # model.save_lora_parameters(save_mode_path)
            except:
                # model.module.save_lora_parameters(save_mode_path)
                torch.save(model.module.state_dict(), save_mode_path)

            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
    
