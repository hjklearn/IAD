import sys
sys.path.append('---path')

import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
from importlib import import_module

import numpy as np
from tqdm import tqdm
from medpy.metric import dc
from scipy.ndimage import zoom

from ACDC_Tool.utils import DiceLoss
from datasets.dataset_ACDC import ACDCdataset, RandomGenerator
from test_ACDC import inference
from segment_anything_Stu import sam_model_registry
from segment_anything_Tea import sam_model_registry_tea
from distillation_loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--encoder', default='PVT', help='Name of encoder: PVT or MERIT')
parser.add_argument('--vit_name', type=str,
                    default='vit_l', help='select one vit model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder_tea')
parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA adaptation')
parser.add_argument('--skip_aggregation', default='additive', help='Type of skip-aggregation: additive or concatenation')                        
parser.add_argument("--batch_size", default=6, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument("--save_path", default="/media/lai/data_h/ACDC_Tool/")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="/media/lai/data_h/datas/ACDC/lists_ACDC/")
parser.add_argument("--root_dir", default="/media/lai/data_h/datas/ACDC/")
parser.add_argument("--volume_path", default="/media/lai/data_h/datas/ACDC/test/")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=3)
parser.add_argument('--test_save_dir', default='pre', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--ckpt', type=str, default='/media/lai/data_h/segment_anything_Tea/sam_vit_l_0b3195.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--seed', type=int,
                    default=2345, help='random seed')                   
args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

snapshot_path = '/media/lai/data_h/ACDC_Tool/Pth_CrossKD/'

    
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir,'bs_6')
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)


# #----------------- teacher model -----------------#
sam_tea, img_embedding_size = sam_model_registry_tea[args.vit_name](image_size=args.img_size,
                                                            num_classes=args.num_classes,
                                                            checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                            pixel_std=[1, 1, 1])
pkg = import_module(args.module)
net_tea = pkg.LoRA_Sam(sam_tea, args.rank).cuda()
net_tea.load_lora_parameters('/media/lai/data_h/ACDC_Tool/Pth_tea/best.pth')
print('teacher pretrain loading!!!')


#----------------- student model -----------------#
sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                            num_classes=args.num_classes,
                                                            checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                            pixel_std=[1, 1, 1])

pkg = import_module('sam_lora_image_encoder_stu')
net = pkg.LoRA_Sam(sam, args.rank).cuda()
# new_state_dict = torch.load('/media/lai/data_h/ACDC_Tool/Pth_stu/pth.pth')
# net.load_state_dict(new_state_dict)
# print('student pretrain loading!!!')


train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val=ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader=DataLoader(db_val, batch_size=1, shuffle=False)
db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)

net = net.cuda()
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(4)

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.80
Best_test_dcs = 0.80
logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(Train_loader)
print('max_iterations', len(Train_loader))
base_lr = args.lr

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1)



def resize_to_32x32(x):
    return F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)



def val():
    logging.info("Validation ===>")
    dc_sum=0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.squeeze(0).cpu().detach().numpy(), val_label_batch.squeeze(0).cpu().detach().numpy()
        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3)
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()
        
        P_tea, _, _ = net(val_image_batch)
        val_outputs = P_tea
        
        val_outputs = torch.softmax(val_outputs, dim=1)
        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()

        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs
        dc_sum+=dc(val_outputs,val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    return performance

    
for epoch in iterator:
    net.train()
    net_tea.eval()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        with torch.no_grad():
            tea_out, tea_ima_embed, inter_features = net_tea(image_batch)

        P_tea, stu_local_fea, stu_global = net(image_batch)
    
        # loss_ce = ce_loss(P_tea, label_batch[:].long())
        # loss_dice = dice_loss(P_tea, label_batch, softmax=True)
        # loss1 = (0.1 * loss_ce + 0.9 * loss_dice)
        # loss = loss1 

        # ---- KIS distillation ---
        # log_sigma1 = nn.Parameter(torch.zeros(1)).to(device=stu_global.device)
        # log_sigma2 = nn.Parameter(torch.zeros(1)).to(device=stu_global.device) 
        # loss_ce = ce_loss(P_tea, label_batch[:].long())
        # loss_dice = dice_loss(P_tea, label_batch, softmax=True)
        # loss1 = (0.1 * loss_ce + 0.9 * loss_dice)
        # loss_l2 = nn.MSELoss()(P_tea, tea_out)
        # teacher_eve_cla = comp_class_pro(label_batch, tea_out, num_classes=4)
        # student_eve_cla = comp_class_pro(label_batch, P_tea, num_classes=4)
        # proloss_l2 = nn.MSELoss()(teacher_eve_cla, student_eve_cla)
        # loss = loss1 + log_sigma1 * loss_l2 + log_sigma2 * proloss_l2

        ### KAS distillation ###
        # log_sigma1 = nn.Parameter(torch.zeros(1)).to(device=stu_global.device)
        # log_sigma2 = nn.Parameter(torch.zeros(1)).to(device=stu_global.device) 
        # log_sigma3 = nn.Parameter(torch.zeros(1)).to(device=stu_global.device)
        # stu_global_fea = nn.Conv2d(256, 4, kernel_size=1).to(device=stu_global.device)(stu_global)
        # stu_global_fea = resize_to_32x32(stu_global_fea)
        # loss_ce1 = ce_loss(P_tea, label_batch[:].long())
        # loss_dice1 = dice_loss(P_tea, label_batch, softmax=True)
        # loss1 = ((1 - 0.9) * loss_ce1 + 0.9 * loss_dice1)
        # loss_l2 = nn.MSELoss()(stu_global_fea, P_tea)
        # stu_global_fea_s = torch.softmax(stu_global_fea, dim=1)
        # outputs1_s = torch.softmax(P_tea, dim=1)
        # inter = (stu_global_fea_s * outputs1_s).sum(axis=(2, 3))
        # unior = (stu_global_fea_s + outputs1_s).sum(axis=(2, 3))
        # self_dice = (2 * inter + 1e-6) / (unior + 1e-6)
        # self_loss2 = 1 - self_dice.mean()
        # neg_out = 1 - outputs1_s
        # neg_f = 1 - stu_global_fea_s
        # neg_out = nn.Conv2d(4, 4, kernel_size=3, padding=1).to(device=stu_global.device)(neg_out + neg_f)
        # c_loss = contrastive_loss()(P_tea, stu_global_fea, neg_out)
        # loss = loss1 + log_sigma1 * self_loss2 + log_sigma2 * c_loss + log_sigma3 * loss_l2


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if iter_num%50 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss/len(train_dataset))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    
    save_model_path = os.path.join(snapshot_path, 'last.pth')
    try:
        torch.save(net.state_dict(), save_model_path)
 
    except:
 
        torch.save(net.state_dict(), save_model_path)
    logging.info("save model to {}".format(save_model_path))
    
    avg_dcs = val()
  
    if avg_dcs >= Best_dcs:
        save_model_path = os.path.join(snapshot_path, 'best.pth')
        torch.save(net.state_dict(), save_model_path)

        try:
            torch.save(net.state_dict(), save_model_path)
 
        except:

            torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        Best_dcs = avg_dcs
        
        avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
        print("test avg_dsc: %f" % (avg_test_dcs))
        logging.info("test avg_dsc: %f" % (avg_test_dcs))
        Test_Accuracy.append(avg_test_dcs)
        if(Best_test_dcs <= avg_test_dcs):
            Best_test_dcs = avg_test_dcs
            save_model_path = os.path.join(snapshot_path, 'test_best.pth')
            torch.save(net.state_dict(), save_model_path)

            try:
                torch.save(net.state_dict(), save_model_path)
 
            except:
 
                torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            print("save model to {}".format(save_model_path))
        
    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path,  'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        try:
            torch.save(net.state_dict(), save_model_path)
        except:
 
            torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        iterator.close()
        break
