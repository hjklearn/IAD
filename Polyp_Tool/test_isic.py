import sys
sys.path.append('/media/lai/data_h/KD_model')
import numpy as np
import os, argparse
from torch.autograd import Variable
import imageio
from importlib import import_module
# from sam_lora_image_encoder import *
from utils_data.dataloader import BaseSegmentationExperiment
from Stage_SAM_model import *
from matplotlib import pyplot as plt
from segment_anything_Tea import sam_model_registry_tea
from segment_anything_Stu import sam_model_registry
from medpy import metric
from datetime import datetime
from medpy.metric.binary import hd95


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0
    

def calculate_multiclass_metric(pred, gt, num_classes=9):
    dice_list = []
    hd95_list = []
    iou_list = []
    
    for cls in range(num_classes):
        pred_cls = pred[cls, :, :]  # [H, W]
        gt_cls = gt[cls, :, :]      # [H, W]
        
        pred_cls = (pred_cls > 0).astype(np.uint8)
        gt_cls = (gt_cls > 0).astype(np.uint8)
        
        if pred_cls.sum() > 0 and gt_cls.sum() > 0:
            # Dice
            dice = metric.binary.dc(pred_cls, gt_cls)
            # HD95
            hd95 = metric.binary.hd95(pred_cls, gt_cls)
            # IoU
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            iou = intersection / (union + 1e-6)
            
        elif pred_cls.sum() > 0 and gt_cls.sum() == 0:
            dice, hd95, iou = 1, 0, 0  # IoU=0 因为 gt 没有该类
        
        else:
            dice, hd95, iou = 0, 0, 0
        
        dice_list.append(dice)
        hd95_list.append(hd95)
        iou_list.append(iou)
    
    return np.mean(dice_list), np.mean(hd95_list), np.mean(iou_list)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/lai/data_h/KD_model/dataset/BioMedicalDataset')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['PolypSegData', 'DSB2018', 'ISIC2018', 'COVID19', 'BUSI'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['DSB2018', 'MonuSeg2018', 'ISIC2018', 'PH2', 'COVID19', 'COVID19_2', 'BUSI', 'STU',
                                                                               'CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB'])
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--save_path', type=str, default='/media/lai/data_h/KD_model/Polyp_Tool/Pth_CRDKD/ETIS-LaribPolypDB/', help='path to save inference segmentation')

    parser.add_argument('--ckpt', type=str, default='/media/lai/data_h/KD_model/segment_anything_Tea/sam_vit_l_0b3195.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='/media/lai/data_h/KD_model/Polyp_Tool/Pth_CRDKD/PolypSegData/best.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_l', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_tea')

    opt = parser.parse_args()

    #----------------- tracher model -----------------#
    # sam, img_embedding_size = sam_model_registry_tea[opt.vit_name](image_size=opt.img_size,
    #                                                             num_classes=8,
    #                                                             checkpoint=opt.ckpt, pixel_mean=[0, 0, 0],
    #                                                             pixel_std=[1, 1, 1])
    
    # pkg = import_module(opt.module)
    # model = pkg.LoRA_Sam(sam, opt.rank).cuda()
    # model = PVT_net().cuda()

    #----------------- student model -----------------#
    sam, img_embedding_size = sam_model_registry[opt.vit_name](image_size=opt.img_size,
                                                                num_classes=0,
                                                                checkpoint=opt.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module('sam_lora_image_encoder_stu')
    net_student = pkg.LoRA_Sam(sam, opt.rank).cuda()

    net_student.load_state_dict(torch.load(opt.lora_ckpt))
    model = net_student
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    # print('evaluating model: ', opt.ckpt_path)

    opt.train_dataset_dir = os.path.join(opt.data_path, opt.train_data_type)
    opt.test_dataset_dir = os.path.join(opt.data_path, opt.test_data_type)

    test_loader = BaseSegmentationExperiment(opt).test_loader

    DSC = 0.0
    JACARD = 0.0
    HD95 = 0.0
    preds = []
    gts = []
    num1 = len(test_loader)

    save_path_txt = "/media/lai/data_h/ETIS-LaribPolypDB/p_result.txt"
    open(save_path_txt, "w").close()

    for i, pack in enumerate(test_loader, start=1):
        image, gt = pack
        image = Variable(image).cuda()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        with torch.no_grad():
            outputs, _, _ = model(image)
        res = outputs

        res = res.sigmoid().data.cpu().numpy().squeeze()

        if opt.save_path is not None:
            # 转换为 0-255 的 uint8 格式
            sample_res = res
            sample_gt = gt.squeeze()
            sample_res = (sample_res * 255).astype(np.uint8)
            sample_gt = (sample_gt * 255).astype(np.uint8)
            imageio.imwrite(opt.save_path+'/'+str(i)+'_pred.jpg', sample_res)
            imageio.imwrite(opt.save_path+'/'+str(i)+'_gt.jpg', sample_gt)

            #### 可视化image######
            image = image.cpu().detach().squeeze().permute(1, 2, 0).numpy()  # 转为 NumPy
            image = (image - image.min()) / (image.max() - image.min())
            save_path = opt.save_path + f'/{str(i)}_image.png'
            plt.imsave(save_path, arr=image, cmap='gray')

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

        with open(save_path_txt, "a") as f:
            f.write(f"{dice}\n")
            f.write(f"{jacard}\n")
            
        print(f"✅ 结果已保存到 {save_path_txt}")
    
        
    print('*****************************************************')
    print('Dice Score: ' + str(DSC/num1))
    print('Jacard Score: ' + str(JACARD/num1))
    print('HD95 Score: ' + str(HD95/num1))
    print('*****************************************************')

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{time_now}] Dice Score {DSC/num1 if DSC/num1 is not None else '-'} | DSC Score: {DSC/num1:.4f} | HD95: {HD95/num1:.4f}\n"
    parent_path = os.path.dirname(opt.save_path)
    sav_pth = os.path.join(parent_path, "result.txt")  # 拼接文件名
    with open(sav_pth, "a") as f:
        f.write(log_line)
