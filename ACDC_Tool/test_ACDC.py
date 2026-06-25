import sys
sys.path.append('path')
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from T_SNet import *
from sklearn.preprocessing import StandardScaler
from utils import test_single_volume
from datasets.dataset_ACDC import ACDCdataset
from segment_anything_Tea import sam_model_registry_tea
from segment_anything_Stu import sam_model_registry
from importlib import import_module


        
def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    total_all_features = []
    total_all_labels = []

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i, metric_list_dice, all_features, all_labels = test_single_volume(image, label, model, classes=args.num_classes, 
                            patch_size=[args.img_size, args.img_size],test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)

            total_all_features.extend(all_features)
            total_all_labels.extend(all_labels)


            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))


        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        logging.info("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='PVT', help='Name of encoder: PVT or MERIT')
    parser.add_argument('--skip_aggregation', default='additive', help='Type of skip-aggregation: additive or concatenation')
    parser.add_argument("--batch_size", default=16, help="batch size")
    parser.add_argument("--lr", default=0.0026, help="learning rate")
    parser.add_argument("--max_epochs", default=400)
    parser.add_argument("--img_size", default=256)
    parser.add_argument("--save_path", default="/media/lai/data_h/ACDC_Tool")
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--list_dir", default="/media/lai/data_h/datas/ACDC/lists_ACDC/")
    parser.add_argument("--root_dir", default="/media/lai/data_h/datas/ACDC/")
    parser.add_argument("--volume_path", default="/media/lai/data_h/datas/ACDC/test/")
    parser.add_argument("--z_spacing", default=10)
    parser.add_argument("--num_classes", default=4)
    parser.add_argument('--test_save_dir', default='prediction', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
    parser.add_argument('--lora_ckpt', type=str, default='/media/lai/data_h/ACDC_Tool/Pth_stu/best.pth', 
                        help='The checkpoint from LoRA')
    parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
    parser.add_argument('--vit_name', type=str,
                    default='vit_l', help='select one vit model') 
    parser.add_argument('--ckpt', type=str, default='/media/lai/data_h/segment_anything_Tea/sam_vit_l_0b3195.pth',
                    help='Pretrained checkpoint')  
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_stu')
    parser.add_argument('--rank', type=int, default=8, help='Rank for LoRA adaptation')           
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

    snapshot_path = '/media/lai/data_h/ACDC_Tool/SAMed'


    #----------------- teacher model -----------------#
    # sam, img_embedding_size = sam_model_registry_tea[args.vit_name](image_size=args.img_size,
    #                                                             num_classes=3,
    #                                                             checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
    #                                                             pixel_std=[1, 1, 1])
    # pkg = import_module('sam_lora_image_encoder_tea')
    # net = pkg.LoRA_Sam(sam, args.rank).cuda()

    #----------------- student model -----------------#
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                            num_classes=3,
                                                            checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                            pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()
    net.load_state_dict(torch.load(args.lora_ckpt))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'Tool_ACDC/test_log/test_log_' + 'bs_12'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
    test_save_path = os.path.join(args.test_save_dir, 'bs_6', snapshot_name)
    os.makedirs(test_save_path, exist_ok=True)
    
    
    db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    
    results = inference(args, net, testloader, test_save_path)


