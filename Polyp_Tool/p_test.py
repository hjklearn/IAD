import os
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# 配置
base_dir = ""
my_method = ""
methods_list = []
datasets = []

def read_dice(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    dice_values = [float(x.strip()) for x in lines]
    return np.array(dice_values)

for method in methods_list:
    print(f"\nComparing {method} vs {my_method} (merged datasets):")
    
    # 初始化合并数组
    my_all = []
    comp_all = []
    
    for dataset in datasets:
        my_file = os.path.join(base_dir, my_method, dataset, "p_result.txt")
        comp_file = os.path.join(base_dir, method, dataset, "p_result.txt")
        
        my_dice = read_dice(my_file)
        comp_dice = read_dice(comp_file)
        
        if len(my_dice) != len(comp_dice):
            print(f"Warning: {dataset} has different number of samples! Skipping this dataset.")
            continue
        
        # 合并
        my_all.extend(my_dice)
        comp_all.extend(comp_dice)
    
    my_all = np.array(my_all)
    comp_all = np.array(comp_all)
    
    # 计算最终配对t检验
    t_stat, p_val = ttest_rel(my_all, comp_all)
    print(f"Final merged p-value = {p_val:.22E}")
    
