# From Infusion to Assimilation Distillation for Medical Image Segmentation (CVPR 2026)

![GitHub](https://img.shields.io/github/license/hjklearn/IAD?logo=license)
![](https://img.shields.io/github/repo-size/hjklearn/IAD?color=green)
![](https://img.shields.io/github/stars/hjklearn/IAD)
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/hjklearn)
[![Paper](https://img.shields.io/badge/Paper-CVF-blue.svg)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf)

> **This repository contains the official PyTorch implementation of our CVPR 2026 paper:** > **"From Infusion to Assimilation Distillation for Medical Image Segmentation"** [[PDF]](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf)  
> 📢 *Updates:* The code will be systematically updated here. Stay tuned!

---

## 💡 Introduction

Knowledge Distillation (KD) has shown immense potential in improving the performance of lightweight models for medical image segmentation. However, conventional KD methods often force the student network to rigidly mimic the teacher's features (**Infusion**), which can lead to feature mismatch and sub-optimal performance due to the inherent capacity gap between the two networks.

To address this, we propose **IAD (Infusion to Assimilation Distillation)**. Our framework redefines the distillation process by transitioning from hard feature *infusion* to adaptive knowledge *assimilation*. This allows the student network to selectively digest and integrate the teacher's knowledge based on its own structural capacity, preserving crucial anatomical boundaries and topologies in medical images.

## ✨ Highlights / Key Contributions

- 🚀 **A Novel KD Paradigm:** We shift the distillation process from conventional "Infusion" to an adaptive "Assimilation" mechanism, bridging the semantic gap between teacher and student models.
- 🧠 **Domain-Specific Design:** Specifically optimized for medical image segmentation, effectively handling complex anatomical structures and class imbalances.
- 🏆 **State-of-the-Art Performance:** Achieves superior segmentation accuracy on mainstream medical datasets (e.g., *[Insert Dataset 1]*, *[Insert Dataset 2]*) while maintaining a highly efficient inference speed.
- 🧩 **Plug-and-Play:** The proposed IAD module can be seamlessly integrated into various existing CNN and Transformer-based segmentation architectures.

## 🖼️ Framework

<div align="center">
  <!-- 请将下面 src 的路径替换为你仓库中架构图的真实路径 -->
  <img src="assets/framework.png" alt="IAD Framework" width="90%">
  <br>
  <em>Figure 1: The overall architecture of the proposed IAD (Infusion to Assimilation Distillation) framework.</em>
</div>

## 🛠️ Environment Setup

Please follow the steps below to build the required environment for this project:

```
# 1. Create and activate a conda environment
conda create -n emcadenv python=3.8
conda activate emcadenv

# 2. Install PyTorch (v1.11.0 with CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)

# 3. Install mmcv-full
pip install mmcv-full -f [https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html](https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html)

# 4. Install other dependencies
pip install -r requirements.txt
```

## 🗂️ Dataset Preparation

For dataset downloading and processing pipelines, our repository follows the standards established by previous works. Please refer to the following links to prepare your datasets:

*   **Download and Preparation:** Follow the instructions provided in the [MADGNet Repository](https://github.com/Inha-CVAI/MADGNet) and [EMCAD Repository](https://github.com/SLDGroup/EMCAD).

| Dataset | Download Link |
| :--- | :--- |
| **Synapse** | [Download Here](https://github.com/SLDGroup/EMCAD) |
| **ACDC** | [Download Here](https://github.com/SLDGroup/EMCAD) |
| **Polyp** | [Download Here](https://github.com/Inha-CVAI/MADGNet) |

## 📦 Pre-trained Models

We provide the pre-trained weights for both the Teacher and Student models in training stage . You can download them from the links below:

| Training Datasets | Model pre-trained weights Download Link |
| :--- | :--- |
| **Synapse** | [teacher pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) and [student pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) Code: 2234|
| **ACDC** | [teacher pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) and [student pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) Code: 2234|
| **Polyp** | [teacher pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) and [student pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) Code: 2234|

We provide the pre-trained weights for both the Teacher and Student models in knowledge distillation training stage. You can download them from the links below:

| Knowledge distillation training Datasets | Model pre-trained weights Download Link |
| :--- | :--- |
| **Synapse** | [student pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) Code: 2234|
| **ACDC** | [student pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) Code: 2234|
| **Polyp** | [student pre-trained weights and prediction results](https://pan.baidu.com/s/1AplUfjHojNvPJykyecSwig?pwd=2234) Code: 2234|


## 🚀 Running the Code

### Training

To train the teacher and student model, run:

```
# Example for teacher and on Synapse dataset
CUDA_VISIBLE_DEVICES="0"  python train.py  --split='train' --warmup --AdamW --max_epochs=300 --stop_epoch=300 --vit_name='vit_l'
```

```
# Example for teacher and on Polyp dataset
python train_isic.py --train_data_type PolypSegData --test_data_type CVC-ColonDB
```

```
# Example for teacher and on ACDC dataset
CUDA_VISIBLE_DEVICES="0" python train_ACDC.py 
```

### Knowledge Distillation Training

To train the student model under our proposed Assimilation Distillation framework, run:

```
# Example for teacher and on Synapse dataset
CUDA_VISIBLE_DEVICES="0"  python train.py  --split='train' --warmup --AdamW --max_epochs=300 --stop_epoch=300 --vit_name='vit_l'
```

```
# Example for teacher and on Polyp dataset
python train_isic.py --train_data_type PolypSegData --test_data_type CVC-ColonDB
```

```
# Example for teacher and on ACDC dataset
CUDA_VISIBLE_DEVICES="0" python train_ACDC.py 
```

### Evaluation

To evaluate the performance (e.g., Dice score, HD95) of the trained teacher and student model:

```
# Example for teacher and on Synapse dataset
CUDA_VISIBLE_DEVICES="0" python test.py --is_savenii --stage=3
```

```
# Example for teacher and on Polyp dataset
python test_isic.py --train_data_type PolypSegData --test_data_type CVC-ColonDB
```

```
# Example for teacher and on ACDC dataset
CUDA_VISIBLE_DEVICES="0" python test_ACDC.py 
```

## 🙏 Acknowledgement

We are very grateful for the following excellent open-source works, which have provided the solid basis for our framework:

*   [EMCAD](https://github.com/SLDGroup/EMCAD)
*   [MADGNet](https://github.com/Inha-CVAI/MADGNet)
*   [TransUNet](https://github.com/Beckschen/TransUNet)


## 📝 Citation

If you find our framework, code, or pre-trained models beneficial to your research, please consider citing our CVPR 2026 paper:

```
@inproceedings{hong2026infusion,
  title={From Infusion to Assimilation Distillation for Medical Image Segmentation},
  author={Hong, Jiankang and Luo, Ye and Liu, Yinan and Yuan, Junsong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20985--20995},
  year={2026}
}
```
