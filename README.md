Markdown# From Infusion to Assimilation Distillation for Medical Image Segmentation (CVPR 2026)

![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) 
![last commit](https://img.shields.io/github/last-commit/hjklearn/GPIENet)
![GitHub](https://img.shields.io/github/license/hjklearn/GPIENet?logo=license)
![](https://img.shields.io/github/repo-size/hjklearn/GPIENet?color=green)
![](https://img.shields.io/github/stars/hjklearn/GPIENet)
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/hjklearn)

> **This repository contains the official PyTorch implementation of our CVPR 2026 paper:**  
> **"From Infusion to Assimilation Distillation for Medical Image Segmentation"**  
> 📢 *Updates:* The code will be systematically updated here. Stay tuned!

---

## 🛠️ Environment Setup

Please follow the steps below to build the required environment for this project:

```bash
# 1. Create and activate a conda environment
conda create -n emcadenv python=3.8
conda activate emcadenv

# 2. Install PyTorch (v1.11.0 with CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)

# 3. Install mmcv-full
pip install mmcv-full -f [https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html](https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html)

# 4. Install other dependencies
pip install -r requirements.txt
🗂️ Dataset PreparationFor dataset downloading and processing pipelines, our repository follows the standards established by previous works. Please refer to the following links to prepare your datasets:Download and Preparation: Follow the instructions provided in the MADGNet Repository.Processing Utilities: Please utilize the utils.py script provided by EMCAD: EMCAD/utils/utils.py.📦 Pre-trained ModelsWe provide the pre-trained weights for both the Teacher and Student models evaluated in our paper. You can download them from the links below:DatasetDownload LinkSynapseDownload HereACDCDownload HerePolypDownload Here(Note: Replace the placeholder links above with your actual Google Drive / Baidu Pan / OneDrive links before publishing).🙏 AcknowledgementWe are very grateful for the following excellent open-source works, which have provided the solid basis for our framework:timmEMCADMADGNetTransUNet📝 CitationIf you find our paper, code, or weights useful for your research, please consider citing our work:代码段@inproceedings{hong2026infusion,
  title={From Infusion to Assimilation Distillation for Medical Image Segmentation},
  author={Hong, Jiankang and Luo, Ye and Liu, Yinan and Yuan, Junsong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20985--20995},
  year={2026}
}
