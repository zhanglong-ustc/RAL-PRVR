# Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning

<!-- This repository provides the PyTorch implementation of our 2025 paper **“Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning (RAL)”**.  
RAL is a plug‑and‑play framework that explicitly models **data uncertainty** and performs **confidence‑aware alignment** to improve retrieval robustness under **query ambiguity** and **partial video relevance**. -->

This repository provides the PyTorch implementation of our EMNLP 2025 paper, ["Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning"](https://arxiv.org/abs/2509.01383), based on [GMMFormer v2](https://github.com/huangmozhi9527/GMMFormer_v2).

> **TL;DR.** We explicitly model the **data uncertainty** in PRVR to make retrieval more robust. We model the video and query embeddings as **Gaussian distributions**, where the variance measures the inherent uncertainty of each instance. Based on these distributional representations, we construct **text and video proxies** as multiple possible alignment candidates, allowing the model to capture **diverse cross-modal relations**. In addition, we weight word–frame similarities with **learnable word confidence**. RAL integrates seamlessly with PRVR baselines (e.g., **MS‑SL**, **GMMFormer**, **GMMFormer v2**) and achieves **state‑of‑the‑art** results on **TVR** and **ActivityNet Captions**.


The core implementation of our framework resides in `./src/Models/`.

## Contents
1. [Method Overview](#method-overview)   
2. [Getting Started](#getting-started)
3. [Run](#run) 
5. [Results](#results)  
6. [Reproducibility Notes](#reproducibility-notes)  
7. [Citation](#citation)  
8. [Acknowledgements](#acknowledgements)

---
### Method Overview

![Overview of the proposed framework.](https://raw.githubusercontent.com/zhanglong-ustc/RAL-PRVR/main/fig/main_fig.jpg)
  
**1) Multimodal Semantic Robust Alignment (MSRA)**  
  
- We collect a query support set containing all queries related to the video, obtaining its features $\text{Q}_s$ with rich contexts.  
- Then, we apply multi-granularity aggregation to obtain holistic semantics, and generate distributional representations parameterized by mean vector $\boldsymbol{\mu}$ and variance vector $\boldsymbol{\sigma}$.
- A proxy matching loss $\mathcal{L}\_{PM}$ and a distribution alignment loss $\mathcal{L}_{\mathrm{DA}}$ are used to unify the video and text domains.


**2) Confidence‑aware Set‑to‑Set Alignment (CSA)**  
- We adopt a confidence predictor to assign confidence weights to each word, which is used to adjust the word-frame similarity matrix for video retrieval.

**Final objective.**  
```math
\mathcal{L}
= \lambda_1 \, \mathcal{L}_{\text{InfoNCE}}
+ \lambda_2 \, \mathcal{L}_{\text{Triplet}}
+ \lambda_3 \, \mathcal{L}_{\text{DA}}
+ \lambda_4 \, \mathcal{L}_{\text{PM}}.
```
---
---

## Getting Started

1. **Clone this repository**
```bash
git clone https://github.com/zhanglong-ustc/RAL-PRVR.git
cd RAL-PRVR
```

2. **Create environment & install dependencies**
```bash
conda create -n prvr python=3.9
conda activate prvr
# Install PyTorch (select a CUDA version that matches your system)
conda install pytorch==1.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

3. **Datasets**
We follow common practice and use **pre-extracted visual/text features** for **TVR** and **ActivityNet Captions**. You may reuse features released by prior work [MS-SL](https://github.com/HuiGuanLab/ms-sl). Please download them from [here](https://github.com/HuiGuanLab/ms-sl).   
Set `root` and `data_root` in the config files (e.g., `./Configs/tvr.py`).

---

## Run

> RAL is *plug‑and‑play* and can be attached to PRVR backbones. 

**1. RAL Training**
```bash
cd src
python main.py -d tvr --gpu 0
```

**2. RAL Evaluate**

After training, you can inference using the saved model on val or test set:

Below is an example of inference using RAL, which produces the best performance reported in the paper.

```bash
python main.py -d tvr --gpu 0 --eval --resume ./checkpoints/ral_gmmv2_tvr.pth
```

The model can also be downloaded from [Here](https://drive.google.com/drive/folders/1sH0gansGfOEVBq9W4xeAMTC8fJLFO292?usp=share_link).


> NOTE: The pretrained checkpoints are now available.

## Results

**Benchmarks:** TVR and ActivityNet Captions.  
**Backbone:** GMMFormer v2, GMMFormer and MS-SL.
![Overview of the proposed framework.](https://raw.githubusercontent.com/zhanglong-ustc/RAL-PRVR/main/fig/exp.jpg)

---

## Reproducibility Notes

- **Features:** ResNet + I3D for vision; RoBERTa for text (same as prior PRVR works).
- **Optimizer:** Adam, LR `1e-4`, batch `128`, train `100` epochs with early‑stopping (no SumR improvement for 10 epochs).
- **Loss weights:** $\lambda_1=0.05$, $\lambda_2=1$, $\lambda_3=0.001$, $\lambda_4=0.004$.
- **Proxy number:** `K=6` for $\mathcal{L}_{\mathrm{PM}}$.
- **Retrieval score:** Frame‑level score from CSA is summed with clip‑level score for final ranking.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{zhang2025enhancing,
  title={Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning},
  author={Zhang, Long and Song, Peipei and Dong, Jianfeng and Li, Kun and Yang, Xun},
  journal={arXiv preprint arXiv:2509.01383},
  year={2025}
}
```

---

## Acknowledgements

We build on prior open‑source implementations, including:
- **[MS‑SL](https://github.com/HuiGuanLab/ms-sl)** 
- **[GMMFormer](https://github.com/gimpong/AAAI24-GMMFormer)** 
- **[GMMFormer v2](https://github.com/huangmozhi9527/GMMFormer_v2)**
- **[UATVR](https://github.com/bofang98/UATVR)**

Thanks to the community for releasing datasets and features that make PRVR research possible.
