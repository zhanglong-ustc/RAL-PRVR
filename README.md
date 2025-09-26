# Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning

<!-- This repository provides the PyTorch implementation of our 2025 paper **“Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning (RAL)”**.  
RAL is a plug‑and‑play framework that explicitly models **data uncertainty** and performs **confidence‑aware alignment** to improve retrieval robustness under **query ambiguity** and **partial video relevance**. -->

This repository provides the PyTorch implementation of our EMNLP 2025 paper, ["Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning"](https://arxiv.org/abs/2509.01383), based on [GMMFormer v2].

> **TL;DR.** We represent text and video as **Gaussian distributions** to quantify aleatoric uncertainty, train with **distribution alignment (KL)** and **proxy matching** losses, and weight word–frame similarities with **learnable word confidence**. RAL integrates seamlessly with PRVR baselines (e.g., **MS‑SL**, **GMMFormer**, **GMMFormer v2**) and achieves **state‑of‑the‑art** results on **TVR** and **ActivityNet Captions**.

The core implementation of our framework resides in `./src/Models/`.

## Contents
1. [Getting Started](#getting-started)  
2. [Run](#run)  
3. [Method Overview](#method-overview)  
4. [Results](#results)  
5. [Reproducibility Notes](#reproducibility-notes)  
6. [Citation](#citation)  
7. [Acknowledgements](#acknowledgements)

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
We follow common practice and use **pre-extracted visual/text features** for **TVR** and **ActivityNet Captions**. You may reuse features released by prior work [MS-SL]. Please download them from [here](https://github.com/HuiGuanLab/ms-sl).   
Set `root` and `data_root` in the config files (e.g., `./Configs/tvr.py`).

---

## Run

> RAL is *plug‑and‑play* and can be attached to PRVR backbones. The commands below mirror the standard training/eval pipeline; RAL will be enabled via the config/model flags.

**Train on TVR**
```bash
cd src
python main.py -d tvr --gpu 0
```

**Evaluate on TVR**
```bash
python main.py -d tvr --gpu 0 --eval --resume ./checkpoints/ral_gmmv2_tvr.pth
```

**Train on ActivityNet Captions**
```bash
cd src
python main.py -d act --gpu 0
```

**Evaluate on ActivityNet Captions**
```bash
python main.py -d act --gpu 0 --eval --resume ./checkpoints/ral_gmmv2_act.pth
```

> Pretrained checkpoints will be released once cleaned up.

---

## Method Overview

RAL contains two key components:

### 1) Multimodal Semantic Robust Alignment (MSRA)
- **Uncertainty‑aware embeddings.** Encode video frames and query words as **multivariate Gaussians** \(\mathcal{N}(\mu, \sigma^2 I)\) to capture aleatoric uncertainty.
- **Query support set.** Build a **support set of all sentences associated with a video** and aggregate them to model richer text uncertainty.
- **Multi‑granularity aggregation.** Combine **global mean‑pooled features** with **local gated‑attention features** before estimating \(\mu, \sigma\).
- **Losses.**
  - **Distribution Alignment (L<sub>DA</sub>)**: KL divergence between text/video distributions + prior regularization.
  - **Proxy Matching (L<sub>PM</sub>)**: Sample **K** proxies via reparameterization and optimize a multi‑instance InfoNCE.

### 2) Confidence‑aware Set‑to‑Set Alignment (CSA)
- Predict **word‑level confidence** and **weight word–frame similarities**, down‑weighting uninformative words (e.g., stop words) when computing the final query–video score.

The final objective is:
\[ \mathcal{L} = \lambda_1 \mathcal{L}_{\text{InfoNCE}} + \lambda_2 \mathcal{L}_{\text{Triplet}} + \lambda_3 \mathcal{L}_{\text{DA}} + \lambda_4 \mathcal{L}_{\text{PM}}. \]

---

## Results

**Benchmarks:** TVR and ActivityNet Captions.  
**Backbone:** GMMFormer v2 (+ RAL).

| Dataset | R@1 | R@5 | R@10 | R@100 | SumR |
|---|---:|---:|---:|---:|---:|
| **TVR** | **18.2** | **40.4** | **52.1** | **88.0** | **198.8** |
| **ActivityNet** | **8.9** | **27.7** | **40.4** | **79.1** | **156.1** |


---

## Reproducibility Notes

- **Features:** ResNet + I3D for vision; RoBERTa for text (same as prior PRVR works).
- **Optimizer:** Adam, LR `1e-4`, batch `128`, train `150` epochs with early‑stopping (no SumR improvement for 10 epochs).
- **Loss weights:** `λ1=0.05`, `λ2=1`, `λ3=0.004`, `λ4=0.001`.
- **Proxy number:** `K=6` for L<sub>PM</sub>.
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
- **MS‑SL** (ACM MM 2022)
- **GMMFormer / GMMFormer v2** (AAAI 2024 & 2024 arXiv)
- **TVRetrieval**, **ReLoCLNet**

Thanks to the community for releasing datasets and features that make PRVR research possible.
