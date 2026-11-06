# PAR-CLIP

**Projection, Alignment, and Regularization for Resource Efficient Age and Race Bias Control in CLIP**

Research implementation accompanying the PAR-CLIP conference paper.

## Overview

PAR-CLIP studies mitigation of demographic associations in frozen contrastive vision-language representations. It combines three operations on CLIP embeddings:

1. **Projection (P):** estimate demographic directions from training embeddings with singular value decomposition (SVD) and partially attenuate the associated subspace.
2. **Alignment (A):** apply an orthogonal Procrustes map with an identity anchor to align transformed image features with textual class representations.
3. **Regularization (R):** train a lightweight residual head using demographic prompt contrastive learning and a regularizer measuring parity across neutral occupation prompts.

The model uses pretrained OpenAI CLIP ViT-B/32 with frozen image and text encoders. The paper evaluates age and race classification utility separately from occupation prediction disparity.

## Datasets

| Dataset | Use in this research | Download |
| --- | --- | --- |
| **FairFace** | Age and race classification, demographic subspace estimation, and occupation disparity | [Official FairFace repository](https://github.com/joojs/fairface) |
| **Flickr8k** | Independent image-to-text and text-to-image retrieval experiment | [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) |
| **MS COCO 2017** | Independent image-to-text and text-to-image retrieval experiment | [Official COCO downloads](https://cocodataset.org/#download) |

The datasets are not redistributed in this repository. Download them separately and configure their local paths.

### Reported data partitions

- **FairFace:** 20,000 examples from the official training partition, with 16,000 used to fit the method and 4,000 used for internal model selection. The separate official FairFace validation partition contains 10,954 examples used for final reported evaluation.
- **Flickr8k:** 6,400 training, 800 validation, and 800 evaluation image-caption pairs.
- **COCO:** 16,000 training, 2,000 validation, and 2,000 evaluation image-caption pairs.
- **Seed:** 42.

## Methodology and experiment stages

| Stage | Purpose | Implementation |
| --- | --- | --- |
| **1. Image and text retrieval** | Check a separate retrieval training pipeline on Flickr8k and COCO | Scratch-trained ResNet18 image encoder and GRU text encoder; Recall@1 and Recall@5 in both retrieval directions |
| **2. Pretrained CLIP baseline** | Establish FairFace reference metrics | Frozen CLIP ViT-B/32 and demographic/occupation prompts |
| **3. Vision-language bias removal comparison** | Compare against the method of Zhang et al. under a common local evaluation | ViT-B/32 cached-feature implementation with two MLPs, momentum and counterfactual prompts |
| **4. PAR-CLIP** | Evaluate proposed projection, alignment, and regularization modules | SVD subspace attenuation, Procrustes transformation, fairness-regularized residual head |
| **5. Evaluation** | Compare classification and fairness | Accuracy, macro-F1, occupation DPG and descriptive fairness-utility score |

The Zhang comparison implements the relevant approach using cached image/text embeddings and the same FairFace evaluation setting. It is not equivalent to the paper's original image-level training or its published ranking-based metrics.

### PAR-CLIP computation

For a normalized image embedding $z$, demographic subspace basis $B$, and projection strength $\beta$, the attenuation step is

$$
\widetilde z = \frac{z - \beta BB^\top z}{\lVert z - \beta BB^\top z\rVert}.
$$

An orthogonal Procrustes map aligns the adjusted representation with class prompt vectors. A residual head is then optimized with a demographic prompt contrastive loss and a neutral occupation parity penalty. The CLIP image and text encoder weights remain frozen.

### Main reported settings

| Parameter | Value |
| --- | --- |
| Image and text backbone | OpenAI CLIP ViT-B/32, frozen |
| Embedding dimension | 512 |
| Effective SVD rank | 5 |
| Projection strength $\beta$ | 0.3 (age and race) |
| Fairness loss weight $\lambda$ | 1.0 (age and race) |
| Procrustes identity anchor $\gamma$ | 1 |
| Contrastive temperature | 0.07 |
| Batch size | 256 |
| Learning rate | $10^{-4}$ |
| Training budget | Up to 30 epochs |
| Random seed | 42 |

The paper reports internally selected checkpoints. For exact reproduction, use the final settings and partition definitions rather than assuming all values in the example configuration are the published settings.

## Reported FairFace results

**All methods are evaluated on the common 10,954-image FairFace partition.**

| Attribute | Method | Accuracy ↑ | Macro-F1 ↑ | Occupation DPG ↓ |
| --- | --- | ---: | ---: | ---: |
| Age | Baseline CLIP | 0.4557 | 0.4807 | 0.1144 |
| Age | Zhang et al. comparison | 0.4549 | 0.4809 | 0.1140 |
| Age | **PAR-CLIP** | **0.5915** | **0.5572** | **0.0445** |
| Race | Baseline CLIP | 0.6242 | 0.6236 | 0.0610 |
| Race | Zhang et al. comparison | 0.6235 | 0.6228 | 0.0614 |
| Race | **PAR-CLIP** | **0.7110** | **0.7032** | **0.0265** |

**Metric interpretation:** Accuracy and macro-F1 summarize demographic class predictions. Occupation DPG measures disparities in predictions over 16 neutral occupation prompts across demographic groups (lower is better). These are distinct outcomes.

### Independent retrieval results

| Dataset | Image→Text R@1 | Image→Text R@5 | Text→Image R@1 | Text→Image R@5 |
| --- | ---: | ---: | ---: | ---: |
| Flickr8k | 0.0063 | 0.0425 | 0.0063 | 0.0450 |
| COCO | 0.0115 | 0.0500 | 0.0080 | 0.0540 |

These retrieval results come from a separately trained image/text model and are not fairness results from the pretrained ViT-B/32 model.

## Repository structure

```text
PAR-CLIP/
├── run.py                      # Dataset loading and experiment stages
├── methods.py                  # Projection, alignment, model heads and metrics
├── zhang_reproduction/
│   └── train.py                # Matched ViT-B/32 comparison implementation
├── build_conference_figures.py # Figures from measured results
├── config.example.json         # Configuration template
├── requirements.txt            # Dependencies
├── setup.sh                    # Python environment setup
├── test_smoke.py               # Basic implementation checks
└── README.md                   # Documentation
```

Generated outputs are stored in `results/` and comparison-specific files may be written to `zhang_reproduction/`.

## Installation

The project targets Python 3.11 and a compatible CUDA-enabled PyTorch installation. Experiments were conducted on an NVIDIA RTX 4000 Ada Generation GPU with 20 GB of VRAM.

```bash
git clone https://github.com/ayesha114/PAR-CLIP.git
cd PAR-CLIP
bash setup.sh
cp config.example.json config.json
```

Edit the dataset path values in `config.json` after downloading FairFace, Flickr8k, and COCO. The configuration file is excluded from version control because it contains local machine paths.

## Running the experiments

```bash
.venv/bin/python run.py check
.venv/bin/python test_smoke.py

# Stage 1: separate retrieval experiment
.venv/bin/python run.py baseclip

# Stage 2: frozen pretrained CLIP baseline
.venv/bin/python run.py baseline

# Stage 3: ViT-B/32 comparison used for the paper
.venv/bin/python zhang_reproduction/train.py --attribute both

# Stage 4: proposed PAR-CLIP model
.venv/bin/python run.py parclip

# After generating the required result files:
.venv/bin/python build_conference_figures.py
```

**Important for reproduction:** `run.py zhang` is a separate, simplified ViT-B/16 comparison path. The paper's matched ViT-B/32 comparison is implemented in `zhang_reproduction/train.py`, as shown above. The example `config.example.json` is not a frozen record of the exact final age model selection: confirm the reported age fairness weight and projection selection settings before running.

## Scope and limitations

The study measures age and race classification and disparity among the specified neutral occupation prompts on FairFace. Results are based on a single reported seed and require validation beyond this setting before broader claims can be made. The age evaluation is exploratory because the official partition was examined during development, as discussed in the manuscript.

## Reference

**PAR-CLIP: Projection, Alignment, and Regularization for Resource Efficient Age and Race Bias Control in CLIP.** Conference manuscript.

Comparator: H. Zhang, Y. Guo, and M. Kankanhalli, “Joint Vision-Language Social Bias Removal for CLIP,” *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025.
