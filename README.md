# PAR-CLIP | Clean Conference Rerun

## 0. Scientific accuracy — read this first

- **New experiments do not guarantee your old numbers** (`DPG age 0.390, race 0.264`). The script **never inserts** those numbers. Check the produced CSVs and report actual results only.
- This is a **fresh, self-contained implementation**, not the precise original code/checkpoints that produced your old values. Differences in prompts, training data, metrics, seed, and hyperparameters mean results can change.
- Stage 3 implements a **small Zhang-style residual-bias adaptation**, **not the exact CVPR 2025 Zhang et al. method** (which uses momentum modules and queues). You **cannot** cite its output as an exact reproduction of Zhang et al. For an exact reproduction you must use the authors' original code and protocol, or your previously verified baseline-reproduction experiment.
- Stage 4 uses frozen CLIP and only trains a small residual head. Image demographic directions are fitted on the **training** subset. A regularized Procrustes map is fitted on training group centroids plus fixed text prompts (the identity anchor is an explicit extension for numerical stability).
- **Accuracy and macro-F1:** predict FairFace age/race labels from demographic prompts. **Occupation DPG:** predict one of 16 neutral occupations, measure maximum–minimum predicted occupation rate across demographic groups, average across occupations. These are different prediction tasks and are labeled separately.
- We use FairFace's official **train / val** splits. `val` is *not* an independent untouched test set if used for model selection. The code chooses the smallest validation loss in stage 4. Do not claim independent test validation.
- Stage 1 is a *small scratch-trained CLIP-style pipeline verifier*, not full reproduction of the 16,526,657-parameter BaseCLIP network.

## 1. Installation on the Ubuntu 26.04 system

**Simplest setup:** once the ZIP is extracted, run `bash setup.sh` inside this folder. It installs an isolated Python environment, dependencies, checks CUDA, and runs the smoke tests. Internet is required.

Manual alternative (only if needed):

```bash
cd PAR_CLIP_Conference
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.11
~/.local/bin/uv venv --python 3.11 .venv
~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python test_smoke.py
```

If CUDA reports `False`, install the supported GPU version of PyTorch using the PyTorch official CUDA wheels and rerun the test. The NVIDIA driver reports its supported CUDA level; **the standalone `nvcc` toolkit is not required**.

## 2. Edit only `config.json`

Set seven absolute paths to datasets you already have:

| Field | Point to |
|---|---|
| `fairface_images` | the folder containing `train/` and `val/` images |
| `fairface_train_csv` | official `train_labels.csv` |
| `fairface_val_csv` | official `val_labels.csv` |
| `flickr8k_images` | Flickr8k `Images/` folder |
| `flickr8k_captions` | Flickr8k `captions.txt` |
| `coco_images` | COCO `train2017/` folder |
| `coco_captions` | COCO `annotations/captions_train2017.json` |

The data is **not included** in this ZIP (dataset copyrights and size). Do not commit the datasets to GitHub.

`train_cap` defaults to 20,000 images for a faster rerun. Set it to `null` to use the **entire 86,744-image FairFace training split**. The default validation cap is `null` (all 10,954 validation images). Every result records the actual subset size. **Do not compare reduced-data runs to old full-data experiments as if identical.**

## 3. Run in sequence

```bash
python run.py check       # check paths / PyTorch / GPU
python run.py baseclip    # STAGE 1: train scratch CLIP-style verifier on Flickr8k and COCO
python run.py baseline    # STAGE 2: pretrained ViT-B/32 baseline on FairFace age/race
python run.py zhang       # STAGE 3: Zhang-STYLE adapter only; NOT exact reproduction
python run.py parclip     # STAGE 4: PAR-CLIP SVD + Procrustes + fairness head
```

Alternatively `python run.py all` runs the stages in that order (stops on first error). You may run baseline and PAR-CLIP first if time is short.

**You do not create more folders.** `results/` is automatically created under this project and contains per-stage CSVs, selected `.pt` checkpoints, settings, histories, and reusable cached embeddings. Git ignores `results/` because the embeddings are large and derived from face images; if you need to share results, upload only the validated summary CSV and settings, not face image embeddings.

## 4. Results to use for the conference manuscript

- `results/stage2_baseline.csv` — actual baseline Accuracy, macro-F1, occupation DPG
- `results/stage4_PARCLIP.csv` — actual PAR-CLIP metrics, requested/effective rank `k`, fairness weight
- `results/conference_comparison.csv` — direct comparison (only available if stage 2 ran)
- `results/stage4_age_settings.json` and `...race_settings.json` — exact training sizes, ranks, definitions
- `results/stage1_scratch_verification.csv` — compact retrieval verification
- `results/stage3_zhang_style_NOT_official.csv` — adapted control only; **not a published baseline reproduction**

The manuscript also contains a **theoretical mismatch** to check before submission: its current text says a fully frozen CLIP and an InfoNCE-trained lightweight head, whereas the old extended GitHub code fine-tunes some CLIP layers. This new code implements the *conference-described frozen-backbone variant*, **not the extended fine-tuning code**. Check that the new measurements actually substantiate the statements in the manuscript. Rerun is not a guarantee of acceptance.

## 5. Project contents (no other source folders)

- `run.py` — dataset loaders, 4-stage CLI, caching, results
- `methods.py` — metrics, projection, alignment, PAR-CLIP & Zhang-style adaptation
- `config.json` — edit paths & hyperparameters
- `requirements.txt` — packages
- `test_smoke.py` — quick offline, synthetic code checks
- `.gitignore` — excludes local environment/large derived files
- `README.md` — this guide

To push: `git init && git add . && git commit -m "Conference PAR-CLIP reproducible runner"`, then connect your chosen remote. **Do not upload hardcoded local path settings with sensitive directories or personal data.**
# PAR-CLIP
