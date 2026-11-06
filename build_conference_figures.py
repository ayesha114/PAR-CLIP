import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
RESULTS = ROOT / "results"
ZHANG = ROOT / "zhang_reproduction"
OUT = RESULTS / "conference_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300
})

SINGLE_W = 3.45
DOUBLE_W = 7.10

def save_all(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)

# -------------------------------------------------
# Build final FairFace comparison table
# -------------------------------------------------
age_json = json.loads((RESULTS / "age_retrain_evaluation.json").read_text())
paper_vi = pd.read_csv(RESULTS / "paper_table_VI_measured.csv")
zhang_df = pd.read_csv(ZHANG / "zhang_vitb32_matched_results.csv")

rows = []

# AGE
age_baseline = age_json["official_baseline"]
age_parclip = age_json["official_candidate"]
age_zhang = zhang_df[zhang_df["attribute"] == "age"].iloc[0].to_dict()

rows.append({
    "attribute": "Age",
    "method": "Baseline CLIP",
    "accuracy": float(age_baseline["accuracy"]),
    "macro_f1": float(age_baseline["macro_f1"]),
    "occupation_dpg": float(age_baseline["occupation_dpg"])
})
rows.append({
    "attribute": "Age",
    "method": "Zhang ViT-B/32 (matched-budget)",
    "accuracy": float(age_zhang["accuracy"]),
    "macro_f1": float(age_zhang["macro_f1"]),
    "occupation_dpg": float(age_zhang["occupation_dpg"])
})
rows.append({
    "attribute": "Age",
    "method": "PAR-CLIP",
    "accuracy": float(age_parclip["accuracy"]),
    "macro_f1": float(age_parclip["macro_f1"]),
    "occupation_dpg": float(age_parclip["occupation_dpg"])
})

# RACE
race_block = paper_vi[paper_vi["attribute"].str.lower() == "race"]
race_baseline = race_block[race_block["method"] == "Baseline CLIP"].iloc[0]
race_parclip = race_block[race_block["method"].str.contains("PAR-CLIP", case=False)].iloc[0]
race_zhang = zhang_df[zhang_df["attribute"] == "race"].iloc[0].to_dict()

rows.append({
    "attribute": "Race",
    "method": "Baseline CLIP",
    "accuracy": float(race_baseline["accuracy"]),
    "macro_f1": float(race_baseline["macro_f1"]),
    "occupation_dpg": float(race_baseline["occupation_dpg"])
})
rows.append({
    "attribute": "Race",
    "method": "Zhang ViT-B/32 (matched-budget)",
    "accuracy": float(race_zhang["accuracy"]),
    "macro_f1": float(race_zhang["macro_f1"]),
    "occupation_dpg": float(race_zhang["occupation_dpg"])
})
rows.append({
    "attribute": "Race",
    "method": "PAR-CLIP",
    "accuracy": float(race_parclip["accuracy"]),
    "macro_f1": float(race_parclip["macro_f1"]),
    "occupation_dpg": float(race_parclip["occupation_dpg"])
})

final_df = pd.DataFrame(rows)
final_df.to_csv(OUT / "final_fairface_comparison.csv", index=False)

# -------------------------------------------------
# Figure 1: FairFace combined metrics (double-column)
# -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.5))
metrics = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("occupation_dpg", "Occupation DPG ↓")
]

for ax, (col, title) in zip(axes, metrics):
    pivot = final_df.pivot(index="attribute", columns="method", values=col)
    pivot = pivot[["Baseline CLIP", "Zhang ViT-B/32 (matched-budget)", "PAR-CLIP"]]
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend_.remove()

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
save_all(fig, "Fig1_FairFace_Comparison_DoubleColumn")

# -------------------------------------------------
# Figure 2: Accuracy/F1 only (single-column friendly)
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.3))
for ax, (col, title) in zip(axes, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
    pivot = final_df.pivot(index="attribute", columns="method", values=col)
    pivot = pivot[["Baseline CLIP", "Zhang ViT-B/32 (matched-budget)", "PAR-CLIP"]]
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend_.remove()

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
save_all(fig, "Fig2_Accuracy_F1")

# -------------------------------------------------
# Figure 3: DPG only
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(SINGLE_W, 2.4))
pivot = final_df.pivot(index="attribute", columns="method", values="occupation_dpg")
pivot = pivot[["Baseline CLIP", "Zhang ViT-B/32 (matched-budget)", "PAR-CLIP"]]
pivot.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Occupation DPG (lower is better)")
ax.set_xlabel("")
ax.set_ylabel("DPG")
ax.legend(frameon=False)
save_all(fig, "Fig3_DPG")

# -------------------------------------------------
# Figure 4: Stage 1 retrieval
# -------------------------------------------------
stage1 = pd.read_csv(RESULTS / "stage1_scratch_verification.csv")

fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.6))
stage1 = stage1.copy()
stage1["dataset"] = stage1["dataset"].str.upper()

plot_df = stage1.set_index("dataset")[[
    "image_to_text_R1",
    "image_to_text_R5",
    "text_to_image_R1",
    "text_to_image_R5"
]]
plot_df.columns = ["I2T R@1", "I2T R@5", "T2I R@1", "T2I R@5"]
plot_df.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Stage 1 BaseCLIP Retrieval on Held-out Test Sets")
ax.set_ylabel("Recall")
ax.set_xlabel("")
ax.legend(frameon=False, ncol=2)
save_all(fig, "Fig4_Stage1_Retrieval")

# -------------------------------------------------
# Figure 5 & 6: Training curves if available
# -------------------------------------------------
def plot_history(csv_path, title, out_name):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() != "epoch"]
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.4))
    for c in cols:
        ax.plot(df["epoch"], df[c], label=c)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend(frameon=False)
    save_all(fig, out_name)

flickr_hist = RESULTS / "stage1_flickr8k_history.csv"
if flickr_hist.exists():
    plot_history(flickr_hist, "Flickr8k Training History", "Fig5_Flickr8k_TrainingCurve")

coco_hist = RESULTS / "stage1_coco_history.csv"
if coco_hist.exists():
    plot_history(coco_hist, "COCO Training History", "Fig6_COCO_TrainingCurve")

print("Saved all conference-ready figures to:", OUT.resolve())
print("Files:")
for p in sorted(OUT.iterdir()):
    print("-", p.name)
