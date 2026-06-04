import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Config ----------
csv_path = "output/ceph_hrnet_notebook/training_history.csv"  # change if needed
output_dir = Path("report_figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv(csv_path)

# Basic checks
required_cols = ["epoch", "train_loss", "val_loss", "val_mre_px", "val_mre_mm"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Find best epoch by val_mre_mm (lower is better)
best_idx = df["val_mre_mm"].idxmin()
best_row = df.loc[best_idx]
best_epoch = int(best_row["epoch"])

# ---------- Plot ----------
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

# 1) Loss curves
ax = axes[0, 0]
ax.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="Train Loss")
ax.plot(df["epoch"], df["val_loss"], marker="s", linewidth=2, label="Validation Loss")
ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.8, label=f"Best Epoch = {best_epoch}")
ax.set_title("Loss vs Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

# 2) Validation MRE (px)
ax = axes[0, 1]
ax.plot(df["epoch"], df["val_mre_px"], color="#1f77b4", marker="o", linewidth=2)
ax.scatter([best_epoch], [best_row["val_mre_px"]], color="red", zorder=5)
ax.annotate(
    f"Best: {best_row['val_mre_px']:.2f}px",
    (best_epoch, best_row["val_mre_px"]),
    textcoords="offset points",
    xytext=(10, -10),
)
ax.set_title("Validation MRE (pixels)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MRE (px)")

# 3) Validation MRE (mm)
ax = axes[1, 0]
ax.plot(df["epoch"], df["val_mre_mm"], color="#2ca02c", marker="o", linewidth=2)
ax.scatter([best_epoch], [best_row["val_mre_mm"]], color="red", zorder=5)
ax.annotate(
    f"Best: {best_row['val_mre_mm']:.3f} mm",
    (best_epoch, best_row["val_mre_mm"]),
    textcoords="offset points",
    xytext=(10, -10),
)
ax.set_title("Validation MRE (mm)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MRE (mm)")

# 4) Generalization gap (val_loss - train_loss)
ax = axes[1, 1]
gap = df["val_loss"] - df["train_loss"]
ax.plot(df["epoch"], gap, color="#ff7f0e", marker="d", linewidth=2)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Generalization Gap (val_loss - train_loss)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Gap")

# Figure-level title
fig.suptitle(
    "Training History Summary",
    fontsize=16,
    fontweight="bold"
)

# Save high-quality outputs for report
png_path = output_dir / "training_history_summary.png"
pdf_path = output_dir / "training_history_summary.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print(
    f"Best epoch by val_mre_mm: {best_epoch} | "
    f"val_mre_mm={best_row['val_mre_mm']:.4f}, "
    f"val_mre_px={best_row['val_mre_px']:.4f}, "
    f"val_loss={best_row['val_loss']:.6f}"
)

plt.show()