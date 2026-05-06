import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    base_dir = "/home/nhatnm/code/myprojects/ceph-landmark-detection/ceph/output"
    out_dir = "/home/nhatnm/code/myprojects/ceph-landmark-detection/ceph/visualization"
    os.makedirs(out_dir, exist_ok=True)
    
    # Define models and their corresponding directory prefixes
    models = ["hrnet", "resnet50", "unet"]
    colors = ["blue", "green", "red"]
    labels = ["HRNetV2-W32", "ResNet50", "UNet"]

    # 1. Plot Validation MRE (mm)
    plt.figure(figsize=(10, 6))
    for model, color, label in zip(models, colors, labels):
        csv_path = os.path.join(base_dir, f"{model}_ceph_landmark_detection", "training_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['val_mre_mm'], color=color, label=label, linewidth=2)
        else:
            print(f"Warning: {csv_path} does not exist.")

    plt.title("Validation Mean Radial Error (MRE) in mm")
    plt.xlabel("Epoch")
    plt.ylabel("MRE (mm)")
    plt.legend()
    plt.grid(True)
    
    out_path_mre = os.path.join(out_dir, "val_mre_mm_comparison.png")
    plt.savefig(out_path_mre)
    print(f"Saved MRE plot to {out_path_mre}")
    plt.close()

    # 2. Plot Validation Loss
    plt.figure(figsize=(10, 6))
    for model, color, label in zip(models, colors, labels):
        csv_path = os.path.join(base_dir, f"{model}_ceph_landmark_detection", "training_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['val_loss'], color=color, label=label, linewidth=2)

    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    out_path_loss = os.path.join(out_dir, "val_loss_comparison.png")
    plt.savefig(out_path_loss)
    print(f"Saved Validation Loss plot to {out_path_loss}")
    plt.close()
    
    # 3. Plot Training Loss
    plt.figure(figsize=(10, 6))
    for model, color, label in zip(models, colors, labels):
        csv_path = os.path.join(base_dir, f"{model}_ceph_landmark_detection", "training_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['train_loss'], color=color, label=label, linewidth=2)

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    out_path_train_loss = os.path.join(out_dir, "train_loss_comparison.png")
    plt.savefig(out_path_train_loss)
    print(f"Saved Training Loss plot to {out_path_train_loss}")
    plt.close()

if __name__ == "__main__":
    main()
