import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

def main():
    base_dir = "/home/nhatnm/code/myprojects/ceph-landmark-detection/ceph/output"
    out_dir = "/home/nhatnm/code/myprojects/ceph-landmark-detection/ceph/visualization"
    os.makedirs(out_dir, exist_ok=True)
    
    models = ["hrnet", "resnet50", "unet"]
    labels = ["HRNetV2-W32", "ResNet50", "UNet"]
    
    data = []
    
    for model, label in zip(models, labels):
        csv_path = os.path.join(base_dir, f"{model}_ceph_landmark_detection", "model_comparison.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Take the first row as the metrics for this model
            row = df.iloc[0].to_dict()
            data.append({
                "Model": label,
                "Params\n(M)": round(row.get("params_m", 0), 2),
                "MRE\n(px)": round(row.get("mre_px", 0), 2),
                "MRE\n(mm)": round(row.get("mre_mm", 0), 2),
                "SDR 2.0mm\n(%)": round(row.get("sdr_2_0mm", 0), 2),
                "SDR 2.5mm\n(%)": round(row.get("sdr_2_5mm", 0), 2)
            })
        else:
            print(f"Warning: {csv_path} does not exist.")
            
    if not data:
        print("No data found to create table.")
        return
        
    df_compare = pd.DataFrame(data)
    
    # Create a plot for the table
    fig, ax = plt.subplots(figsize=(10, 3)) # set size frame
    ax.axis('off')
    ax.axis('tight')
    
    table_plot = ax.table(cellText=df_compare.values, colLabels=df_compare.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(12)
    table_plot.scale(1.2, 1.2)

    # Style the table
    for (i, j), cell in table_plot.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4CAF50')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
    
    out_path = os.path.join(out_dir, "model_metrics_table.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved table image to {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
