# ============================================================
# LIST ALL FILES INSIDE:
# /kaggle/input/datasets/hriitk2026/weight-version-v10
# ============================================================

import os
from pathlib import Path
import pandas as pd

ROOT = "/kaggle/input/datasets/hriitk2026/weight-version-v10"

all_files = []

for root, dirs, files in os.walk(ROOT):
    for f in files:

        full_path = os.path.join(root, f)

        size_mb = os.path.getsize(full_path) / (1024 * 1024)

        all_files.append({
            "file": f,
            "size_mb": round(size_mb, 3),
            "path": full_path
        })

df = pd.DataFrame(all_files)

df = df.sort_values("path").reset_index(drop=True)

print("=" * 100)
print(f"TOTAL FILES: {len(df)}")
print("=" * 100)

display(df)

# OPTIONAL:
# save file list

save_path = "/kaggle/working/all_dataset_files.csv"
df.to_csv(save_path, index=False)

print(f"\nSaved file list -> {save_path}")
