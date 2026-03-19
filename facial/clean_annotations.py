"""
Clean annotation JSON by removing entries for images not present in the local folder.
Removes 3 missing files: TS004, TS013, TS014
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
ann_file = Path("facial/data/sample_train/anotation_sample_train.json")
img_dir = Path("facial/data/sample_train")
backup_file = Path("facial/data/sample_train/anotation_sample_train.backup.json")

print("=" * 80)
print("ANNOTATION CLEANER - Remove missing image records")
print("=" * 80)

# Load annotation JSON
print(f"\n1. Loading annotation file: {ann_file}")
with open(ann_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"   Total records: {len(data)}")

# Get local files
local_files = set(f.name for f in img_dir.glob("*.jpeg"))
print(f"\n2. Local JPEG files in {img_dir}: {len(local_files)}")

# Extract image names from JSON and check existence
records_to_remove = []
records_to_keep = []

print(f"\n3. Checking each record:")
for i, task in enumerate(data):
    file_upload = task.get("file_upload", "")
    
    # Extract filename (remove hash prefix if present)
    if "-" in file_upload:
        local_name = file_upload.split("-", 1)[-1]
    else:
        local_name = file_upload
    
    exists = local_name in local_files
    
    if exists:
        records_to_keep.append(task)
        status = "✓ KEEP"
    else:
        records_to_remove.append((local_name, file_upload))
        status = "✗ REMOVE"
    
    print(f"   [{i+1:2d}] {status:8s} -> {local_name}")

# Summary
print(f"\n" + "=" * 80)
print(f"SUMMARY:")
print(f"  Records to KEEP:   {len(records_to_keep)}")
print(f"  Records to REMOVE: {len(records_to_remove)}")

if records_to_remove:
    print(f"\nRemoving the following:")
    for local_name, file_upload in records_to_remove:
        print(f"  - {local_name} (JSON: {file_upload})")

# Backup original
if len(records_to_keep) < len(data):
    print(f"\n4. Backing up original: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Backup saved")

    # Save cleaned JSON
    print(f"\n5. Saving cleaned JSON: {ann_file}")
    with open(ann_file, 'w', encoding='utf-8') as f:
        json.dump(records_to_keep, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Cleaned JSON saved")
    
    print(f"\n" + "=" * 80)
    print(f"✓ SUCCESS: Removed {len(records_to_remove)} missing records")
    print(f"  Before: {len(data)} records")
    print(f"  After:  {len(records_to_keep)} records")
    print(f"=" * 80)
else:
    print(f"\n✓ No records to remove - all images exist locally")
