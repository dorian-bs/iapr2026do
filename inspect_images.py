import os
import cv2
from collections import Counter

def inspect_dir(directory):
    files = sorted([f for f in os.listdir(directory) if not f.startswith('.')])
    count = len(files)
    shapes = Counter()
    samples = []
    
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(directory, f))
        if img is not None:
            h, w = img.shape[:2]
            shapes[(h, w)] += 1
            if len(samples) < 3:
                samples.append((f, (h, w)))
        else:
            shapes["Invalid"] += 1
            
    return count, shapes, set(os.path.splitext(f)[0] for f in files), samples

img_dir = "project/training_data/training_images/augmented_scenes"
mask_dir = "project/training_data/training_masks/augmented_scenes"

if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
    print(f"Directories missing: {img_dir} or {mask_dir}")
    exit(1)

img_count, img_shapes, img_stems, img_samples = inspect_dir(img_dir)
mask_count, mask_shapes, mask_stems, mask_samples = inspect_dir(mask_dir)

print(f"--- Images: {img_dir} ---")
print(f"Total: {img_count}")
print(f"Shapes: {dict(img_shapes)}")
print(f"Samples: {img_samples}")

print(f"\n--- Masks: {mask_dir} ---")
print(f"Total: {mask_count}")
print(f"Shapes: {dict(mask_shapes)}")
print(f"Samples: {mask_samples}")

print(f"\nStem sets match: {img_stems == mask_stems}")
if img_stems != mask_stems:
    diff = img_stems.symmetric_difference(mask_stems)
    print(f"Symmetric difference (first 5): {list(diff)[:5]}")

