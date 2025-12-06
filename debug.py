import pickle
import numpy as np

# Load predictions
with open('scratch/proposals/test_predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

# Check first image
first_image = list(predictions.keys())[0]
pred = predictions[first_image]

print(f"First image: {first_image}")
print(f"Keys in prediction: {pred.keys()}")
print(f"boxes shape: {pred['boxes'].shape}")
print(f"scores shape: {pred['scores'].shape}")
print(f"labels shape: {pred['labels'].shape}")
print(f"\nboxes type: {type(pred['boxes'])}")
print(f"scores type: {type(pred['scores'])}")
print(f"\nFirst 5 boxes:\n{pred['boxes'][:5]}")
print(f"\nFirst 5 scores:\n{pred['scores'][:5]}")
print(f"\nFirst 5 labels:\n{pred['labels'][:5]}")


import pickle

with open('scratch/proposals/labels_test.pkl', 'rb') as f:
    gt = pickle.load(f)

print(f"Type of gt: {type(gt)}")
print(f"Length: {len(gt)}")
print(f"\nFirst few items:")

if isinstance(gt, dict):
    for i, (k, v) in enumerate(list(gt.items())[:3]):
        print(f"  Key: {k}, Value type: {type(v)}, Value: {v}")
elif isinstance(gt, list):
    for i in range(min(3, len(gt))):
        print(f"  Item {i}: type={type(gt[i])}, value={gt[i]}")


















