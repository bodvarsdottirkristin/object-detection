import pickle
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class PotholeDataset(Dataset):
    def __init__(self, db_path, transform=None):
        with open(db_path, "rb") as f:
            self.samples = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. Load FULL image
        img = cv2.imread(item["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. CROP the proposal
        x1, y1, x2, y2 = map(int, item["box"])

        # Safety check for image boundaries
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop
        patch = img[y1:y2, x1:x2]

        # Handle empty crops (rare edge case)
        if patch.size == 0:
            patch = np.zeros((224, 224, 3), dtype=np.uint8)

        # 3. Convert to PIL for TorchVision transforms
        patch_pil = Image.fromarray(patch)

        if self.transform:
            patch_pil = self.transform(patch_pil)

        return patch_pil, item["label"]


# Example Usage
if __name__ == "__main__":
    tfs = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PotholeDataset("src/datasets/proposals/train_db.pkl", transform=tfs)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    images, labels = next(iter(loader))
    print(f"Batch Shape: {images.shape}")  # Should be [32, 3, 224, 224]
