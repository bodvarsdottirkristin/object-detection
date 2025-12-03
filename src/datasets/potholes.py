import torch
import pickle
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ProposalDataset(Dataset):
    """Dataset for loading and classifying proposals."""
    
    def __init__(self, dataset_path, proposals_dict, labels_dict, img_size=224):
        self.images_dir = Path(dataset_path) / 'images'
        self.proposals_dict = proposals_dict
        self.labels_dict = labels_dict
        self.img_size = img_size
        
        # Create list of (image_filename, proposal_idx) pairs
        self.samples = []
        for image_filename in proposals_dict.keys():
            num_proposals = len(proposals_dict[image_filename]['proposals'])
            for prop_idx in range(num_proposals):
                self.samples.append((image_filename, prop_idx))
        
        self.mean = None
        self.std = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_filename, prop_idx = self.samples[idx]
        
        # Load image
        image = self._load_image(image_filename)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Crop proposal
        x1, y1, x2, y2 = self.proposals_dict[image_filename]['proposals'][prop_idx]
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        if self.mean is not None and self.std is not None:
            crop = (crop - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        label = self.labels_dict[image_filename][prop_idx]
        
        return crop, torch.tensor(label, dtype=torch.long)
    
    def _load_image(self, image_filename):
        image_path = self.images_dir / image_filename
        if not image_path.exists():
            for ext in ['.jpg', '.png', '.jpeg']:
                alt_path = self.images_dir / (image_filename.split('.')[0] + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            return None
        
        image = cv2.imread(str(image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
    
    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std
    
    def compute_normalization_stats(self):
        """Compute mean and std from all samples"""
        all_pixels = []
        
        for idx in range(min(1000, len(self))):  # Sample first 1000 for speed
            image_filename, prop_idx = self.samples[idx]
            image = self._load_image(image_filename)
            
            if image is not None:
                x1, y1, x2, y2 = self.proposals_dict[image_filename]['proposals'][prop_idx]
                crop = image[y1:y2, x1:x2]
                crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                crop = crop.astype(np.float32) / 255.0
                all_pixels.append(crop.reshape(-1, 3))
        
        if all_pixels:
            all_pixels = np.concatenate(all_pixels, axis=0)
            mean = torch.tensor(all_pixels.mean(axis=0), dtype=torch.float32)
            std = torch.tensor(all_pixels.std(axis=0), dtype=torch.float32)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        
        return mean, std
    
    def get_class_weights(self):
        """Compute weights for handling class imbalance"""
        labels = []
        for img_file in self.labels_dict.keys():
            labels.extend(self.labels_dict[img_file])
        
        labels = np.array(labels)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        sample_weights = class_weights[labels]
        return torch.from_numpy(sample_weights).double()
    
    @staticmethod
    def get_dataloaders(dataset_path, proposals_dir, splits_path, batch_size=32, num_workers=0):
        """Create train, val, test dataloaders"""
        import json
        
        proposals_dir = Path(proposals_dir)
        
        with open(splits_path, 'r') as f:
            json.load(f)  # Load splits (not used here, but available)
        
        loaders = {}
        train_dataset = None
        
        for split in ['train', 'val', 'test']:
            # Load proposals and labels
            with open(proposals_dir / f'proposals_{split}.pkl', 'rb') as f:
                proposals_dict = pickle.load(f)
            with open(proposals_dir / f'labels_{split}.pkl', 'rb') as f:
                labels_dict = pickle.load(f)
            
            dataset = ProposalDataset(dataset_path, proposals_dict, labels_dict)
            
            # Compute normalization only on training set
            if split == 'train':
                mean, std = dataset.compute_normalization_stats()
                train_dataset = dataset
            
            dataset.set_normalization(mean, std)
            
            # Create loader with weighted sampler for training
            if split == 'train':
                weights = dataset.get_class_weights()
                sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
                loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
            else:
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            loaders[split] = loader
        
        return loaders['train'], loaders['val'], loaders['test'], train_dataset


if __name__ == "__main__":
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    PROPOSALS_DIR = "proposals"
    SPLITS_PATH = "splits.json"
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset = ProposalDataset.get_dataloaders(
        DATASET_PATH, PROPOSALS_DIR, SPLITS_PATH, batch_size=32
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print(f"\nMean: {train_dataset.mean}")
    print(f"Std: {train_dataset.std}")
    
    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Label distribution: {torch.bincount(labels)}")