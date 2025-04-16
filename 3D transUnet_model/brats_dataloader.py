import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F

class MedicalImageDataset(Dataset):
    def __init__(self, path, target_size=(128, 128, 128),resize=True):
        self.path = path
        self.directories = [f.path for f in os.scandir(path) if f.is_dir()]
        self.ids = self.pathListIntoIds(self.directories)
        self.target_size = target_size
        self.resize=resize

    def pathListIntoIds(self, dirList):
        return [os.path.basename(dirList[i]) for i in range(len(dirList))]

    def __len__(self):
        return len(self.ids)  # Each ID represents a set of 4 modalities

    def resize_volume(self, volume, is_segmentation=False):
        # Resize the volume to the target size
        B, C, D, H, W = volume.shape
        if is_segmentation:
            # Cast to float before resizing, then cast back to long after resizing
            volume = volume.float()
            resized_volume = F.interpolate(volume, size=self.target_size, mode='nearest')  # Nearest neighbor for masks
            return resized_volume
        else:
            return F.interpolate(volume, size=self.target_size, mode='trilinear', align_corners=False)


    def create_binary_masks(self, seg):
        """
        Create binary masks for Background, WT, TC, and ET.
        """
        background_mask = (seg == 0).astype(np.float32)              # Background
        wt_mask = np.isin(seg, [1, 2, 4]).astype(np.float32)         # Whole Tumor
        tc_mask = np.isin(seg, [1, 4]).astype(np.float32)            # Tumor Core
        et_mask = (seg == 4).astype(np.float32)         # Enhancing Tumor

        return background_mask, wt_mask, tc_mask, et_mask
        
    def normalize_volume(self, volume):
        # Normalize the volume per scan: zero mean and unit variance
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            volume = (volume - mean) / std
        else:
            volume = volume - mean  # If std is zero, set the mean to zero
        return volume

    def get_class_distribution(self):
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Background, WT, TC, ET
        for idx in range(len(self)):
            _, seg_tensor = self.__getitem__(idx)
            counts = seg_tensor.sum(dim=(1, 2, 3)).numpy()  # Sum over spatial dimensions
            for c in range(4):
                class_counts[c] += counts[c]
        return class_counts

    

    def __getitem__(self, index):
        id = self.ids[index]

        # Load all 4 modalities for the current ID as channels
        t1_file = f"{self.path}/{id}/{id}_t1.nii.gz"
        t2_file = f"{self.path}/{id}/{id}_t2.nii.gz"
        t1ce_file = f"{self.path}/{id}/{id}_t1ce.nii.gz"
        flair_file = f"{self.path}/{id}/{id}_flair.nii.gz"
        seg_file = f"{self.path}/{id}/{id}_seg.nii.gz"

        # Load images and segmentation
        t1 = nib.load(t1_file).get_fdata()
        t2 = nib.load(t2_file).get_fdata()
        t1ce = nib.load(t1ce_file).get_fdata()
        flair = nib.load(flair_file).get_fdata()
        seg = nib.load(seg_file).get_fdata()

        # Normalize each modality
        t1 = self.normalize_volume(t1)
        t2 = self.normalize_volume(t2)
        t1ce = self.normalize_volume(t1ce)
        flair = self.normalize_volume(flair)

        # Create binary masks for Background, WT, TC, ET
        background_mask, wt_mask, tc_mask, et_mask = self.create_binary_masks(seg)


        # Stack the four modalities into a single tensor as different channels
        image = np.stack([t1, t2, t1ce, flair], axis=0)  # Shape: (4, D, H, W)

        # Stack the binary masks into a multi-channel mask
        seg = np.stack([background_mask, wt_mask, tc_mask, et_mask], axis=0)  # Shape: (4, D, H, W)


        # Convert image and segmentation mask to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)  # Shape: (4, D, H, W)
        seg_tensor = torch.tensor(seg, dtype=torch.float32)      # Shape: (4, D, H, W)

        if self.resize:
            # Resize the images
            image_tensor = self.resize_volume(image_tensor.unsqueeze(0)).squeeze(0)  # Shape: (4, target_d, target_h, target_w)
            # Resize the segmentation mask (using nearest neighbor interpolation)
            seg_tensor = self.resize_volume(seg_tensor.unsqueeze(0), is_segmentation=True).squeeze(0)  # Shape: (4, target_d, target_h, target_w)

        return image_tensor, seg_tensor


# Functions to save and load dataset splits
def save_splits(splits, split_dir='splits'):
    os.makedirs(split_dir, exist_ok=True)
    for split_name, split_indices in splits.items():
        np.save(os.path.join(split_dir, f'{split_name}_indices.npy'), split_indices)


def load_splits(split_dir='splits'):
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_indices = np.load(os.path.join(split_dir, f'{split_name}_indices.npy'))
        splits[split_name] = split_indices
    return splits



def get_dataloaders(path, batch_size=2, train_split=0.8, val_split=0.1, num_workers=2, pin_memory=True, target_size=(128, 128, 128),use_saved_splits=True, split_dir='splits'):
    dataset = MedicalImageDataset(path, target_size)
    
    if use_saved_splits:
        # Load the saved splits
        splits = load_splits(split_dir=split_dir)
        train_indices = splits['train']
        val_indices = splits['val']
        test_indices = splits['test']
    else:
        # Generate random indices and split them manually
        total_size = len(dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        # Split the indices for train, validation, and test
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Save the splits for reuse
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        save_splits(splits, split_dir=split_dir)

    # Use Subset to create DataLoaders with the same split indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
     
    print(f"train dataset: {len(train_dataset)} val dataset: {len(val_dataset)} test dataset: {len(test_dataset)}") 

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader