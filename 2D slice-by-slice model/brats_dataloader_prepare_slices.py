import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
import cv2  # Added import for OpenCV
import matplotlib.pyplot as plt  

class MedicalImageDataset(Dataset):
    def __init__(self, path, resize=False):
        self.path = path
        self.directories = [f.path for f in os.scandir(path) if f.is_dir()]
        self.ids = self.pathListIntoIds(self.directories)
        self.resize = resize

    def pathListIntoIds(self, dirList):
        return [os.path.basename(dirList[i]) for i in range(len(dirList))]

    def __len__(self):
        return len(self.ids)

    def create_binary_masks(self, seg):
        """
        Create binary masks for Background, WT, TC, and ET.
        """
        background_mask = (seg == 0).astype(np.float32)              # Background
        wt_mask = np.isin(seg, [1, 2, 4]).astype(np.float32)         # Whole Tumor (WT)
        tc_mask = np.isin(seg, [1, 4]).astype(np.float32)            # Tumor Core (TC)
        et_mask = (seg == 4).astype(np.float32)                      # Enhancing Tumor (ET)

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

    def get_largest_tumor_slices(self, seg_tensor):
        """
        Find the 10 slices with the largest tumor distribution and pick one randomly.
        """
        # Combine WT, TC, ET masks to get the tumor distribution
        tumor_mask = seg_tensor[1] + seg_tensor[2] + seg_tensor[3]  # Shape: (D, H, W)
        tumor_area_per_slice = tumor_mask.sum(dim=(1, 2))  # Sum over H and W for each slice D

        # Get indices of the slices with the largest tumor distribution
        top_10_indices = torch.topk(tumor_area_per_slice, 10).indices

        # Pick one randomly from the top 10 slices
        selected_slice_idx = random.choice(top_10_indices.tolist())

        return selected_slice_idx

    def smooth_channel(self, channel, kernel_size=0, blur_size=(21, 21)):
        """Applies morphological closing, dilation, and Gaussian blur to a single mask channel."""
        # Ensure the channel is a NumPy array with type uint8
        channel = channel.astype(np.uint8)

        # Step 1: Morphological Closing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        smoothed_channel = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)

        # Check for NaNs after morphological closing
        if np.isnan(smoothed_channel).any():
            print("NaN detected after morphological closing in smooth_channel.")
            #smoothed_channel = np.nan_to_num(smoothed_channel, nan=0.0)

        # Step 2: Additional Dilation (optional)
        smoothed_channel = cv2.dilate(smoothed_channel, kernel, iterations=1)

        # Check for NaNs after dilation
        if np.isnan(smoothed_channel).any():
            print("NaN detected after dilation in smooth_channel.")
            #smoothed_channel = np.nan_to_num(smoothed_channel, nan=0.0)

        # Step 3: Gaussian Blur and thresholding
        blurred = cv2.GaussianBlur(smoothed_channel.astype(np.float32), blur_size, 0)
        
        # Check for NaNs after Gaussian Blur
        if np.isnan(blurred).any():
            print("NaN detected after Gaussian Blur in smooth_channel.")
            #blurred = np.nan_to_num(blurred, nan=0.0)
        
        # Apply thresholding
        _, smoothed_channel = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)

        # Ensure output is float32 and check for NaNs again
        smoothed_channel = smoothed_channel.astype(np.float32)
        if np.isnan(smoothed_channel).any():
            print("NaN detected after thresholding in smooth_channel.")
            #smoothed_channel = np.nan_to_num(smoothed_channel, nan=0.0)

        return smoothed_channel

    def smooth_mask_slice(self, mask_slice):
        """
        Perform morphological smoothing on the mask slice for each channel.
        """
        # Convert to numpy array
        mask_slice_np = mask_slice.numpy()


        if np.isnan(mask_slice_np).any():
            print("NaN detected in mask_slice_np.")
            #mask_slice_np = np.nan_to_num(mask_slice_np, nan=0.0)


        smoothed_channels = []
        for i in range(mask_slice_np.shape[0]):
            channel = mask_slice_np[i]
            smoothed_channel = self.smooth_channel(channel)
            smoothed_channels.append(smoothed_channel)

        smoothed_mask_slice_np = np.stack(smoothed_channels, axis=0)
        
        # Check for NaNs in the combined result before converting to tensor
        if np.isnan(smoothed_mask_slice_np).any():
            print("NaN detected in smoothed_mask_slice_np.")
            #smoothed_mask_slice_np = np.nan_to_num(smoothed_mask_slice_np, nan=0.0)
        
        # Convert back to tensor
        smoothed_mask_slice = torch.tensor(smoothed_mask_slice_np, dtype=torch.float32)

        # Ensure no NaNs in the final tensor
        if torch.isnan(smoothed_mask_slice).any():
            print("NaN detected in smoothed_mask_slice tensor.")
            #smoothed_mask_slice = torch.nan_to_num(smoothed_mask_slice, nan=0.0)

        return smoothed_mask_slice

    def __getitem__(self, index):
        id = self.ids[index]

        # Load all 4 modalities for the current ID as channels
        t1_file = os.path.join(self.path, id, f"{id}_t1.nii.gz")
        t2_file = os.path.join(self.path, id, f"{id}_t2.nii.gz")
        t1ce_file = os.path.join(self.path, id, f"{id}_t1ce.nii.gz")
        flair_file = os.path.join(self.path, id, f"{id}_flair.nii.gz")
        seg_file = os.path.join(self.path, id, f"{id}_seg.nii.gz")

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

        image = np.transpose(image, (0, 3, 1, 2))
        # Stack the binary masks into a multi-channel mask
        seg = np.stack([background_mask, wt_mask, tc_mask, et_mask], axis=0)  # Shape: (4, D, H, W)
        seg = np.transpose(seg, (0, 3, 1, 2))
        # Convert image and segmentation masks to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)  # Shape: (4, D, H, W)
        seg_tensor = torch.tensor(seg, dtype=torch.float32)      # Shape: (4, D, H, W)

        # Optionally resize
        if self.resize:
            # Implement resizing if needed
            pass

        # Find the slice with the largest tumor distribution
        selected_slice_idx = self.get_largest_tumor_slices(seg_tensor)

        # Get the selected image slice and mask slice
        mask_slice = seg_tensor[:, selected_slice_idx, :, :]     # Shape: (4, H, W)

        # Perform smoothing on the selected mask slice
        smoothed_mask_slice = self.smooth_mask_slice(mask_slice)

        return image_tensor, seg_tensor, smoothed_mask_slice, selected_slice_idx

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

def get_dataloaders(path, batch_size=1, num_workers=2, pin_memory=True, use_saved_splits=True, split_dir='splits2'):
    dataset = MedicalImageDataset(path)

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

        # Set fixed sizes
        train_size = 1000
        val_size = 125
        test_size = 126

        # Split the indices for train, validation, and test
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]

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

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
