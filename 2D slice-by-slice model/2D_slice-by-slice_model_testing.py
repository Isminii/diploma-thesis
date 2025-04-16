import os
import torch
from bigger_predictor3D_model import BrainTumorPredictorTransformer
from brats_dataloader_prepare_slices import get_dataloaders
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def dice_loss_per_slice(pred_slice, target_slice, epsilon=1e-6):
    """
    Compute Dice loss for a single slice (all classes included).
    pred_slice and target_slice should be shape (C, H, W).
    """
    # Flatten
    pred_flat = pred_slice.contiguous().view(-1)
    target_flat = target_slice.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum().item()
    sum_pred = pred_flat.sum().item()
    sum_target = target_flat.sum().item()

    dice_score = (2.0 * intersection + epsilon) / (sum_pred + sum_target + epsilon)
    dice_loss = 1 - dice_score
    return dice_loss

def compute_tumor_proportion(target_slice):
    """
    Compute the tumor proportion for a single slice.
    Assuming 4 classes: 0=Background, 1=WT, 2=TC, 3=ET.
    Tumor proportion = (# of tumor voxels) / (total voxels in the slice).
    """
    # target_slice shape: (4, H, W)
    # Classes 1,2,3 represent tumor regions
    tumor_voxels = (target_slice[1] > 0).sum().item() + \
                   (target_slice[2] > 0).sum().item() + \
                   (target_slice[3] > 0).sum().item()

    total_voxels = target_slice.shape[1] * target_slice.shape[2]
    tumor_proportion = tumor_voxels / total_voxels
    return tumor_proportion

def plot_mean_metrics_per_slice(counts_per_slice, total_loss_per_slice, total_tumor_per_slice, 
                                output_path='mean_loss_and_tumor_proportion_per_slice.png'):
    """
    Plots the mean tumor proportion and mean Dice loss per slice index.
    """
    if not counts_per_slice:
        print("No slice data available to plot.")
        return

    sorted_slice_indices = sorted(counts_per_slice.keys())

    mean_loss_per_slice = []
    mean_tumor_proportion_per_slice = []

    for i in sorted_slice_indices:
        mean_loss = total_loss_per_slice[i] / counts_per_slice[i]
        mean_tumor_proportion = total_tumor_per_slice[i] / counts_per_slice[i]
        mean_loss_per_slice.append(mean_loss)
        mean_tumor_proportion_per_slice.append(mean_tumor_proportion)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(sorted_slice_indices, mean_loss_per_slice, color='r', label='Mean Dice Loss')
    ax1.set_xlabel('Slice Index')
    ax1.set_ylabel('Mean Dice Loss', color='r')

    ax2.plot(sorted_slice_indices, mean_tumor_proportion_per_slice, color='b', label='Mean Tumor Proportion')
    ax2.set_ylabel('Mean Tumor Proportion', color='b')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Mean Dice Loss and Tumor Proportion per Slice Index')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved at {output_path}")



def test_model(model, test_loader, device, save_outputs=False,num_samples=20, output_dir='test_outputs_transformer'):
    model.eval()
    dice_total = 0.0
    iou_total = 0.0
    dice_per_class = [0.0] * 4  # Assuming 4 classes: Background, WT, TC, ET
    iou_per_class = [0.0] * 4


    # Initialize dictionaries to collect per-slice losses and tumor proportions
    total_loss_per_slice = defaultdict(float)
    total_tumor_per_slice = defaultdict(float)
    counts_per_slice = defaultdict(int)


    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
        samples_saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Unpack the batch
            image_tensor, seg_tensor, initial_mask_slice, selected_slice_idx = batch

            # Move data to device
            image_tensor = image_tensor.squeeze(0).to(device)
            seg_tensor = seg_tensor.squeeze(0).to(device)
            initial_mask_slice = initial_mask_slice.squeeze(0).to(device)
            selected_slice_idx = selected_slice_idx.item()

            D = image_tensor.shape[1]
            H, W = image_tensor.shape[2], image_tensor.shape[3]
            full_mask = torch.zeros((4, D, H, W), device=image_tensor.device)
            full_mask[:, selected_slice_idx, :, :] = initial_mask_slice


            front_indices = list(range(selected_slice_idx + 1, D))
            back_indices = list(range(selected_slice_idx - 1, -1, -1))


            # Initialize previous mask
            prev_mask = initial_mask_slice
            # Process forward slices
            for idx in front_indices:
                current_image_slice = image_tensor[:, idx, :, :]
                target_mask_slice = seg_tensor[:, idx, :, :]

                # Forward pass
                next_mask_pred = model(current_image_slice, prev_mask)
                
                full_mask[:, idx, :, :] = next_mask_pred
                # Update prev_mask
                prev_mask = next_mask_pred.detach()


            # Initialize previous mask
            prev_mask = initial_mask_slice
            # Process forward slices
            for idx in back_indices:
                current_image_slice = image_tensor[:, idx, :, :]
                target_mask_slice = seg_tensor[:, idx, :, :]

                # Forward pass
                next_mask_pred = model(current_image_slice, prev_mask)

                full_mask[:, idx, :, :] = next_mask_pred
                # Update prev_mask
                prev_mask = next_mask_pred.detach()

            
            # Apply sigmoid to get probabilities
            fullmask_sigmoid = torch.sigmoid(full_mask)

            # Threshold to get binary masks
            predicted = (fullmask_sigmoid > 0.5).float()

            # Compute metrics
            for c in range(4):
                pred_flat = predicted[c].contiguous().view(-1)
                target_flat = seg_tensor[c].contiguous().view(-1)
                intersection = (pred_flat * target_flat).sum().item()
                dice_score = (2.0 * intersection + 1e-6) / (pred_flat.sum().item() + target_flat.sum().item() + 1e-6)
                union = pred_flat.sum().item() + target_flat.sum().item() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                dice_per_class[c] += dice_score
                iou_per_class[c] += iou
                dice_total += dice_score
                iou_total += iou


            # Compute slice-wise metrics
            # Iterate over each slice
            for i in range(D):
                pred_slice = predicted[:, i, :, :]   # Shape: (4, H, W)
                target_slice = seg_tensor[:, i, :, :]  # Shape: (4, H, W)

                # Compute dice loss per slice
                slice_dice_loss = dice_loss_per_slice(pred_slice, target_slice)

                # Compute tumor proportion per slice from target
                slice_tumor_prop = compute_tumor_proportion(target_slice)

                # Update the dictionaries
                total_loss_per_slice[i] += slice_dice_loss
                total_tumor_per_slice[i] += slice_tumor_prop
                counts_per_slice[i] += 1

           

            # Save sample outputs if requested
            if save_outputs and samples_saved < num_samples:
                save_sample_outputs(image_tensor, predicted, seg_tensor, samples_saved, output_dir)
                samples_saved += 1


    num_samples = len(test_loader)
    avg_dice_score = dice_total / (4 * num_samples)
    avg_iou = iou_total / (4 * num_samples)

    # Compute the average Dice score and IoU for each class over all samples
    class_names = ['Background', 'WT', 'TC', 'ET']
    for c in range(4):
        dice_per_class[c] /= num_samples
        iou_per_class[c] /= num_samples
        print(f"Class {class_names[c]} - Dice Score: {dice_per_class[c]:.4f}, IoU: {iou_per_class[c]:.4f}")

    print(f'Test Results - Avg Dice: {avg_dice_score:.4f}, Avg IoU: {avg_iou:.4f}')
    
    # After evaluation, plot the mean metrics per slice
    #plot_mean_metrics_per_slice(counts_per_slice, total_loss_per_slice, total_tumor_per_slice)
    
                            

def save_sample_outputs(data, predicted, target, sample_idx, output_dir):
    """
    Save plots of input images, predicted masks, and ground truth masks.

    Args:
    - data: Input image tensor at original resolution (shape: [4, D, H, W])
    - predicted: Predicted mask tensor at original resolution (shape: [4, D, H, W])
    - target: Ground truth mask tensor at original resolution (shape: [4, D, H, W])
    - sample_idx: Index of the sample in the dataset
    - output_dir: Directory to save the output images
    """
    class_names = ['Background', 'WT', 'TC', 'ET']
    class_colors = {
        0: (0, 0, 0),       # Background - Black
        1: (0, 255, 0),     # Whole Tumor - Green
        2: (255, 0, 0),     # Tumor Core - Red
        3: (0, 0, 255)      # Enhancing Tumor - Blue
    }

    # Convert tensors to numpy arrays
    data_np = data.cpu().numpy()        # Shape: [4, D, H, W]
    predicted_np = predicted.cpu().numpy()  # Shape: [4, D, H, W]
    target_np = target.cpu().numpy()    # Shape: [4, D, H, W]

    # Select a middle slice for visualization
    D = data_np.shape[1]
    mid_slice_idx = D // 2

    # Get the modality to visualize (e.g., FLAIR modality at index 3)
    modality_idx = 3  # Change this index if you want to visualize a different modality

    # Input image slice
    input_slice = data_np[modality_idx, mid_slice_idx, :, :]

    # Predicted masks
    predicted_masks = predicted_np[:, mid_slice_idx, :, :]  # Shape: [4, H, W]

    # Ground truth masks
    target_masks = target_np[:, mid_slice_idx, :, :]        # Shape: [4, H, W]

    # Create color overlays for predicted and ground truth masks
    pred_overlay = apply_color_mask(predicted_masks, class_colors)
    target_overlay = apply_color_mask(target_masks, class_colors)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(input_slice, cmap='gray')
    axs[0].set_title('Input Image (FLAIR, Mid-slice)')
    axs[0].axis('off')

    axs[1].imshow(input_slice, cmap='gray')
    axs[1].imshow(pred_overlay, alpha=0.5)
    axs[1].set_title('Predicted Mask Overlay')
    axs[1].axis('off')

    axs[2].imshow(input_slice, cmap='gray')
    axs[2].imshow(target_overlay, alpha=0.5)
    axs[2].set_title('Ground Truth Mask Overlay')
    axs[2].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"sample_{sample_idx}.png")
    plt.savefig(output_path)
    plt.close(fig)


    # Now, save individual mask channels
    for c in range(predicted_masks.shape[0]):
        class_name = class_names[c]
        class_color = class_colors.get(c, (255, 255, 255))

        # Predicted mask for class c
        pred_mask = predicted_masks[c]  # Shape: [H, W]
        # Ground truth mask for class c
        target_mask = target_masks[c]   # Shape: [H, W]

        # Create a color mask for predicted mask
        pred_color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        pred_color_mask[pred_mask > 0] = class_color

        # Create a color mask for ground truth mask
        target_color_mask = np.zeros((target_mask.shape[0], target_mask.shape[1], 3), dtype=np.uint8)
        target_color_mask[target_mask > 0] = class_color

        # Plot predicted mask
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(input_slice, cmap='gray')
        ax.imshow(pred_color_mask, alpha=0.5)
        ax.set_title(f'Predicted Mask - {class_name}')
        ax.axis('off')
        output_path = os.path.join(output_dir, f"sample_{sample_idx}_pred_{class_name}.png")
        plt.savefig(output_path)
        plt.close(fig)

        # Plot ground truth mask
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(input_slice, cmap='gray')
        ax.imshow(target_color_mask, alpha=0.5)
        ax.set_title(f'Ground Truth Mask - {class_name}')
        ax.axis('off')
        output_path = os.path.join(output_dir, f"sample_{sample_idx}_gt_{class_name}.png")
        plt.savefig(output_path)
        plt.close(fig) 


def apply_color_mask(masks, colors):
    """
    Apply color masks to the segmentation masks.

    Args:
    - masks: Segmentation masks (shape: [C, H, W])
    - colors: Dictionary mapping class indices to RGB color tuples

    Returns:
    - color_mask: RGB image with applied color masks (shape: [H, W, 3])
    """
    H, W = masks.shape[1], masks.shape[2]
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(masks.shape[0]):
        mask = masks[c]
        color = colors.get(c, (255, 255, 255))
        color_mask[mask > 0] = color
    return color_mask


if __name__ == "__main__":
    # Dataset path
    data_path = "/home/ubuntu/ismini/thesis/brats-2021/data/"  # Replace with your data path

    # Get test DataLoader
    _, _, test_loader = get_dataloaders(
        data_path,
        batch_size=1,  # Batch size 1 due to variable sequence lengths
        num_workers=2,
        pin_memory=True,
        use_saved_splits=True,
        split_dir='splits2'
    )

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainTumorPredictorTransformer(num_heads=4, num_layers=4)
    model.to(device)

    # Load the saved model checkpoint
    checkpoint_path = "/home/ubuntu/ismini/thesis/slicebyslice_model_checkpoints_splits2/checkpoint_epoch_100.pth"  # Replace with your model checkpoint path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test the model and save sample outputs
    test_model(model, test_loader, device, save_outputs=True,num_samples=0, output_dir='test_outputs_slicebyslice_model_all_classes_splits1_checking')
