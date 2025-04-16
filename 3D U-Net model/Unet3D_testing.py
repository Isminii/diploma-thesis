import os
import torch
from torch.utils.data import DataLoader
from Unet3D import UNet3D
from test_dataloader import get_test_dataloader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, test_loader, device, save_outputs=False, num_samples=20, output_dir='test_outputs'):
    model.eval()
    dice_total = 0.0
    iou_total = 0.0
    dice_per_class = [0.0] * 4  # Assuming 4 classes: Background, WT, TC, ET
    iou_per_class = [0.0] * 4

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
        samples_saved = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = batch  # data and target are at original resolution
            data = data.to(device)    # Shape: (1, 4, D, H, W)
            target = target.to(device)  # Shape: (1, 4, D, H, W)

            # Resize data to target_size for model input
            data_resized = F.interpolate(data, size=(128, 128, 128), mode='trilinear', align_corners=False)
            # data_resized shape: (1, 4, 128, 128, 128)

            # Run model
            outputs = model(data_resized)  # outputs shape: (1, 4, 128, 128, 128)

            # Resize outputs back to original size
            outputs_resized = F.interpolate(outputs, size=target.shape[2:], mode='trilinear', align_corners=False)
            # outputs_resized shape: (1, 4, D, H, W)

            outputs_sigmoid = torch.sigmoid(outputs_resized)
            predicted = (outputs_sigmoid > 0.5).float()

            # Compute Dice and IoU per class
            for c in range(4):  # For Background, WT, TC, ET
                pred_flat = predicted[:, c].contiguous().view(-1)
                target_flat = target[:, c].contiguous().view(-1)

                intersection = (pred_flat * target_flat).sum().item()
                dice_score = (2.0 * intersection + 1e-6) / (pred_flat.sum().item() + target_flat.sum().item() + 1e-6)
                union = pred_flat.sum().item() + target_flat.sum().item() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)

                dice_total += dice_score
                iou_total += iou

                dice_per_class[c] += dice_score
                iou_per_class[c] += iou

            # Save sample outputs if requested
            if save_outputs and samples_saved < num_samples:
                save_sample_outputs(data[0], predicted[0], target[0], idx, output_dir)
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
    path = "/home/ubuntu/ismini/thesis/brats-2021/data/"

    # Create test DataLoader
    test_loader = get_test_dataloader(
        path,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        split_dir='splits'
    )

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=4, base_channels=32, num_classes=4)
    model.to(device)

    # Load the saved model checkpoint
    checkpoint_path = "/home/ubuntu/ismini/thesis/Unet3D_checkpoints4/checkpoint_epoch_150.pth"  # Replace with your model checkpoint path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test the model and save sample outputs
    test_model(model, test_loader, device, save_outputs=True, num_samples=20, output_dir='test_outputs')
