import time
import sys
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from diffusers.optimization import get_cosine_schedule_with_warmup
from Unet3D import UNet3D
from brats_dataloader import get_dataloaders

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Directory to save the worst validation models
worst_models_dir = "worst_validation_models"
os.makedirs(worst_models_dir, exist_ok=True)

worst_validation_losses = []

# Keep track of the saved worst models' filenames
saved_model_paths = []

# Define the log file path
worst_log_file = os.path.join(worst_models_dir, "worst_losses_log.txt")

def save_worst_loss_info(model, epoch, val_loss, dice_per_class, iou_per_class, worst_validation_losses, n=3):

    # Update class names for logging
    class_names = ['Background', 'WT', 'TC', 'ET']

    # Store the current information including per-class Dice and IoU scores
    current_info = {
        'epoch': epoch,
        'val_loss': val_loss,
        'dice_per_class': dice_per_class,  # List of Dice scores for each class
        'iou_per_class': iou_per_class,    # List of IoU scores for each class
        'model_state_dict': model.state_dict()
    }

    # Check if the list has fewer than 'n' worst losses
    if len(worst_validation_losses) < n:
        worst_validation_losses.append(current_info)
    else:
        # Find the best (smallest) validation loss in the worst list
        worst_in_list = min(worst_validation_losses, key=lambda x: x['val_loss'])
        
        # If the current loss is worse than the best loss in the worst list, replace it
        if val_loss > worst_in_list['val_loss']:
            worst_validation_losses.remove(worst_in_list)
            worst_validation_losses.append(current_info)

    # Sort the worst losses by validation loss in descending order
    worst_validation_losses.sort(key=lambda x: x['val_loss'], reverse=True)

    # Save the worst models only if they're part of the top `n` worst models
    worst_model_path = os.path.join(worst_models_dir, f"worst_model_epoch_{epoch}_loss_{val_loss:.4f}.pth")
    
    # Save model if this is one of the worst losses
    if worst_model_path not in saved_model_paths:
        torch.save(model.state_dict(), worst_model_path)
        saved_model_paths.append(worst_model_path)
        print(f"Saved worst model with validation loss: {val_loss:.4f} at epoch {epoch}")
    
    # Keep only the `n` worst models in the list
    if len(saved_model_paths) > n:
        # Remove the model associated with the best loss in the worst list
        worst_to_remove = saved_model_paths.pop(0)  # Remove the first (best) model
        if os.path.exists(worst_to_remove):
            os.remove(worst_to_remove)
            print(f"Removed model: {worst_to_remove}")

    # Write the top `n` worst models to the log file
    with open(worst_log_file, "w") as log_file:  # 'w' mode clears the file before writing
        for i, info in enumerate(worst_validation_losses):
            log_file.write(f"Epoch: {info['epoch']}, Validation Loss: {info['val_loss']:.4f}\n")
            for j, (dice, iou) in enumerate(zip(info['dice_per_class'], info['iou_per_class'])):
                class_name = ['Background', 'WT', 'TC', 'ET'][j]
                log_file.write(f"  Class {class_name} - Dice: {dice:.4f}, IoU: {iou:.4f}\n")
            log_file.write("=" * 40 + "\n")

# Define the loss functions
def dice_loss(pred, target, smooth=1e-6):
    # Apply sigmoid to predictions
    pred = torch.sigmoid(pred)
    # Flatten the tensors to (B, C, -1)
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    # Compute intersection and union per class
    intersection = (pred_flat * target_flat).sum(2)
    union = pred_flat.sum(2) + target_flat.sum(2)
    dice_score = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score.mean()
    return dice_loss

def combined_loss(pred, target):
    # pred: (B, 4, D, H, W)
    # target: (B, 4, D, H, W)
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)
    dice_loss_value = dice_loss(pred, target)
    return bce_loss + dice_loss_value

# Function to save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, dice_score, iou_score, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'dice_score': dice_score,
        'iou_score': iou_score,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with validation loss {val_loss:.4f}, "
          f"Dice: {dice_score:.4f}, IoU: {iou_score:.4f}.")

# Function to plot metrics
def plot_metrics(train_losses, val_losses, val_dice_scores, val_iou_scores, num_epochs):
    epochs = range(5, num_epochs + 1,5)

    plt.figure(figsize=(16, 12))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot validation loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    # Plot validation Dice score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_dice_scores, 'g', label='Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score')
    plt.legend()

    # Plot validation IoU score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_iou_scores, 'm', label='Validation IoU Score')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.title('Validation IoU Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")


def plot_sample_outputs(model, val_loader, device, num_samples=20,save_dir="output_samplings"):
    model.eval()
    samples_shown = 0

    class_colors = {
        0: (0, 0, 0),       # Background - Black
        1: (0, 255, 0),     # Whole tumor - Green
        2: (255, 0, 0),     # Tumor core - Red
        3: (0, 0, 255)      # Enhancing tumor - Blue
    }

    def apply_color_mask(masks, colors):
        # masks: (C, H, W)
        color_mask = np.zeros((*masks.shape[1:], 3), dtype=np.uint8)
        for c in range(masks.shape[0]):
            color_mask[masks[c] > 0] = colors[c]
        return color_mask

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            outputs_resized = F.interpolate(outputs, size=target.shape[2:], mode='trilinear', align_corners=False)
            outputs_sigmoid = torch.sigmoid(outputs_resized)
            predicted = (outputs_sigmoid > 0.5).cpu().numpy()
            target = target.cpu().numpy()

            for i in range(data.size(0)):
                if samples_shown >= num_samples:
                    return

                mid_slice_idx = data.shape[2] // 2

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(data[i, 0, mid_slice_idx, :, :].cpu().numpy(), cmap='gray')
                axs[0].set_title('Input Image (Mid-slice)')
                axs[0].axis('off')

                pred_colored = apply_color_mask(predicted[i, :, mid_slice_idx, :, :], class_colors)
                axs[1].imshow(pred_colored)
                axs[1].set_title('Predicted Segmentation (Mid-slice)')
                axs[1].axis('off')

                target_colored = apply_color_mask(target[i, :, mid_slice_idx, :, :], class_colors)
                axs[2].imshow(target_colored)
                axs[2].set_title('Ground Truth (Mid-slice)')
                axs[2].axis('off')

                # Save the figure
                output_path = os.path.join(save_dir, f"sample_{samples_shown + 1}.png")
                plt.savefig(output_path)
                plt.close(fig)

                print(f"Saved {output_path}")

                samples_shown += 1


def model_memory_info(model):
    """
    Calculate the number of parameters and memory required to store them in a model.

    Args:
    - model: PyTorch model

    Returns:
    - num_params: Total number of parameters in the model
    - memory_mb: Total memory required in megabytes (MB)
    """
    num_params = sum(p.numel() for p in model.parameters())  # Total number of parameters
    memory_bytes = num_params * 4  # Assuming float32 (4 bytes)
    memory_mb = memory_bytes / (1024 ** 2)  # Convert to MB

    return num_params, memory_mb

if __name__ == "__main__":


    # Dataset path
    path = "/home/ubuntu/ismini/thesis/brats-2021/data/"
    
    train_loader, val_loader, test_loader = get_dataloaders(
        path,
        batch_size=2,
        train_split=0.8,
        val_split=0.1,
        num_workers=2,
        pin_memory=True, use_saved_splits=True, split_dir='splits'
    )

    # Training settings
    num_epochs = 150
    warmup_steps=100
    num_steps=num_epochs*len(train_loader)

    # Initialize model and optimizer
    model = UNet3D(in_channels=4, base_channels=32, num_classes=4)

    num_params, num_bytes = model_memory_info(model)
    print(f"Model parameters: {num_params}, Memory required: {num_bytes:.2f} MB")


    
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    scheduler = get_cosine_schedule_with_warmup( optimizer = optimizer,num_warmup_steps = warmup_steps,num_training_steps = num_steps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    additional_info = {
        'batch_size': 2,
        'learning_rate': 0.001,
        'num_epochs': 150,
        'description': 'Trained with Adam optimizer, CE loss+Dice loss and cosine scheduler with warmup, warmup_steps=100.'
    }
    
    checkpoint_dir = "Unet3D_checkpoints5"
    os.makedirs(checkpoint_dir, exist_ok=True)  
   

    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]", leave=False):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            outputs_resized = F.interpolate(outputs, size=target.shape[2:], mode='trilinear', align_corners=False)
            loss = combined_loss(outputs_resized, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)

        if epoch % 5 == 0:
            train_losses.append(epoch_loss)
        
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} sec')

        if epoch % 5 == 0:
            # Validation loop
            model.eval()
            val_loss = 0.0
            dice_total = 0.0
            iou_total = 0.0
            dice_per_class = [0.0] * 4  # Initialize per-class Dice
            iou_per_class = [0.0] * 4   # Initialize per-class IoU
  
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Validation]", leave=False):
                    data, target = batch
                    data = data.to(device)
                    target = target.to(device)

                    outputs = model(data)
                    outputs_resized = F.interpolate(outputs, size=target.shape[2:], mode='trilinear', align_corners=False)
                    loss = combined_loss(outputs_resized, target)
                    val_loss += loss.item() * data.size(0)

                    # Apply sigmoid and threshold
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

            val_loss /= len(val_loader)
            avg_dice_score = dice_total / (4 * len(val_loader))  # Average over classes and batches
            avg_iou = iou_total / (4 * len(val_loader))  # Average over classes and batches

            val_losses.append(val_loss)  # Store validation loss
            val_dice_scores.append(avg_dice_score)  # Store Dice score
            val_iou_scores.append(avg_iou)  # Store IoU score

            # Compute the average Dice score and IoU for each class over all batches
            class_names = ['Background', 'WT', 'TC', 'ET']
            for c in range(4):
                dice_per_class[c] /= len(val_loader)
                iou_per_class[c] /= len(val_loader)
                print(f"Class {class_names[c]} - Dice Score: {dice_per_class[c]:.4f}, IoU: {iou_per_class[c]:.4f}")

            save_worst_loss_info(model, epoch, val_loss, dice_per_class, iou_per_class, worst_validation_losses, n=3)
            print(f'Validation Loss: {val_loss:.4f}, Avg Dice: {avg_dice_score:.4f}, Avg IoU: {avg_iou:.4f}')
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, avg_dice_score, avg_iou, checkpoint_path)

    plot_metrics(train_losses, val_losses, val_dice_scores, val_iou_scores, num_epochs)
    plot_sample_outputs(model, val_loader, device, num_samples=20)



