import time
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from bigger_predictor3D_model import BrainTumorPredictorTransformer
from brats_dataloader_prepare_slices import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Directory to save checkpoints
checkpoint_dir = "slicebyslice_model_checkpoints_justcheck"
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to save sample outputs during validation
def plot_sample_outputs(model, val_loader, device, epoch, num_samples=5, save_dir="validation_outputs"):
    model.eval()
    samples_shown = 0

    # Define colors for each class
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
            mask = masks[c]
            color = colors.get(c, (255, 255, 255))  # Default to white if class not in colors
            color_mask[mask > 0] = color
        return color_mask

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if samples_shown >= num_samples:
                break

            # Unpack the batch
            image_tensor, seg_tensor, initial_mask_slice, selected_slice_idx = batch

            # Move data to device
            image_tensor = image_tensor.squeeze(0).to(device)  # Shape: (4, D, H, W)
            seg_tensor = seg_tensor.squeeze(0).to(device)      # Shape: (4, D, H, W)
            initial_mask_slice = initial_mask_slice.squeeze(0).to(device)  # Shape: (4, H, W)
            selected_slice_idx = selected_slice_idx.item()

            D = image_tensor.shape[1]
            front_indices = list(range(selected_slice_idx + 1, D))


            # Initialize previous mask
            prev_mask = initial_mask_slice

            # Collect outputs for visualization
            outputs_list = []

            for slice_idx in front_indices:
                current_image_slice = image_tensor[:, slice_idx, :, :]
                target_mask_slice = seg_tensor[:, slice_idx, :, :]

                # Forward pass
                next_mask_pred = model(current_image_slice, prev_mask)

                # Apply sigmoid to get probabilities
                outputs_sigmoid = torch.sigmoid(next_mask_pred)

                # Threshold to get binary masks
                predicted = (outputs_sigmoid > 0.5).cpu().numpy()  # Shape: (4, H, W)
                target_mask = target_mask_slice.cpu().numpy()

                # For visualization, select a few slices
                if slice_idx in [selected_slice_idx, D // 2]:
                    input_slice = image_tensor[3, slice_idx, :, :].cpu().numpy()
                    pred_colored = apply_color_mask(predicted, class_colors)
                    target_colored = apply_color_mask(target_mask, class_colors)

                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                    axs[0].imshow(input_slice, cmap='gray')
                    axs[0].set_title(f'Input Image Slice {slice_idx} (FLAIR)')
                    axs[0].axis('off')

                    axs[1].imshow(input_slice, cmap='gray')
                    axs[1].imshow(pred_colored, alpha=0.5)
                    axs[1].set_title('Predicted Mask Overlay')
                    axs[1].axis('off')

                    axs[2].imshow(input_slice, cmap='gray')
                    axs[2].imshow(target_colored, alpha=0.5)
                    axs[2].set_title('Ground Truth Mask Overlay')
                    axs[2].axis('off')

                    plt.tight_layout()
                    output_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{samples_shown + 1}_slice_{slice_idx}.png")
                    plt.savefig(output_path)
                    plt.close(fig)

                    print(f"Saved {output_path}")

                # Update prev_mask
                prev_mask = next_mask_pred.detach()

            samples_shown += 1

    print(f"Saved {samples_shown} sample outputs to {save_dir}")
    
# Define the loss functions
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.reshape(-1)  # Use .reshape() instead of .view()
    target_flat = target.reshape(-1)  # Use .reshape() instead of .view()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice_score = (2 * intersection + smooth) / (union + smooth)
    dice_loss_value = 1 - dice_score
    return dice_loss_value

def combined_loss(pred, target):
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)
    dice_loss_value = dice_loss(pred, target)
    return  bce_loss + dice_loss_value

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
    epochs = range(5, num_epochs + 1, 5)

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
    plt.savefig("training_metrics_3Dpredictor_splits2.png")

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
    data_path = "/home/ubuntu/ismini/thesis/brats-2021/data/"  # Replace with your data path

    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        batch_size=1,  # Batch size 1 due to variable sequence lengths
        num_workers=2,
        pin_memory=True,
        use_saved_splits=True,
        split_dir='splits2'
    )

    # Training settings
    num_epochs = 100
    warmup_steps = 100
    num_steps = num_epochs * len(train_loader)

    # Initialize model and optimizer
    model = BrainTumorPredictorTransformer(num_heads=4, num_layers=4)

    num_params, num_bytes = model_memory_info(model)
    print(f"Model parameters: {num_params}, Memory required: {num_bytes:.2f} MB")


    optimizer = optim.Adam(params=model.parameters(), lr=0.00001, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ####################################################################################

    def check_model_weights_for_nan_inf(model, filename):
        """
        Iterates through each layer of the model and checks for NaN or Inf in the weights.
        Saves the stats to a specified file instead of printing them.
        
        Args:
        - model: The neural network model to inspect.
        - filename: The file to save the results to.
        """
        with open(filename, 'w') as file:
            file.write("Checking model weights for NaN/Inf values...\n\n")
            
            for name, module in model.named_modules():
                print(filename)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    has_nan = weight.isnan().any().item()
                    has_inf = weight.isinf().any().item()
                    
                    if has_nan or has_inf:
                        file.write(f"Module: {name}\n")
                        file.write(f" - NaN found: {has_nan}\n")
                        file.write(f" - Inf found: {has_inf}\n")
                        file.write(f" - Weight tensor stats -> Min: {weight.min().item()}, "
                                f"Max: {weight.max().item()}, Mean: {weight.mean().item()}\n\n")
            
            file.write("Check complete.\n")

    

    def save_gradient_stats(model, filename):
        """
        Calculates statistics for gradients of all model parameters and saves them to a file.
        
        Args:
        - model: The neural network model to inspect.
        - filename: The file to save the gradient statistics to.
        """
        with open(filename, 'w') as file:
            file.write("Gradient Statistics Before Optimizer Step\n")
            file.write("=" * 50 + "\n\n")
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    has_nan = grad.isnan().any().item()
                    has_inf = grad.isinf().any().item()
                    
                    file.write(f"Parameter: {name}\n")
                    file.write(f" - Gradient Min: {grad.min().item()}\n")
                    file.write(f" - Gradient Max: {grad.max().item()}\n")
                    file.write(f" - Gradient Mean: {grad.mean().item()}\n")
                    file.write(f" - Contains NaN: {has_nan}\n")
                    file.write(f" - Contains Inf: {has_inf}\n\n")
                else:
                    file.write(f"Parameter: {name}\n")
                    file.write(" - No gradient available (probably a frozen parameter)\n\n")
            
            file.write("=" * 50 + "\n")
            file.write("Check complete.\n")



    def log_tensor_stats(tensor, name):

        if not isinstance(tensor, torch.Tensor):
            return

        if tensor is not None:
            print(f"{name} - Min: {tensor.min().item()}, Max: {tensor.max().item()}, "
                f"Mean: {tensor.mean().item()}, Contains NaN: {tensor.isnan().any().item()}")
        else:
            print(f"{name} is None")

    def forward_hook(module, input, output):
        print(f"Layer: {module.__class__.__name__}")
        
        # Log input tensor stats
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                log_tensor_stats(inp, f"Input {i}")
        else:
            log_tensor_stats(input, "Input")
        
        # Log module weights stats (if applicable)
        if hasattr(module, 'weight') and module.weight is not None:
            log_tensor_stats(module.weight, "Weight")
        
        # Log output tensor stats
        log_tensor_stats(output, "Output")
        print("-" * 50)

    #hooks = []
    #for layer in model.modules():
     #   if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and layer != model:
      #      hooks.append(layer.register_forward_hook(forward_hook))

    ####################################################################################


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
            # Unpack the batch
            image_tensor, seg_tensor, initial_mask_slice, selected_slice_idx = batch
            
            # Move data to device
            image_tensor = image_tensor.squeeze(0).to(device)
            seg_tensor = seg_tensor.squeeze(0).to(device)
            initial_mask_slice = initial_mask_slice.squeeze(0).to(device)
            selected_slice_idx = selected_slice_idx.item()

            

            # Prepare the indices for forward and backward directions
            D = image_tensor.shape[1]
            front_indices = list(range(selected_slice_idx + 1, D))
            back_indices = list(range(selected_slice_idx - 1, -1, -1))


            prev_mask = initial_mask_slice
            for idx in front_indices:
                current_image_slice = image_tensor[:, idx, :, :]
                target_mask_slice = seg_tensor[:, idx, :, :]
            

                # Forward pass
                next_mask_pred = model(current_image_slice, prev_mask)
    
                # Compute loss
                loss = combined_loss(next_mask_pred, target_mask_slice)
            
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                #save_gradient_stats(model, "gradient_stats.txt")

                # Apply gradient clipping to prevent NaNs from gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                #check_model_weights_for_nan_inf(model, "/home/ubuntu/pre_opt_step.txt")

                optimizer.step()
            
                #check_model_weights_for_nan_inf(model, "/home/ubuntu/post_opt_step.txt")

                # Update prev_mask for the next iteration
                prev_mask = next_mask_pred.detach()
                
                running_loss += loss.item()
            
            # Reset previous mask to initial for backward processing
            prev_mask = initial_mask_slice

            # Process backward slices
            for idx in back_indices:
                current_image_slice = image_tensor[:, idx, :, :]
                target_mask_slice = seg_tensor[:, idx, :, :]
                
                # Forward pass
                next_mask_pred = model(current_image_slice, prev_mask)
                
                # Compute loss
                loss = combined_loss(next_mask_pred, target_mask_slice)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                # Optional: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Detach next_mask_pred
                prev_mask = next_mask_pred.detach()
                
                running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / (len(train_loader)*(len(front_indices)+len(back_indices)))
        epoch_time = time.time() - start_time

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} sec')

        if epoch % 5 == 0:
            train_losses.append(epoch_loss)

            # Validation loop
            model.eval()
            val_loss = 0.0
            dice_total = 0.0
            iou_total = 0.0
            dice_per_class = [0.0] * 4  # Initialize per-class Dice
            iou_per_class = [0.0] * 4   # Initialize per-class IoU

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Validation]", leave=False):
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

                    loss = combined_loss(full_mask, seg_tensor)
                    val_loss += loss.item() 
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

            # Compute average losses and metrics
            num_samples = len(val_loader)
            val_loss /= num_samples
            avg_dice_score = dice_total / (4 * num_samples)
            avg_iou = iou_total / (4 * num_samples)

            val_losses.append(val_loss)
            val_dice_scores.append(avg_dice_score)
            val_iou_scores.append(avg_iou)

            # Compute the average Dice score and IoU for each class over all batches
            class_names = ['Background', 'WT', 'TC', 'ET']
            for c in range(4):
                dice_per_class[c] /= num_samples
                iou_per_class[c] /= num_samples
                print(f"Class {class_names[c]} - Dice Score: {dice_per_class[c]:.4f}, IoU: {iou_per_class[c]:.4f}")

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, avg_dice_score, avg_iou, checkpoint_path)
            print(f'Validation Loss: {val_loss:.4f}, Avg Dice: {avg_dice_score:.4f}, Avg IoU: {avg_iou:.4f}')

            

    # Plot metrics
    plot_metrics(train_losses, val_losses, val_dice_scores, val_iou_scores, num_epochs)
    # Call the function to plot sample outputs
    plot_sample_outputs(model, val_loader, device, epoch, num_samples=20, save_dir="validation_outputs_splits2")





