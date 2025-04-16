import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import sys
import os

# Custom LayerNorm with epsilon to prevent division by zero
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        return self.weight * x + self.bias

# Custom Transformer Block for 2D inputs with residual connections
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim) 
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x.clone()
        # Clamp inputs before LayerNorm
        #x = torch.clamp(x, min=-1e4, max=1e4)
        x = self.norm1(x)
        if torch.isnan(x).any():
            print("NaN detected after norm1")
            sys.exit()
        attn_out, _ = self.attn(x, x, x)
        if torch.isnan(attn_out).any():
            print("NaN detected in attention output")
            sys.exit()
        x = residual + attn_out  # Residual connection after attention
        # Clamp inputs before next LayerNorm
        #x = torch.clamp(x, min=-1e4, max=1e4)
        x = self.norm2(x)
        if torch.isnan(x).any():
            print("NaN detected after norm2")
            sys.exit()
        # Clamp inputs before feed-forward to prevent extreme values
        #x = torch.clamp(x, min=-1e4, max=1e4)
        ff_out = self.ff(x)
        if torch.isnan(ff_out).any():
            print("NaN detected in feed-forward output")
            sys.exit()
        return x + ff_out  # Residual connection after feed-forward

# Positional Encoding for variable-length sequences
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, num_patches, d_model = x.size()
        device = x.device

        pe = torch.zeros(num_patches, d_model, device=device)
        position = torch.arange(0, num_patches, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, num_patches, d_model)
        return x + pe

# Transformer Encoder for 2D data
class TransformerEncoder2D(nn.Module):
    def __init__(self, embed_dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # No additional residual connection needed

# Updated BrainTumorPredictorTransformer model
class BrainTumorPredictorTransformer(nn.Module):
    def __init__(self, img_size=240, in_channels=8, out_channels=4, base_channel=32,
                 num_heads=4, num_layers=4, d_model=256, patch_size=8):
        super(BrainTumorPredictorTransformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.base_channel = base_channel  # Added base_channel parameter
        
        # U-Net Encoder with 3 stages
        self.enc_conv1 = self.double_conv(in_channels, base_channel)
        self.pool1 = nn.MaxPool2d(2) 
        
        self.enc_conv2 = self.double_conv(base_channel, base_channel * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc_conv3 = self.double_conv(base_channel * 2, base_channel * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_conv4 = self.double_conv(base_channel * 4, base_channel * 8)
        
        # Adjusted scaling factor due to pooling layers
        #scale_factor = 8  # Since there are 3 pooling layers with kernel_size=2
        # Compute dimensions after pooling
        #bottleneck_dim = math.ceil(self.img_size / scale_factor)
        # Calculate the number of patches
        #self.bottleneck_patch_size = max(1, self.patch_size // scale_factor)
        #num_patches_per_dim = math.ceil(bottleneck_dim / self.bottleneck_patch_size)
        #self.num_patches = num_patches_per_dim ** 2
        # Adjusted for the encoder output channels (base_channel * 4)
        #self.patch_dim = (base_channel * 8) * self.bottleneck_patch_size * self.bottleneck_patch_size
        
        # Positional Encoding
        self.positional_embedding = PositionalEncoding(d_model)
        
        # Linear projection of flattened patches
        #self.patch_proj = nn.Linear(self.patch_dim, d_model)
        #self.inverse_patch_proj = nn.Linear(d_model, self.patch_dim)
        
        
        self.transformer_input_proj = nn.Sequential(
            nn.Conv2d(base_channel * 8, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )

        self.transformer_output_proj = nn.Sequential(
            nn.Conv2d(d_model, base_channel * 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel * 8, base_channel * 8, kernel_size=1)
        )
        

        # Custom Transformer Encoder
        self.transformer_encoder = TransformerEncoder2D(
            embed_dim=d_model,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=d_model * 2,
            dropout=0.1
        )
        
        # U-Net Decoder with 3 stages
        self.dec_conv4 = self.double_conv(8*base_channel, base_channel * 4)

        self.upconv3 = nn.ConvTranspose2d(base_channel * 4, base_channel * 4, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(base_channel * 4 + base_channel * 4, base_channel * 2)
        
        self.upconv2 = nn.ConvTranspose2d(base_channel * 2, base_channel * 2, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(base_channel * 2 + base_channel * 2, base_channel )
        
        self.upconv1 = nn.ConvTranspose2d(base_channel , base_channel, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(base_channel + base_channel, base_channel)
        
        self.final_conv = nn.Conv2d(base_channel, out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # padding=1 to maintain spatial dimensions
            nn.InstanceNorm2d(out_channels),  # Replaced BatchNorm2d with InstanceNorm2d
            nn.ReLU(),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # padding=1
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, CustomLayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
    
    def adjust_size(self, tensor, target_size):
        _, _, h, w = tensor.size()
        th, tw = target_size

        # Adjust height
        if h > th:
            x1 = (h - th) // 2
            x2 = x1 + th
            tensor = tensor[:, :, x1:x2, :]
        elif h < th:
            pad_top = (th - h) // 2
            pad_bottom = th - h - pad_top
            tensor = F.pad(tensor, (0, 0, pad_top, pad_bottom))
        
        # Adjust width
        if w > tw:
            y1 = (w - tw) // 2
            y2 = y1 + tw
            tensor = tensor[:, :, :, y1:y2]
        elif w < tw:
            pad_left = (tw - w) // 2
            pad_right = tw - w - pad_left
            tensor = F.pad(tensor, (pad_left, pad_right, 0, 0))
        
        return tensor
    
    def save_next_mask_pred(self, next_mask_pred, pos, direction):
        """
        Saves statistics and the next_mask_pred tensor to files.

        Args:
            next_mask_pred (torch.Tensor): The predicted mask for the next slice.
            pos (int): The relative position of the slice.
            direction (str): 'forward' or 'backward' indicating processing direction.
        """
        # Ensure the output directory exists
        output_dir = "next_mask_pred_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Prepare statistics
        stats = {
            'Position': pos,
            'Direction': direction,
            'Min': next_mask_pred.min().item(),
            'Max': next_mask_pred.max().item(),
            'Mean': next_mask_pred.mean().item(),
            'Std': next_mask_pred.std().item(),
            'NaNs': torch.isnan(next_mask_pred).sum().item(),
            'Infs': torch.isinf(next_mask_pred).sum().item(),
        }

        # Append statistics to a text file
        stats_file = os.path.join(output_dir, "next_mask_pred_stats.txt")
        with open(stats_file, 'a') as f:
            f.write(f"Slice Position: {pos}, Direction: {direction}\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

        # Save the next_mask_pred tensor
        tensor_file = os.path.join(output_dir, f"next_mask_pred_pos_{pos}_{direction}.pt")
        torch.save(next_mask_pred, tensor_file)




    def forward(self, current_image_slice, prev_mask):
        """
        Processes a single slice and returns the predicted mask for the next slice.

        Args:
            current_image_slice (torch.Tensor): Current image slice tensor of shape (C, H, W).
            prev_mask (torch.Tensor): Previous mask tensor of shape (C_mask, H, W).

        Returns:
            next_mask_pred (torch.Tensor): Predicted mask tensor for the next slice.
        """
        # Check for NaNs or Infs in current_image_slice and prev_mask
        if torch.isnan(current_image_slice).any() or torch.isinf(current_image_slice).any():
            print_stats("current_image_slice", current_image_slice)
            print("NaN or Inf detected in current_image_slice")
        if torch.isnan(prev_mask).any() or torch.isinf(prev_mask).any():
            print_stats("prev_mask", prev_mask)
            print("NaN or Inf detected in prev_mask")
            
        input_slice = torch.cat([current_image_slice, prev_mask], dim=0).unsqueeze(0)  # (B=1, in_channels, H, W)

        # Encoder
        x1 = self.enc_conv1(input_slice)  # (B, base_channel, H, W)
        x = self.pool1(x1)  # (B, base_channel, H/2, W/2)

        x2 = self.enc_conv2(x)  # (B, base_channel * 2, H/2, W/2)
        x = self.pool2(x2)  # (B, base_channel * 2, H/4, W/4)

        x3 = self.enc_conv3(x)  # (B, base_channel * 4, H/4, W/4)
        x = self.pool3(x3)  # (B, base_channel * 4, H/8, W/8)

        x4 = self.enc_conv4(x)   # (B, base_channel*8, H/8, W/8)

        # Prepare for Transformer
        transformer_input = self.transformer_input_proj(x4)
        B, C, H_p, W_p = transformer_input.shape
        x = rearrange(transformer_input, 'b c h w -> b (h w) c')

        # Add positional encoding
        x = self.positional_embedding(x)  # (B, num_patches, d_model)

        # Transformer Encoder
        transformer_output = self.transformer_encoder(x)  # (B, num_patches, d_model)

        # Reconstruct spatial dimensions
        x = rearrange(transformer_output, 'b (h w) c -> b c h w', h=H_p, w=W_p)
        x = self.transformer_output_proj(x)
        
        # Decoder with skip connections
        x = self.dec_conv4(x)
        x = self.upconv3(x)
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec_conv3(x)
        x = self.upconv2(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_conv2(x)
        x = self.upconv1(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_conv1(x)
        next_mask_pred = self.final_conv(x)  # (B, out_channels, H_out, W_out)

        # Ensure output mask has the same spatial dimensions as input
        H, W = current_image_slice.shape[1], current_image_slice.shape[2]
        if next_mask_pred.shape[2:] != (H, W):
            next_mask_pred = F.interpolate(next_mask_pred, size=(H, W), mode='bilinear', align_corners=False)

        next_mask_pred = next_mask_pred.squeeze(0)  # (out_channels, H, W)

        return next_mask_pred

# Helper function to print tensor statistics
def print_stats(name, tensor, pos=None):
    """
    Print statistics of a tensor, optionally including a slice identifier.

    Args:
        name (str): Name of the tensor (e.g., layer name).
        tensor (torch.Tensor): The tensor to analyze.
        pos (int, optional): The position or slice identifier. Defaults to None.
    """
    if pos is not None:
        print(f"Slice position: {pos}")
    print(f"{name}:")
    print(f"  Min: {tensor.min().item()}")
    print(f"  Max: {tensor.max().item()}")
    print(f"  Mean: {tensor.mean().item()}")
    print(f"  Std: {tensor.std().item()}")
    print(f"  NaNs: {torch.isnan(tensor).sum().item()} NaN values")
    print(f"  Infs: {torch.isinf(tensor).sum().item()} Inf values")
