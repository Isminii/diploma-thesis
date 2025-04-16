import torch
import torch.nn as nn

# Custom weight initialization function for consistency
def initialize_weights(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        # Initialize convolutional and transposed convolutional layers
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        # Initialize linear layers
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.InstanceNorm3d) or isinstance(module, nn.BatchNorm3d):
        # Initialize normalization layers if they have weights and biases
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        # Initialize layer normalization layers
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# 3D U-Net Encoder
class UNetEncoder3D(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.ReLU(),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.ReLU(),
            nn.Conv3d(base_channels*4, base_channels*4, 3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, 3, padding=1),
            nn.InstanceNorm3d(base_channels*8),
            nn.ReLU(),
            nn.Conv3d(base_channels*8, base_channels*8, 3, padding=1),
            nn.InstanceNorm3d(base_channels*8),
            nn.ReLU()
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc1_pooled = self.pool(enc1_out)
        enc2_out = self.enc2(enc1_pooled)
        enc2_pooled = self.pool(enc2_out)
        enc3_out = self.enc3(enc2_pooled)
        enc3_pooled = self.pool(enc3_out)
        enc4_out = self.enc4(enc3_pooled)
        return enc4_out, enc3_out, enc2_out, enc1_out

# Modified Decoder for 3D U-Net with similar structure to TransUNet
class UNetDecoder3D(nn.Module):
    def __init__(self, base_channels, num_classes):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.InstanceNorm3d(base_channels * 8),
            nn.ReLU(),
            nn.Conv3d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.InstanceNorm3d(base_channels * 8),
            nn.ReLU()
        )

        self.upconv4 = nn.ConvTranspose3d(base_channels * 8, base_channels * 8, 2, stride=2)
        self.concat_conv4 = nn.Conv3d(base_channels * 8 + base_channels * 4, base_channels * 4, kernel_size=3, padding=1)
        self.dec4 = nn.Sequential(
            nn.InstanceNorm3d(base_channels * 4),
            nn.ReLU(),
            nn.Conv3d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.InstanceNorm3d(base_channels * 4),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose3d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.concat_conv3 = nn.Conv3d(base_channels * 4 + base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.dec3 = nn.Sequential(
            nn.InstanceNorm3d(base_channels * 2),
            nn.ReLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.InstanceNorm3d(base_channels * 2),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose3d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.concat_conv2 = nn.Conv3d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1)
        self.dec2 = nn.Sequential(
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Conv3d(base_channels, num_classes, 1)

    def forward(self, x, enc3_out, enc2_out, enc1_out):
        # Use the same initial conv layer as TransUNet decoder
        x = self.init_conv(x)

        x = self.upconv4(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.concat_conv4(x)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.concat_conv3(x)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.concat_conv2(x)
        x = self.dec2(x)

        return self.final_conv(x)

# 3D U-Net Model with Updated Decoder
class UNet3D(nn.Module):
    def __init__(self, in_channels, base_channels, num_classes):
        super().__init__()
        self.encoder = UNetEncoder3D(in_channels, base_channels)
        self.decoder = UNetDecoder3D(base_channels, num_classes)
        self.apply(initialize_weights)

    def forward(self, x):
        enc4_out, enc3_out, enc2_out, enc1_out = self.encoder(x)
        return self.decoder(enc4_out, enc3_out, enc2_out, enc1_out)

