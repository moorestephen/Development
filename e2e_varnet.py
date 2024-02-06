from torch import nn
import torch
import torch.nn.functional as F
from mask import get_masked_indices
# import matplotlib.pyplot as plt
import numpy as np

def cmplx_to_re(batch, device):
    # Assumes batch has shape [batch, channels, height, width]
    b, c, h, w = batch.shape
    cp = torch.empty((b, c * 2, h, w))
    re, im = torch.real(batch), torch.imag(batch)
    cp[:, ::2, :, :] = re
    cp[:, 1::2, :, :] = im

    return cp.to(device)

def re_to_cmplx(batch, device):
    batch = batch[:, ::2, :, :] +1j * batch[:, 1::2, :, :]

    return batch.to(device)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        store = x
        x = self.maxpool(x)
        return x, store
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    
    def forward(self, x, stored):
        x = self.conv_transpose(x)
        x = torch.cat([x, stored], dim = 1) 
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x - self.relu(x)
        
        return x

class UNet(nn.Module):
    '''
    U-Net Model
    '''

    def __init__(
        self,
        chans : int,
        in_channels : int,
        out_channels : int,
    ):
        '''
        Arguments
        ---------
        chans : int
            Number of output channels in first convolutional layer
        in_channels : int
            Number of channels passed to the U-Net
        out_channels : int
            Number of channels outputted from the U-Net
        '''

        super().__init__()

        self.depth = 4 # For reference, will hopefully make this modifiable

        # Encoder:
        ch_cnt = chans
        self.enc1 = ConvBlock(in_channels, ch_cnt)
        self.enc2 = ConvBlock(ch_cnt, ch_cnt * 2)
        ch_cnt *= 2
        self.enc3 = ConvBlock(ch_cnt, ch_cnt * 2)
        ch_cnt *= 2
        self.enc4 = ConvBlock(ch_cnt, ch_cnt * 2)
        ch_cnt *= 2

        # Bottleneck:
        self.bottleneck = nn.Sequential(nn.Conv2d(ch_cnt, ch_cnt * 2, kernel_size = 3, padding = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(ch_cnt * 2, ch_cnt * 2, kernel_size = 3, padding = 1),
                                        nn.ReLU()) 
        ch_cnt *= 2

        # Decoder:
        self.dec4 = ConvTransposeBlock(ch_cnt, ch_cnt // 2)
        ch_cnt //= 2
        self.dec3 = ConvTransposeBlock(ch_cnt, ch_cnt // 2)
        ch_cnt //= 2
        self.dec2 = ConvTransposeBlock(ch_cnt, ch_cnt // 2)
        ch_cnt //= 2
        self.dec1 = ConvTransposeBlock(ch_cnt, ch_cnt // 2)
        ch_cnt //= 2

        self.final_conv = nn.Conv2d(ch_cnt, out_channels, kernel_size = 1, padding = 0)
    
    def pad(self, input):
        # Assumes input dimensions [batch, channels, height, width]
        height_dim_s = input.shape[2]
        width_dim_s = input.shape[3]
        to_pad_height = (2 ** self.depth) - (height_dim_s % (2 ** self.depth)) 
        to_pad_width = (2 ** self.depth) - (width_dim_s % (2 ** self.depth))

        to_pad = (0, to_pad_width, 0, to_pad_height)
        # DOuble check int vs complex zero-padding
        padded = F.pad(input, to_pad, "constant", 0) # Currently zero-padding, could consider reflection though
        return padded, height_dim_s, width_dim_s
    
    def unpad(self, input, height, width):
        # Assumes input dimensions [batch, channels, height, width]
        return input[:, :, :height, :width]

    def forward(self, input : torch.tensor, device):
        # Assumes input dimensions [batch, channels, height, width], where channels = 1 and height and width are complex-valued
        
        padded, hdim_s, wdim_s = self.pad(input)

        padded = cmplx_to_re(padded, device)

        for_scale = torch.max(torch.abs(padded))
        padded = padded / for_scale
        
        track, out1 = self.enc1(padded)
        track, out2 = self.enc2(track)
        track, out3 = self.enc3(track)
        track, out4 = self.enc4(track)

        track = self.bottleneck(track)
        track = self.dec4(track, out4)
        track = self.dec3(track, out3)
        track = self.dec2(track, out2)
        track = self.dec1(track, out1)

        out = self.final_conv(track)

        out = out * for_scale

        out = re_to_cmplx(out, device)

        out = self.unpad(out, hdim_s, wdim_s)

        return out

class Refinement(nn.Module):
    '''
    Refinement module from E2E-VarNet
    '''

    def __init__(
        self,
        device
    ):
        '''
        Arguments:
        ----------

        '''
        super().__init__()
        self.device = device
        self.unet = UNet(16, 2, 2)
    
    def forward(self, input, sens_maps):
        '''
        Parameters
        ----------
        input : torch.Tensor
            Current K-Space data
        sens_maps : torch.tensor
            Estimated sensitivity profiles
        '''

        # Assumes undersampled, unnormalized input of [batch, channels, height, width]
        # Assumes sens_maps of dimension [batch, channels, height, width]

        # Shift
        shifted = torch.fft.ifftshift(input, dim = (2, 3))

        # IFFT transform
        transformed = torch.fft.ifft2(shifted, dim = (2, 3))

        # Reduce
        assert transformed.shape == sens_maps.shape # Double check shape
        sens_maps_conj = torch.conj(sens_maps)
        reduced = sens_maps_conj * transformed
        reduced = torch.sum(reduced, dim = 1, keepdim = True)

        # U-Net
        unet_pass = self.unet(reduced, self.device)

        # Expand
        expanded = sens_maps * unet_pass

        return expanded
    
class SME(nn.Module):
    '''
    Sensitivity Map Estimation for E2E-VarNet
    '''

    def __init__(
        self,
        device
    ):
        super().__init__()

        self.device = device
        self.unet = UNet(8, 24, 24)
    
    def forward(self, input):

        b, c, hd, wd = input.shape

        # ACS Mask
        acs_mask = torch.zeros((b, c, hd, wd), dtype = torch.bool).to(self.device)
        zeros = torch.zeros((b, c, hd, wd), dtype = torch.complex64).to(self.device)
        square_size = 20
        start_h = (hd - square_size) // 2
        start_w = (wd - square_size) // 2
        acs_mask[:, :, start_h:start_h + square_size, start_w:start_w + square_size] = True 
        masked = torch.where(acs_mask, input, zeros)

        # Shift
        shifted = torch.fft.ifftshift(masked, dim = (2, 3))

        # IFFT transform
        transformed = torch.fft.ifft2(shifted, dim = (2, 3))

        # U-Net
        unet_pass = self.unet(transformed, self.device)
        # unet_pass should have dimensions [batch, channels = 12, height, width]

        # Division by RSS (Normalization)
        rss = torch.sqrt(torch.sum(unet_pass ** 2, dim = 1, keepdim = True))
        normalized = unet_pass / rss

        return normalized
    
class DC(nn.Module):
    '''
    E2E-VarNet Data Consistency Block
    '''

    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, current : torch.Tensor, ref : torch.Tensor, us_mask : torch.Tensor):
        '''
        Parameters
        ----------
        current : torch.Tensor
            Current k-space data
        ref : torch.Tensor
            Reference k-space data
        '''

        combined = current - ref
        zeros = torch.zeros_like(combined, dtype = torch.complex64)
        masked = torch.where(us_mask, combined, zeros)
        masked *= self.weight
        return masked 

class E2EVarNet(nn.Module):
    '''
    E2E-VarNet Implementation
    '''

    def __init__(
        self,
        device
    ):
        super().__init__()

        self.device = device

        self.refinement1 = Refinement(self.device)
        self.dc1 = DC()
        self.refinement2 = Refinement(self.device)
        self.dc2 = DC()
        self.refinement3 = Refinement(self.device)
        self.dc3 = DC()
        self.refinement4 = Refinement(self.device)
        self.dc4 = DC()
        self.refinement5 = Refinement(self.device)
        self.dc5 = DC()
        self.refinement6 = Refinement(self.device)
        self.dc6 = DC()
        self.refinement7 = Refinement(self.device)
        self.dc7 = DC()
        self.refinement8 = Refinement(self.device)
        self.dc8 = DC()
        self.sme = SME(self.device)
        
    def apply_undersampling_mask(self, input):
        '''
        Applies the passed undersampling mask to input k-space
        '''

        b, c, h, w = input.shape

        mask = torch.zeros((b, c, h, w), dtype = torch.bool).to(self.device)
        zeros = torch.zeros((b, c, h, w), dtype = torch.complex64).to(self.device)
        square_size = 20
        start_h = (h - square_size) // 2
        start_w = (w - square_size) // 2
        mask[:, :, start_h:start_h + square_size, start_w:start_w + square_size] = True 
        mask[:, :, get_masked_indices(), 0:(int(w * 0.85))] = True
        masked = torch.where(mask, input, zeros)

        return masked, mask


    def forward(self, input):
        # Input assumed to be [batch, channels = 12, height, width] 

        # Undersample input
        input, us_mask = self.apply_undersampling_mask(input)

        # Generate estimated sensitivity maps
        sens_map_est = self.sme(input)

        # Copy reference k-space data for data consistency
        ref = input
        current_kspace = input

        # Block 1
        current_kspace = current_kspace - self.dc1(current_kspace, ref, us_mask) + self.refinement1(current_kspace, sens_map_est)

        # Block 2
        current_kspace = current_kspace - self.dc2(current_kspace, ref, us_mask) + self.refinement2(current_kspace, sens_map_est)

        # Block 3
        current_kspace = current_kspace - self.dc3(current_kspace, ref, us_mask) + self.refinement3(current_kspace, sens_map_est)

        # Block 4
        current_kspace = current_kspace - self.dc4(current_kspace, ref, us_mask) + self.refinement4(current_kspace, sens_map_est)

        # Block 5
        current_kspace = current_kspace - self.dc5(current_kspace, ref, us_mask) + self.refinement5(current_kspace, sens_map_est)

        # Block 6
        current_kspace = current_kspace - self.dc6(current_kspace, ref, us_mask) + self.refinement6(current_kspace, sens_map_est)

        # Block 7
        current_kspace = current_kspace - self.dc7(current_kspace, ref, us_mask) + self.refinement7(current_kspace, sens_map_est)

        # Block 8
        current_kspace = current_kspace - self.dc8(current_kspace, ref, us_mask) + self.refinement8(current_kspace, sens_map_est)

        # Shift
        shifted = torch.fft.ifftshift(current_kspace, dim = (2, 3))

        # IFFT transform
        transformed = torch.fft.ifft2(shifted, dim = (2, 3))

        # RSS (to generate final reconstructed images)
        rss = torch.sqrt(torch.sum(transformed ** 2, dim = 1, keepdim = True))

        mag = torch.abs(rss)

        # norm_mag = torch.nn.functional.normalize(mag, dim = (2, 3))
        norm_scale = torch.max(torch.abs(mag))
        norm_mag = mag / norm_scale

        return norm_mag