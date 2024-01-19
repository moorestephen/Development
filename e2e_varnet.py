from torch import nn
import torch
import torch.nn.functional as F

def cmplx_to_re(batch):
    # Assumes batch has shape [batch, channels, xdim, ydim]
    b, c, x, y = batch.shape
    cp = torch.empty((b, c * 2, x, y))
    re, im = torch.real(batch), torch.imag(batch)
    cp[:, ::2, :, :] = re
    cp[:, 1::2, :, :] = im

    return cp

def re_to_cmplx(batch):
    batch = batch[:, ::2, :, :] +1j * batch[:, 1::2, :, :]

    return batch

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
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, stored):
        x = self.conv_transpose(x)
        x = torch.add(x, stored)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        
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
                                        nn.ReLU()) # Change as currently downsamples
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
        # Assumes input dimensions [batch, channels, xdim, ydim]
        xdim_s = input.shape[2]
        ydim_s = input.shape[3]
        to_pad_xdim = (2 ** self.depth) - (xdim_s % (2 ** self.depth)) 
        to_pad_ydim = (2 ** self.depth) - (ydim_s % (2 ** self.depth))

        to_pad = (0, to_pad_ydim, 0, to_pad_xdim)
        padded = F.pad(input, to_pad, "constant", 0) # Currently zero-padding, could consider reflection though
        return padded, xdim_s, ydim_s
    
    def unpad(self, input, xdim, ydim):
        # Assumes input dimensions [batch, channels, xdim, ydim]
        return input[:, :, :xdim, :ydim]

    def forward(self, input : torch.tensor):
        # Assumes input dimensions [batch, channels, xdim, ydim], where channels = 1 and xdim and ydim are complex-valued
        
        padded, xdim_s, ydim_s = self.pad(input)

        padded = cmplx_to_re(padded)
        
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

        out = re_to_cmplx(out)

        out = self.unpad(out, xdim_s, ydim_s)

        return out

class Refinement(nn.Module):
    '''
    Refinement module from E2E-VarNet
    '''

    def __init__(
        self,
    ):
        '''
        Arguments:
        ----------

        '''
        super().__init__()

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

        # Assumes undersampled, unnormalized input of [batch, channels, xdim, ydim]
        # Assumes sens_maps of dimension [batch, channels, xdim, ydim]

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
        unet_pass = self.unet(reduced)

        # Expand
        expanded = sens_maps * unet_pass

        return expanded
    
class SME(nn.Module):
    '''
    Sensitivity Map Estimation for E2E-VarNet
    '''

    def __init__(
        self
    ):
        super().__init__()

        self.unet = UNet(8, 24, 24)
    
    def forward(self, input):

        b, c, xd, yd = input.shape

        # ACS Mask
        acs_mask = torch.zeros((b, c, xd, yd), dtype = torch.bool)
        zeros = torch.zeros((b, c, xd, yd), dtype = torch.complex64)
        square_size = 20
        start_x = (xd - square_size) // 2
        acs_mask[:, :, start_x:start_x + square_size, :] = True 
        masked = torch.where(acs_mask, input, zeros)

        # Shift
        shifted = torch.fft.ifftshift(masked, dim = (2, 3))

        # IFFT transform
        transformed = torch.fft.ifft2(shifted, dim = (2, 3))

        # U-Net
        unet_pass = self.unet(transformed)
        # unet_pass should have dimensions [batch, channels = 12, xdim, ydim]

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

    def forward(self, current : torch.Tensor, ref : torch.Tensor):
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
        mask = torch.zeros_like(combined, dtype = torch.bool)
        square_size = 20
        start_x = (combined.shape[2] - square_size) // 2
        mask[:, :, start_x:start_x + square_size, :] = True 
        masked = torch.where(mask, combined, zeros)
        return masked 

class E2EVarNet(nn.Module):
    '''
    E2E-VarNet Implementation
    '''

    def __init__(
        self
    ):
        super().__init__()

        self.refinement1 = Refinement()
        self.dc1 = DC()
        self.refinement2 = Refinement()
        self.dc2 = DC()
        self.refinement3 = Refinement()
        self.dc3 = DC()
        self.refinement4 = Refinement()
        self.dc4 = DC()
        self.sme = SME()
        

    def forward(self, input, mask : torch.Tensor):
        # Input assumed to be [batch, channels = 12, xdim, ydim] 

        # Generate estimated sensitivity maps
        sens_map_est = self.sme(input)

        # Copy reference k-space data for data consistency
        ref = input
        current_kspace = input

        # Block 1
        current_kspace = current_kspace - self.dc1(current_kspace, ref) + self.refinement1(current_kspace, sens_map_est)

        # Block 2
        current_kspace = current_kspace - self.dc2(current_kspace, ref) + self.refinement2(current_kspace, sens_map_est)

        # Block 3
        current_kspace = current_kspace - self.dc3(current_kspace, ref) + self.refinement3(current_kspace, sens_map_est)

        # Block 4
        current_kspace = current_kspace - self.dc4(current_kspace, ref) + self.refinement4(current_kspace, sens_map_est)

        # Shift
        shifted = torch.fft.ifftshift(current_kspace, dim = (2, 3))

        # IFFT transform
        transformed = torch.fft.ifft2(shifted, dim = (2, 3))

        # RSS (to generate final reconstructed images)
        rss = torch.sqrt(torch.sum(transformed ** 2, dim = 1, keepdim = True))

        return rss

# Define the dimensions of the fake data
batch_size = 3
num_channels = 12
xdim = 160
ydim = 218

# Create fake input data
fake_input = torch.randn((batch_size, num_channels, xdim, ydim))

# Create a mask tensor
mask = torch.randint(0, 2, (batch_size, num_channels, xdim, ydim), dtype=torch.bool)

# Instantiate your E2EVarNet model
test_model = E2EVarNet()

# Forward pass with the fake data
output = test_model(fake_input, mask)

# Print the output shape
print("Output shape:", output.shape)

