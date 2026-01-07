
import numpy as np
import torch
import torch.nn as nn

from pytorch_wavelets import DWT1D, IDWT1D
from pytorch_wavelets import DWT, IDWT
from pytorch_wavelets import DTCWTForward, DTCWTInverse

torch.manual_seed(0)
np.random.seed(0)

# %%
""" Def: 1d Wavelet convolution layer """
class WIB(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='symmetric'):
        super(WIB, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.
        
        Input parameters:
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter (db)
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters:
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Approximate wavelet coefficients (low frequency components)
        self.weights2 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Detailed wavelet coefficients (High frequency components)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only")
        self.wavelet = wavelet 
        self.mode = mode
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode) 
        dummy_data = torch.randn( 1, 1, self.size ) # size corresponds to the size of the input vector (creates a dummy input data : (batch, channel, size)
        mode_data, _ = self.dwt_(dummy_data) # here,  mode_data -> Approximate coefficient (An) ;  _  ->  Higher coeffcients (D1, D2, D3,...Dn) components
        self.modes1 = mode_data.shape[-1]# number of waveleet coeffecients (modes) in low freq component of DWT signal
        
        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1)) # size of high freq comp. is same as that of low freq comp.
        
        

    # Convolution
    def mul1d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x ) 
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        return torch.einsum("bix,iox->box", input, weights) # element wise multiplication

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x]
        
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        
        # Adjusting the level of wavelet decomposition based on the input size
        # to maintain the same number of wavelet coefficients
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level+factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level-factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
            
            # x_coeff is a list of tensor while x_ft is a single tensor
            
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final low pass wavelet coefficients
        out_ft = self.mul1d(x_ft, self.weights1) # convolution-1
        # Multiply the final high pass wavelet coefficients
        out_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2) # convolution-2
    
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
        x = idwt((out_ft, out_coeff)) 
        return x

""" Def: 1d Wavelet convolutional encoder layer """
class WaveEncoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, down_level=1, mode='symmetric'):
        super(WaveEncoder1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only") 
        self.wavelet = wavelet
        self.mode = mode
        if down_level >= level:
            raise Exception('down_level must be smaller than level')
        else:
            self.down_level = down_level
        dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode) 
        dummy_data = torch.randn(1, 1, self.size )
        mode_data, _ = dwt_(dummy_data)
        self.modes = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes))

    # Convolution
    def mul1d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x ) 
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x]
        
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level+factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level-factor, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final low pass and high pass coefficients
        out_ft = self.mul1d(x_ft, self.weights1)
        out_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            x = idwt((out_ft, out_coeff[factor + self.down_level:])) 
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            x = idwt((out_ft, out_coeff[factor - self.down_level:])) 
            
        else:
            x = idwt((out_ft, out_coeff[self.down_level:]))
        return x
   
   
""" Def: Gate Network """
class Gate_context1d(nn.Module):
    def __init__(self, in_channels, out_channels, expert_num, label_lifting, size, level=2, wavelet='db1', down_level=1):
        super(Gate_context1d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.
        
        Input parameters:
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        expert_num   : scalar, number of local wavelet experts
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filters
        
        Output parameters:
        ------------------
        lambda : tensor, shape-[in_channels * out_channels * number of expert]
                  participation coefficients of local experts
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.level = level
        self.size = size 
        self.wavelet = wavelet
        self.down_level = down_level
        self.expert_num = expert_num
        self.label_lifting = label_lifting
        self.lifting_network = nn.Linear(1, self.label_lifting)
        self.wno_encode = WaveEncoder1d(self.in_channels, self.out_channels, self.level,
                                        self.size, self.wavelet, self.down_level)
        self.gate = nn.Sequential(
                    nn.Linear(self.size//2**(down_level) + self.label_lifting, 256),
                    nn.ELU(),
                    nn.Linear(256, 128),
                    nn.Mish(),
                    nn.Linear(128, 64),
                    nn.Mish(),
                    nn.Linear(64, 32),
                    nn.Mish(),
                    nn.Linear(32, self.expert_num),
                    nn.Softmax(dim=-1))
        
        # self.gate = nn.Sequential(
        #             nn.Linear(self.size//2**(down_level) + self.label_lifting, 256),
        #             nn.ELU(),
        #             nn.Linear(256, 128),
        #             nn.ELU(),
        #             nn.Linear(128, 64),
        #             nn.ELU(),
        #             nn.Linear(64, 32),
        #             nn.ELU(),
        #             nn.Linear(32, self.expert_num),
        #             nn.Softmax(dim=-1))
        
        
    def forward(self, x, label):
        lambda_0 = self.lifting_network(label)
        lambda_1 = self.wno_encode(x)

        lambda_ = self.gate( torch.cat((lambda_0,lambda_1), dim=-1) )
        return lambda_
