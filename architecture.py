# Imported from NKI-AI/direct, changed certain parts of the code to better reflect the KIKI-net paper 
# We can replace the 5-layer CNN with any image-to-image model 

import torch
import torch.nn as nn
from data import transforms as T
from LeWin_Model import LeT_transformer


class cnn_layer(nn.Module):
    
    def __init__(self,chans=32):
        super(cnn_layer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2,  chans, 3, padding=1, bias=True),
            nn.InstanceNorm2d(chans, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, 3, padding=1, bias=True),
            nn.InstanceNorm2d(chans, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, 3, padding=1, bias=True),
            nn.InstanceNorm2d(chans, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, 3, padding=1, bias=True),
            nn.InstanceNorm2d(chans, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, 2,  3, padding=1, bias=True)
        )  
       
    def forward(self, x):
        return self.conv(x) + x

    
class MultiCoil(nn.Module):
        
    def __init__(self, 
                 kspace_model,
                 coil_dim=1):
        """Inits :class:`MultiCoil`.
        Parameters
        ----------
        model: nn.Module
            Any nn.Module that takes as input with 4D data (N, H, W, C). Typically a convolutional-like model.
        coil_dim: int
            Coil dimension. Default: 1.
        """
        super(MultiCoil,self).__init__()
        self.model = kspace_model
        self.coil_dim = coil_dim
        
        
    def forward(self,
                x):
        x = x.clone()
        assert self.coil_dim == 1
        batch, coil, height, width, channels = x.size()
        x = x.reshape(batch * coil, height, width, channels).permute(0, 3, 1, 2).contiguous()
        x = self.model(x).permute(0, 2, 3, 1)
        x = x.reshape(batch, coil, height, width, -1)
        
        return x
    

def apply_dc(image,h,w,sensitivity,mask,masked_kspace):
    image = T.complex_img_pad(image,(h,w))
    image = T.complex_multiply(image[...,0].unsqueeze(1), image[...,1].unsqueeze(1), 
                        sensitivity[...,0], sensitivity[...,1])
    kspace = T.fft2(image)

    kspace = (1 - mask) * kspace + mask * masked_kspace
    image = T.ifft2(kspace)
    image = T.complex_multiply(image[...,0], image[...,1], 
                        sensitivity[...,0], 
                        -sensitivity[...,1]).sum(dim=1)
    return T.complex_center_crop(image,(320,320))

    
class KIKInet(nn.Module):
    """Based on KIKINet implementation [1]_. Modified to work with multi-coil k-space data.
    References
    ----------
    .. [1] Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing Undersampled Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp. 2188–201. PubMed, https://doi.org/10.1002/mrm.27201.
    """
    
    def __init__(self,
                 n_iter=1):
        """Inits :class:`KIKINet`.
        Parameters
        ----------
        image_model_architecture: str
            Image model architecture. Currently only implemented for MWCNN and (NORM)UNET. Default: 'MWCNN'.
        kspace_model_architecture: str
            Kspace model architecture. Currently only implemented for CONV and DIDN and (NORM)UNET. Default: 'DIDN'.
        num_iter: int
            Number of unrolled iterations.
        """
        super(KIKInet,self).__init__()
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)
        image_model = cnn_layer(chans=32)
        kspace_model = cnn_layer(chans=32)
        self.n_iter = n_iter
        
        self.kspace_model_list = nn.ModuleList([MultiCoil(kspace_model, self._coil_dim)] * n_iter)
        
        self.attn_model = LeT_transformer(is_sab=True)      # SAB
        self.attn_model2 = LeT_transformer(is_sab=False)    # DAB

        
    def forward(self,
                masked_kspace,
                mask,
                sensitivity):
        """Computes forward pass of :class:`KIKINet`.
        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        
        kspace = masked_kspace.clone()
        batch, ncoils, h, w, chans = kspace.shape
        for idx in range(self.n_iter):
            kspace = self.kspace_model_list[idx](kspace)
            image = T.ifft2(kspace)
            image = T.complex_multiply(image[...,0], image[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)
            
            image = T.complex_center_crop(image,(320,320))

            image = self.attn_model(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            image = apply_dc(image=image,h=h,w=w,sensitivity=sensitivity,mask=mask,masked_kspace=masked_kspace)

            image = self.attn_model2(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            image = apply_dc(image=image,h=h,w=w,sensitivity=sensitivity,mask=mask,masked_kspace=masked_kspace)
            
        return image
        
        
    
