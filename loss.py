import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from kornia.filters import Laplacian
from kornia.losses import charbonnier_loss
from FDL_pytorch.FDL import FDL_loss
from FFL.focal_frequency_loss import FocalFrequencyLoss
from kornia.losses import total_variation




device = "cuda" if torch.cuda.is_available() else "cpu"
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.reduction = None
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        window_1d = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class VGG(nn.Module):
    def __init__(self, conv_index = '54', rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:35]).to(device)
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.vgg.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.vgg(x)
            return x
        sr = sr.repeat(1,3,1,1)#Since VGG loss can only take 3 channel input, I am repeating the image along the channel dimension
        hr = hr.repeat(1,3,1,1) 
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

class SSIM_FOCAL(nn.Module):
    def __init__(self, conv_index = '54', rgb_range=1,w = 0.5,model = 'EffNet',gradient = True):
        super(SSIM_FOCAL, self).__init__()
        self.reduction = "none"
        self.ssim = SSIMLoss()
        self.focal = FDL_loss(device = device,phase_weight=w,model=model,gradient = gradient)
        self.weight = 0.5
    def forward(self, sr, hr):
        ssim_loss = self.ssim(sr, hr)
        focal_loss = self.focal(sr, hr)
        # ssim_loss
        return (self.weight*ssim_loss + (1-self.weight)*focal_loss,ssim_loss,focal_loss)

class FOCAL_L1(nn.Module):
    def __init__(self,w = 0.5):
        super(FOCAL_L1,self).__init__()
        self.reduction = "none"
        self.focal = FDL_loss(device = device)
        self.weight = w
    def forward(self,sr,hr):
        l1_loss = nn.L1Loss()(sr,hr)
        focal_loss = self.focal(sr,hr)
        return (self.weight * l1_loss + (1-self.weight)*focal_loss,l1_loss,focal_loss)
    
class SSIM_L1(nn.Module):
    def __init__(self,w = 0.5):
        super(SSIM_L1,self).__init__()
        self.reduction = "none"
        self.ssim = SSIMLoss()
        self.weight = w

    def forward(self,sr,hr):
        l1_loss = nn.L1Loss()(sr,hr)
        ssim_loss = self.ssim(sr,hr)
        return (self.weight * l1_loss + (1-self.weight) * ssim_loss,l1_loss,ssim_loss)

class FOCAL_Char(nn.Module):
    def __init__(self,w = 0.5):
        super(FOCAL_Char,self).__init__()
        self.reduction = "mean"
        self.focal = FDL_loss(device = device)
        self.weight = w
    
    def forward(self,sr,hr):
        char_loss = charbonnier_loss(sr,hr,reduction = self.reduction)
        focal_loss = self.focal(sr,hr)
        return (self.weight*char_loss + (1-self.weight)*focal_loss,char_loss,focal_loss)


class SSIM_FFL(nn.Module):
    def __init__(self, conv_index = '54', rgb_range=1,w = 0.5):
        super(SSIM_FFL, self).__init__()
        self.reduction = "none"
        self.ssim = SSIMLoss()
        self.focal = FocalFrequencyLoss()
        self.weight = w

    def forward(self, sr, hr):
        ssim_loss = self.ssim(sr, hr)
        focal_loss = self.focal(sr, hr)
        return (self.weight*ssim_loss + (1-self.weight)*focal_loss,ssim_loss,focal_loss)


class mse_FFL(nn.Module):
    def __init__(self, conv_index = '54', rgb_range=1,w = 0.5):
        super(mse_FFL, self).__init__()
        self.reduction = "none"
        self.mse = nn.MSELoss()
        self.focal = FocalFrequencyLoss()
        self.weight = w

    def forward(self, sr, hr):
        mse_loss = self.mse(sr,hr)
        focal_loss = self.focal(sr, hr)
        return (self.weight*mse_loss + (1-self.weight)*focal_loss,mse_loss,focal_loss)


class Laplace_Loss(nn.Module):
    def __init__(self,kernel_size = 3):
        super(Laplace_Loss,self).__init__()
        self.laplace = Laplacian(kernel_size=kernel_size)
    
    def forward(self,sr,hr):
        laplace_sr = self.laplace(sr)
        laplace_hr = self.laplace(hr)
        laplace_diff = torch.abs(laplace_hr-laplace_sr)
        return laplace_diff.mean((1,2,3)).mean()
    

class SSIM_Laplace(nn.Module):
    def __init__(self,kernel_size = 3,w = 0.5):
        super(SSIM_Laplace, self).__init__()
        self.reduction = "none"
        self.ssim = SSIMLoss()
        self.laplace = Laplace_Loss(kernel_size=kernel_size)
        self.weight = w

    def forward(self,sr,hr):
        ssim_loss = self.ssim(sr,hr)
        laplace = self.laplace(sr,hr)
        return (self.weight * ssim_loss + (1-self.weight)*laplace, ssim_loss,laplace)


class Laplace(nn.Module):
    def __init__(self,kernel_size = 3):
        super(Laplace, self).__init__()
        self.reduction = "none"
        self.laplace = Laplace_Loss(kernel_size=kernel_size)

    def forward(self,sr,hr):
        laplace = self.laplace(sr,hr)
        return laplace


class L2_Total(torch.nn.Module):
    def __init__(self, noisy_image):
        super(L2_Total,self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction="mean")
        self.regularization_term = K.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image

# Usage example
if __name__ == "__main__":
    img1 = torch.rand((4, 3, 256, 256))  # Example batch of images
    img2 = torch.rand((4, 3, 256, 256))  # Example batch of reference images

    ssim_loss = SSIMLoss()
    loss = ssim_loss(img1, img2)
    print(f"SSIM Loss: {loss.item()}")

    lap = Laplace_Loss()
    sr =  torch.rand((10,1,192,128))
    hr =  torch.rand((10,1,192,128))
    print("Laplacian Loss:",lap(sr,hr))
    print("Charbonnier Loss:",charbonnier_loss(sr,hr,reduction="mean"))