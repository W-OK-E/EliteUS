import os
import torch
from tqdm import tqdm
from EliteNET import UNet
from utils.utils import score
from dataset import US_dataset
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_sr_test(model, test_loader, save_dir, device='cuda'):
    """
    Run super-resolution test using an existing dataset class.

    Args:
        model: The trained SR model.
        dataset_class: Your dataset class (unchanged).
        data_dir: Directory containing test images.
        device: 'cuda' or 'cpu'.
    """
    model.eval()
    model.to(device)

    psnr_total, ssim_total, count = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (HR_imgs, LR_imgs) in tqdm(enumerate(test_loader)):
            HR_imgs, LR_imgs = HR_imgs.to(device), LR_imgs.to(device)
            
            SR_imgs = model(LR_imgs)

            # Compute metrics (replace with your metric functions)
            
            ssim,psnr = score(HR_imgs,SR_imgs)

            psnr_total += psnr
            ssim_total += ssim
            count += 1

            save_image(LR_imgs[0].cpu(), os.path.join(save_dir,f"LR_{batch_idx}.jpg"))
            save_image(HR_imgs[0].cpu(), os.path.join(save_dir,f"HR_{batch_idx}.jpg"))
            save_image(SR_imgs[0].cpu(), os.path.join(save_dir,f"SR_{batch_idx}.jpg"))


    print(f"Average PSNR: {psnr_total / count:.2f}")
    print(f"Average SSIM: {ssim_total / count:.4f}")

def main(config = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad((0, 18), padding_mode='reflect'),
        transforms.ToTensor()
    ])

    test_dataset = US_dataset(os.path.join(config['dataset']['base_dir'],config['dataset']['test']),transform=transform)
    test_loader = DataLoader(test_dataset,config['test_bs'],shuffle=True)

    model = UNet(in_c=config["model_params"]["in_channels"], n_classes=config["model_params"]["out_channels"], layers=config["model_params"]["layers"]).to(device)
    run_sr_test(model,test_loader,config['res_dir'])
