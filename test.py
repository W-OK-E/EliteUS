import os, shutil, cv2
import numpy as np
import torch
from torchvision import transforms
from unet2 import UNet
from datasets import custom_test_dataset, DAE_dataset
import config as cfg
from psnr import score
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import tqdm

res_dir = cfg.res_dir
if os.path.exists(res_dir):
    shutil.rmtree(res_dir)

if not os.path.exists(res_dir):
    os.mkdir(res_dir)
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')
print('device: ', device)

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((156,128)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.0, std=1),
        transforms.ToPILImage(),
        transforms.Pad((0, 18), padding_mode='reflect'),
        transforms.ToTensor()
    ])
test_transform = transform

test_dir = '/home/omkumar/Denoising/UNET/KGP_Data/tes'
print("test directory:",test_dir)
test_dataset       = DAE_dataset(test_dir, transform=test_transform)
test_loader        = torch.utils.data.DataLoader(test_dataset, batch_size = 4, shuffle = not True)

print('\nlen(test_dataset) : {}'.format(len(test_dataset)))
print('len(test_loader)  : {}  @bs={}'.format(len(test_loader), cfg.test_bs))

# defining the model
model = UNet(in_c=1,n_classes = 1,layers=[4,8,16]).to(device)#, depth = cfg.depth, padding = True).to(device)

ckpt_path = cfg.ckpt
ckpt = torch.load(ckpt_path)
print(f'\nckpt loaded: {ckpt_path}')
model_state_dict = ckpt['model_state_dict']
model.load_state_dict(model_state_dict)
# model = ckpt #Since the entire model was saved, we donot meed to load_state_dict
model.to(device)

print("Results being saved in :",res_dir)
print('\nDenoising noisy images...')
model.eval()
psnr_val = {}
ssim_val = {}
total_psnr = []
total_ssim = []
with torch.no_grad():
    for batch_idx, (imgs,noisy_imgs,scale_info) in tqdm.tqdm(enumerate(test_loader)):
        # print('batch: {}/{}'.format(str(batch_idx + 1).zfill(len(str(len(test_loader)))), len(test_loader)), end='\r')
        imgs, noisy_imgs = imgs.to(device)[:5], noisy_imgs.to(device)[:5]
        out = model(noisy_imgs)
        # ssim,psnr = score(out,imgs)
        # total_psnr.append(psnr)
        # total_ssim.append(ssim)
        # print(psnr)
        idx = 0
        for noisy,denoisy,orig_img,scale in zip(noisy_imgs,out,imgs,scale_info):
            noisy = noisy.reshape(192,128,1).cpu().numpy()
            denoisy = denoisy.reshape(192,128,1).cpu().numpy()
            orig_img = orig_img.reshape(192,128,1).cpu().numpy()
            save_path = os.path.join(res_dir,scale)

            ssim,psnr = structural_similarity(denoisy,orig_img,multichannel = True),peak_signal_noise_ratio(denoisy,orig_img,data_range=1)
            if scale in psnr_val:
                psnr_val[scale].append(psnr)
            else:
                psnr_val[scale] = [psnr]

            if scale in ssim_val:
                ssim_val[scale].append(ssim)
            else:
                ssim_val[scale] = [ssim]
            if(not os.path.exists(save_path)):
                os.mkdir(save_path)
            
            # # print("Max value is denoised:",denoisy.max(),"Mean of noisy:",noisy.mean(),"Mean of denoised:",denoisy.mean())
            # cv2.imwrite(os.path.join(res_dir, f'denoised{str(idx).zfill(3)}_{scale}.jpg'), (denoisy*255).astype(np.uint8))
            # cv2.imwrite(os.path.join(res_dir, f'Noised{str(idx).zfill(3)}_{scale}.jpg'), (noisy*255).astype(np.uint8))
            # cv2.imwrite(os.path.join(res_dir, f'Orig{str(idx).zfill(3)}_{scale}.jpg'), (orig_img*255).astype(np.uint8))

            cv2.imwrite(os.path.join(save_path, f'denoised{str(idx).zfill(3)}.jpg'), (denoisy*300).astype(np.uint8))
            cv2.imwrite(os.path.join(save_path, f'Noised{str(idx).zfill(3)}.jpg'), (noisy*255).astype(np.uint8))
            cv2.imwrite(os.path.join(save_path, f'Orig{str(idx).zfill(3)}.jpg'), (orig_img*255).astype(np.uint8))
            idx += 1
        
# print("Psnr Values:",psnr_val)
# print("Mean PSNR:",np.mean(total_psnr),"Mean SSIM = ",np.mean(total_ssim))

all_scales_psnr = []
all_scales_ssim = []
txt_file = os.path.join(res_dir,"Metrics.txt")
with open(txt_file,'w') as file:
    file.write(f"Image Denoised using: {ckpt_path}\n")
    try:
        for key,val in psnr_val.items():
            file.write(f"For Scale: {key}, Mean PSNR: {np.mean(val)}\n")
            all_scales_psnr.append(np.mean(val))
        for key,val in ssim_val.items():
            file.write(f"For Scale: {key}, Mean SSIM: {np.mean(val)}\n")
            all_scales_ssim.append(np.mean(val))
        file.write(f"MEAN PSNR: {np.mean(all_scales_psnr)}, MEAN_SSIM: {np.mean(all_scales_ssim)}")
    except:
        pass

print("MEAN PSNR:",np.mean(all_scales_psnr),"MEAN_SSIM:",np.mean(all_scales_ssim))
print('\n\nresults saved in \'{}\' directory'.format(res_dir))

print('\nFin.')
