import os, shutil, cv2
import numpy as np
import torch
from torchvision import transforms
from unet2 import UNet
from datasets import custom_test_dataset, DAE_dataset
import config as cfg
from psnr import score

res_dir = "results_4x"
# if os.path.exists(res_dir):
#     shutil.rmtree(res_dir)

if not os.path.exists(res_dir):
    os.mkdir(res_dir)
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')
print('device: ', device)

transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean = 0.0, std = 1),transforms.ToPILImage(),transforms.Pad((0,20),padding_mode = 'reflect'), transforms.ToTensor()])

test_transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean = 0.0, std = 1),transforms.ToPILImage(),transforms.Pad((0,18),padding_mode = 'reflect'), transforms.ToTensor()])

test_dir = '/home/omkumar/Denoising/UNET/data/TEST'
print("test directory:",test_dir)
test_dataset       = DAE_dataset(test_dir, transform=test_transform)
test_loader        = torch.utils.data.DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = not True)

print('\nlen(test_dataset) : {}'.format(len(test_dataset)))
print('len(test_loader)  : {}  @bs={}'.format(len(test_loader), cfg.test_bs))

# defining the model
model = UNet(in_c=1,n_classes = 1,layers=[4,8,16]).to(device)#, depth = cfg.depth, padding = True).to(device)
models_dir = '/home/omkumar/Denoising/UNET/SSIM_FDL_TRAIN_EffNet/lr_0.001_w_0.5'
models = os.listdir(models_dir)
for ckpt in models:
    ckpt_path = os.path.join(models_dir, ckpt)
    ckpt_num = ckpt[20:].replace('.pth','')
    # if(int(ckpt_num) < 240):
    #     continue
    print("Writing for model:",ckpt_num)
    ckpt = torch.load(ckpt_path)
    print(f'\nckpt loaded: {ckpt_path}')
    model_state_dict = ckpt['model_state_dict']
    model.load_state_dict(model_state_dict)
    # model = ckpt #Since the entire model was saved, we donot meed to load_state_dict
    model.to(device)

    print("Results being saved in :",res_dir)
    print('\nDenoising noisy images...')
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs,noisy_imgs,scale_info) in enumerate(test_loader):
            # print('batch: {}/{}'.format(str(batch_idx + 1).zfill(len(str(len(test_loader)))), len(test_loader)), end='\r')
            idx = 100
            print("Index Set to :",idx)
            imgs, noisy_imgs = imgs[idx].unsqueeze(0).to(device), noisy_imgs[idx].unsqueeze(0).to(device)
            out = model(noisy_imgs)
            ssim,psnr = score(out,imgs)
            print(psnr)
            for noisy,denoisy,orig_img in zip(noisy_imgs,out,imgs):                
                noisy = noisy.reshape(192,128,1).cpu().numpy()
                denoisy = denoisy.reshape(192,128,1).cpu().numpy()
                orig_img = orig_img.reshape(192,128,1).cpu().numpy()

                print("Writing Images")
                # print("Max value is denoised:",denoisy.max(),"Mean of noisy:",noisy.mean(),"Mean of denoised:",denoisy.mean())
                cv2.imwrite(os.path.join(res_dir, f'denoised{str(ckpt_num).zfill(3)}.jpg'), (denoisy*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(res_dir, f'Noised{str(ckpt_num).zfill(3)}_X4.00.jpg'), (noisy*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(res_dir, f'Orig{str(ckpt_num).zfill(3)}_x4.00.jpg'), (orig_img*255).astype(np.uint8))
                idx += 1
            break
        
            
print('\n\nresults saved in \'{}\' directory'.format(res_dir))

print('\nFin.')
