import sys, os, time
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.utils import plot_losses,score
from loss import SSIM_FOCAL

from EliteNET import UNet
from dataset import US_dataset
import yaml


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


best_val_loss = 1000
script_time = time.time()


def load_config(config_path):
    """
    Reads a YAML configuration file and returns it as a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_unet(cfg = None):
    global best_val_loss

    #Loading Parameters:
    base_dir = cfg["base_dir"]
    train_dir = cfg["train_dir"]
    val_dir = cfg["val_dir"]
    batch_size = cfg["batch_size"]
    models_dir = cfg["logging"]["models_dir"]
    losses_dir = cfg["logging"]["losses_dir"]
    log_interval = cfg["logging"]["log_interval"]
    epochs = cfg["epochs"]
    resume = cfg["resume"]
    lr = cfg["lr"]
    ckpt_path = cfg['ckpt']

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    if not os.path.exists(losses_dir):
        os.mkdir(losses_dir)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad((0, 18), padding_mode='reflect'),
        transforms.ToTensor()
    ])

    train_dataset = US_dataset(os.path.join(base_dir, train_dir), transform=transform)
    val_dataset = US_dataset(os.path.join(base_dir, val_dir), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 6,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers = 6,pin_memory = True)

    # defining the model
    model = UNet(in_c=1, n_classes=1, layers=[4, 8, 16]).to(device)


    if not resume:
        # print('\nfrom scratch')
        train_epoch_loss = []
        val_epoch_loss = []
        running_train_loss = []
        running_val_loss = []
        epochs_till_now = 0
    else:
        ckpt_path = os.path.join(models_dir, ckpt_path)
        ckpt = torch.load(ckpt_path)

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        losses = ckpt['losses']
        running_train_loss = losses['running_train_loss']
        running_val_loss = losses['running_val_loss']
        train_epoch_loss = losses['train_epoch_loss']
        val_epoch_loss = losses['val_epoch_loss']
        epochs_till_now = ckpt['epochs_till_now']

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = SSIM_FOCAL(w = 0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=20, verbose=True, min_lr=1e-6)
   
    for epoch in range(epochs_till_now, epochs_till_now + epochs):
        running_val_loss = []
        running_train_loss = []
        model.train()
        for batch_idx, (imgs, noisy_imgs) in tqdm(enumerate(train_loader)):
            imgs, noisy_imgs = imgs.to(device), noisy_imgs.to(device)
            optimizer.zero_grad()
            out = model(noisy_imgs)
            loss = loss_fn(out, imgs)
            ssim,psnr = score(imgs,out)
            running_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        train_epoch_loss.append(np.array(running_train_loss).mean())
        model.eval()
        
        with torch.no_grad():
            print("Validating")
            eval_psnr = 0
            eval_ssim = 0
            for batch_idx, (imgs, noisy_imgs) in tqdm(enumerate(val_loader)):
                imgs, noisy_imgs = imgs.to(device), noisy_imgs.to(device)
                out = model(noisy_imgs)
                ssim,psnr = score(imgs,out)
                eval_psnr += psnr
                eval_ssim += ssim
                loss = loss_fn(out, imgs)
                running_val_loss.append(loss.item())
            print("Validation Metrics - PSNR:",psnr,"SSIM:",ssim)

        print("Epoch:",epoch,"Complete")
        val_epoch_loss.append(np.array(running_val_loss).mean())
        scheduler.step(np.array(running_val_loss).mean())

        if(epoch % log_interval == 0):
            plot_losses(running_train_loss,running_val_loss,train_epoch_loss,val_epoch_loss,epoch)

        if np.array(running_val_loss).mean() < best_val_loss:
            best_val_loss = np.array(running_val_loss).mean()
            checkpoint = {
                'epochs_till_now':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'learning_rate':lr,
            }
            torch.save(checkpoint, os.path.join(models_dir, f'model_SSIM_FOCAL{epoch}.pth'))


    with open(f'{losses_dir}/loss_logs.txt','w') as file:
        file.write("Epoch Train Val\n")
        for train_loss,val_loss in zip(train_epoch_loss,val_epoch_loss):
            file.write(f'{epoch} {train_loss:.4f} {val_loss:.4f}\n')


if __name__ == "__main__":
    script_time = time.time()
    train_unet()
    total_script_time = time.time() - script_time
    print(f'\ntotal time taken for running this script: {int(total_script_time // 3600)} hrs {int((total_script_time % 3600) // 60)} mins {int(total_script_time % 60)} secs')
    print('\nFin.')


