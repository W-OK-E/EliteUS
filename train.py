import sys, os, time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import SSIM_FOCAL
plt.switch_backend('agg') # for servers not supporting display

# import necessary libraries for defining the optimizers
import torch.optim as optim
from torchvision import transforms

import utility

from unet import UNet
from datasets import DAE_dataset
import config as cfg
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



best_val_loss = 1000
script_time = time.time()

def q(text=''):
    print('> {}'.format(text))
    sys.exit()


def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters / 1e6  # in terms of millions

def plot_losses(running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss, epoch):
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(train_epoch_loss)
    
    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(val_epoch_loss)

    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(running_train_loss)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(running_val_loss)
    
    plt.savefig(os.path.join(cfg.losses_dir, 'losses_{}.png'.format(str(epoch + 1).zfill(2))))



def train_unet():
    global best_val_loss
    data_dir = cfg.data_dir
    train_dir = cfg.train_dir
    val_dir = cfg.val_dir

    models_dir = cfg.models_dir
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    losses_dir = cfg.losses_dir
    if not os.path.exists(losses_dir):
        os.mkdir(losses_dir)


    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.0, std=1),
        transforms.ToPILImage(),
        transforms.Pad((0, 18), padding_mode='reflect'),
        transforms.ToTensor()
    ])

    train_dataset = DAE_dataset(os.path.join(data_dir, train_dir), transform=transform)
    val_dataset = DAE_dataset(os.path.join(data_dir, val_dir), transform=transform)

    # print('\nlen(train_dataset) : ', len(train_dataset))
    # print('len(val_dataset)   : ', len(val_dataset))

    batch_size = cfg.batch_size

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers = 6,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers = 6,pin_memory = True)

    # print('\nlen(train_loader): {}  @bs={}'.format(len(train_loader), batch_size))
    # print('len(val_loader)  : {}  @bs={}'.format(len(val_loader), batch_size))

    # defining the model
    model = UNet(in_c=1, n_classes=1, layers=[4, 8, 16]).to(device)

    resume = cfg.resume
    if not resume:
        # print('\nfrom scratch')
        train_epoch_loss = []
        val_epoch_loss = []
        running_train_loss = []
        running_val_loss = []
        epochs_till_now = 0
    else:
        ckpt_path = os.path.join(models_dir, cfg.ckpt)
        ckpt = torch.load(ckpt_path)
        # print(f'\nckpt loaded: {ckpt_path}')
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        losses = ckpt['losses']
        running_train_loss = losses['running_train_loss']
        running_val_loss = losses['running_val_loss']
        train_epoch_loss = losses['train_epoch_loss']
        val_epoch_loss = losses['val_epoch_loss']
        epochs_till_now = ckpt['epochs_till_now']
    #Change the metric to be watched to validation loss
    lr = cfg.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = SSIM_FOCAL(w = 0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=20, verbose=True, min_lr=1e-6)
   
   #CHanged the scheduler factor to 0.8 from 0.5
    log_interval = cfg.log_interval
    epochs = cfg.epochs

    for epoch in range(epochs_till_now, epochs_till_now + epochs):
        running_val_loss = []
        running_train_loss = []
        model.train()
        for batch_idx, (imgs, noisy_imgs) in tqdm(enumerate(train_loader)):
            imgs, noisy_imgs = imgs.to(device), noisy_imgs.to(device)
            optimizer.zero_grad()
            out = model(noisy_imgs)
            loss = loss_fn(out, imgs)
            from psnr import score
            ssim,psnr = score(imgs,out)
            print("PSNR:",psnr,"SSIM:",ssim)
            print("Shape:",imgs.shape,"Output shape:",out.shape)
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
                loss = loss_fn(out, imgs)
                running_val_loss.append(loss.item())
                print("Shape of the image:",imgs.shape)
        
        print("Epoch:",epoch,"Complete")
        val_epoch_loss.append(np.array(running_val_loss).mean())
        scheduler.step(np.array(running_val_loss).mean())

        if(epoch % 10 == 0):
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
    with open('/home/omkumar/Denoising/UNET/logs/loss_logs.txt','w') as file:
        file.write("Epoch Train Val\n")
        for train_loss,val_loss in zip(train_epoch_loss,val_epoch_loss):
            file.write(f'{epoch} {train_loss:.4f} {val_loss:.4f}\n')


def main():
    train_unet()
    


if __name__ == "__main__":
    script_time = time.time()
    main()
    total_script_time = time.time() - script_time
    print(f'\ntotal time taken for running this script: {int(total_script_time // 3600)} hrs {int((total_script_time % 3600) // 60)} mins {int(total_script_time % 60)} secs')
    print('\nFin.')


