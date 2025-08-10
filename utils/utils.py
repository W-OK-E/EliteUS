import sys
import numpy as np
import matplotlib.pyplot as plt
from torch import tensor
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# for servers not supporting display
plt.switch_backend('agg')

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


def q(text=''):
    print('> {}'.format(text))
    sys.exit()

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters / 1e6  # in terms of millions


def score(sr:tensor,hr:tensor):
    sr = sr.permute(0,2,3,1).detach().cpu().numpy()
    hr = hr.permute(0,2,3,1).detach().cpu().numpy()
    ssim_score = []
    psnr_score = []

    for s,h in zip(sr,hr):
        ssim_score.append(structural_similarity(s,h,multichannel=True))
        psnr_score.append(peak_signal_noise_ratio(s,h,data_range=1))

    return np.mean(ssim_score),np.mean(psnr_score)