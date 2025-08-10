import numpy as np
import os, glob, cv2, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class DAE_dataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.noisy_imgs_data = self.get_data(os.path.join(self.data_dir,'LR'),noisy=True)
        self.imgs_data       = self.get_data(os.path.join(self.data_dir, 'HR'),noisy_paths=self.noisy_imgs_data)

    def get_data(self, data_path,noisy = False,HR = True,noisy_paths = None):
        data = []
        if(noisy):
            sub_dirs = [os.path.join(data_path,x) for x in os.listdir(data_path)]
            for sub_dir_path in sub_dirs:
                for img_path in glob.glob(sub_dir_path + os.sep + "*"):
                    data.append(img_path)
        elif(HR):
            for im in noisy_paths:
                data.append(os.path.join(data_path,im.split("/")[-1]))
        else:
            for img_path in glob.glob(data_path + os.sep + '*'):
                data.append(img_path)
        return data
        
    def __getitem__(self, index):  
        # read images in grayscale, then invert them
        img       = cv2.imread(self.imgs_data[index] ,0)
        scale_info = self.noisy_imgs_data[index].split("/")[-2]
        noisy_img = cv2.imread(self.noisy_imgs_data[index] ,0)
        print("Getting Item: Original Image:",img.shape,"LR Image:",noisy_img.shape)
        if self.transform is not None:            
            img = self.transform(img)             
            noisy_img = self.transform(noisy_img)  
        print("Inside the dataset:",img.shape,noisy_img.shape)
        return img, noisy_img,scale_info

    def __len__(self):
        return len(self.imgs_data)
    
class custom_test_dataset(Dataset):
    def __init__(self, data_dir, transform = None, out_size = (64, 256)):
        assert out_size[0] <= out_size[1], 'height/width of the output image shouldn\'t not be greater than 1'
        self.data_dir = data_dir
        self.transform = transform
        self.out_size = out_size
        self.imgs_data       = self.get_data(self.data_dir)

    def get_data(self, data_path):
        data = []
        for img_path in glob.glob(data_path + os.sep + '*'):
            data.append(img_path)
        return data
    
    def __getitem__(self, index):  
        # read images in grayscale, then invert them
        img       = cv2.imread(self.imgs_data[index] ,0)
                
        # check if img height exceeds out_size height
        if img.shape[0] > self.out_size[0]:
            resize_factor = self.out_size[0]/img.shape[0]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

        # check if img width exceeds out_size width
        if img.shape[1] > self.out_size[1]:
            resize_factor = self.out_size[1]/img.shape[1]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
       
        # add padding where required
        # pad height
        pad_height = self.out_size[0] - img.shape[0]
        pad_top = int(pad_height/2)
        pad_bottom = self.out_size[0] - img.shape[0] - pad_top
        # pad width
        pad_width = self.out_size[1] - img.shape[1]
        pad_left = int(pad_width/2)
        pad_right = self.out_size[1] - img.shape[1] - pad_left
        
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=(0,0))    
        
        if self.transform is not None:            
            img = self.transform(img)
            print("Transform Applied")     
        
        return img

    def __len__(self):
        return len(self.imgs_data)



if __name__ ==  "__main__":
    data_dir = '/home/omkumar/Denoising/UNET/KGP_Data'
    train_dir = 'train'
    val_dir = 'val'
    imgs_dir = 'imgs'
    noisy_dir = 'noisy'
    debug_dir = 'debug'
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
    train_dataset = DAE_dataset(os.path.join(data_dir, 'tes'), transform=test_transform)
    val_dataset = DAE_dataset(os.path.join(data_dir, val_dir), transform=None)


    loader = DataLoader(train_dataset,batch_size = 4,shuffle=False,num_workers=6,pin_memory=True)

    for batch_idx, (imgs, noisy_imgs,scale_info) in enumerate(loader):
        print(imgs.shape,noisy_imgs.shape)
