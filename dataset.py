import numpy as np
import os, glob, cv2, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class US_dataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.lr_img_data = self.get_data(os.path.join(self.data_dir,'LR'),low_res=True)
        self.imgs_data = self.get_data(os.path.join(self.data_dir, 'HR'),noisy_paths=self.lr_img_data)
    
    def get_data(self, data_path, low_res = False, HR = True, lr_paths = None):
        data = []
        if(low_res):
            sub_dirs = [os.path.join(data_path,x) for x in os.listdir(data_path)]
            for sub_dir_path in sub_dirs:
                for img_path in glob.glob(sub_dir_path + os.sep + "*"):
                    data.append(img_path)
        elif(HR):
            for im in lr_paths:
                data.append(os.path.join(data_path,im.split("/")[-1]))
        else:
            for img_path in glob.glob(data_path + os.sep + '*'):
                data.append(img_path)
        
        return data   
    

    def __getitem__(self, index):  
        img       = cv2.imread(self.imgs_data[index] ,0)
        scale_info = self.noisy_imgs_data[index].split("/")[-2]
        lr_img = cv2.imread(self.noisy_imgs_data[index] ,0)

        if self.transform is not None:            
            img = self.transform(img)             
            lr_img = self.transform(lr_img)  
        
        return img, lr_img, scale_info


    def __len__(self):
        return len(self.imgs_data)

