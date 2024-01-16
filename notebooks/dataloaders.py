
import os
import numpy as np
import pandas as pd
import skimage.io
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Resize


class CustomDataset(Dataset):
    def __init__(self, data_dir ,train=True,transform=None, poison_dir =None, return_names = False , poison_num = None):
        """
        Args:
            data (list): A list of input data.
            labels (list): A list of corresponding labels.
            train (bool): If True, load training data. If False, load test data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.transform = transform
        self.data_dir = data_dir
        self.poison_num = poison_num
        self.poison_dir = poison_dir
        self.resize =Resize((128,128), antialias = True)
        self.return_names = return_names
        if self.poison_dir != None : 
            self.poison_names = os.listdir(poison_dir)
            if self.poison_num != None : 
                self.poison_names = self.poison_names[:self.poison_num]
        if self.train : 
            self.path= os.path.join(data_dir, "GTSRB_Final_Train")
            
        else : 
            self.path = path= os.path.join(self.data_dir, 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/')
            self.gt_df = pd.read_csv(os.path.join(self.data_dir, "GTSRB_Final_Test_GT/GT-final_test.csv"), sep = ";")
    def load_train(self, idx) :
        name = [x for x in os.listdir(self.path) if "csv" not in x ][idx]
        c = name.split('@')[0]
        hyp_name = name.replace("ppm","npy") if len(c)==2 else "0"+name.replace("ppm","npy")
        if self.poison_dir !=None : 
            if hyp_name in self.poison_names :
                image =np.moveaxis(np.load(os.path.join(self.poison_dir,  hyp_name )),0,-1)
                
            else : 
                image = skimage.io.imread(os.path.join(self.path, name))
        else :
            image = skimage.io.imread(os.path.join(self.path, name))
        if self.return_names : 
            return [image, int(c),name]
        else:  
            return [image, int(c)]
    def load_test(self,idx) : 
        name = [x for x in os.listdir(self.path) if "csv" not in x ][idx]
        image = skimage.io.imread(os.path.join(self.path, name))
        c = self.gt_df[self.gt_df['Filename'] == name]['ClassId'].values[0]
        if self.return_names : 
            return [image, int(c),name]
        else:  
            return [image, int(c)]
    def __len__(self):
        return len([x for x in os.listdir(self.path) if "csv" not in x ])

    def __getitem__(self, idx):
        if self.train : 
            sample = self.load_train(idx)
        else :
            sample = self.load_test(idx)
        H,W,_ = sample[0].shape
        if H>W: 
            sample[0]= sample[0][:W]
        else :
            sample[0] = sample[0][:,:H]
        sample[0] = self.resize(torch.from_numpy(sample[0]).moveaxis(-1,0)).to(torch.float32)/255
        if self.transform is not None :
            sample = self.transform(sample)
        
        return sample

