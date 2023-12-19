
import os
import numpy as np
import pandas as pd
import skimage.io
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Resize


class CustomDataset(Dataset):
    def __init__(self, data_dir ,train=True,transform=None):
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
        self.resize =Resize((128,128))
        if self.train : 
            self.path= os.path.join(data_dir, "GTSRB_Final_Train")
            
        else : 
            self.path = path= os.path.join(self.data_dir, 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/')
            self.gt_df = pd.read_csv(os.path.join(self.data_dir, "GTSRB_Final_Test_GT/GT-final_test.csv"), sep = ";")
    def load_train(self, idx) :
        name = os.listdir(os.path.join(self.path))[idx]
        c = name.split('@')[0]
        image = skimage.io.imread(os.path.join(self.path, name))
        return [image, c]
    def load_test(self,idx) : 
        
        image = skimage.io.imread(os.path.join(self.path, os.listdir(self.path)[idx]))
        c = self.gt_df[self.gt_df['Filename'] == os.listdir(self.path)[idx]]['ClassId'].values
        return [image, c]
    def __len__(self):
        return len(os.listdir(self.path))

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
        sample[0] = self.resize(torch.from_numpy(sample[0]).moveaxis(-1,0))/to(torch.float32)/255
        if self.transform is not None :
            sample = self.transform(sample)

        return sample

