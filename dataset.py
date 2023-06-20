import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
from data import transforms as T
import torch
import os 


class KneeData(Dataset):

    def __init__(self, root): 

        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        for fname in sorted(files):
            self.examples.append(fname)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i] 

        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])
            img_gt    = T.complex_center_crop(img_gt,(320,320))
            img_und   = torch.from_numpy(data['img_und'][:])
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])
            masks = torch.from_numpy(data['masks'][:])
            sensitivity = torch.from_numpy(data['sensitivity'][:])
            
            return img_gt,img_und,rawdata_und,masks,sensitivity


class KneeDataDev(Dataset):

    def __init__(self, root):

        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        for fname in sorted(files):
            self.examples.append(fname) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])
            img_gt    = T.complex_center_crop(img_gt,(320,320))
            img_und   = torch.from_numpy(data['img_und'][:])
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])
            masks = torch.from_numpy(data['masks'][:])
            sensitivity = torch.from_numpy(data['sensitivity'][:])
 
       
        return  img_gt,img_und,rawdata_und,masks,sensitivity,str(fname.name)




