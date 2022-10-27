from __future__ import print_function
from PIL import Image
import os
import os.path
from os.path import join
import numpy as np
import sys
import torchvision.transforms as transforms
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from utils import download_url, check_integrity, multiclass_noisify, Cutout
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import h5py


class VISDAC(data.Dataset):
    def __init__(self, root, domain, train=True, transform=None, from_file=False):
        self.train = train
        self.transform = transform
        
        if domain == 'source':
            domain = 'train_scaled256'
        else:
            if self.train:
                domain = 'validation_scaled256'
            else:
                domain = 'test_scaled256'
        
        if not from_file:
            data = []
            labels = []
            for c in range(12):
                files = glob.glob(os.path.join(root,domain,str(c),"*"))
                data.extend(files)
                labels.extend([c]*len(files))

            np.random.seed(1234)
            idx = np.random.permutation(len(data))

            self.data = np.array(data)[idx]
            self.labels = np.array(labels)[idx]

            test_perc = 20
            
            test_len = len(self.data)*test_perc//100                   
            
            if domain == 'source':
                if self.train:
                    self.data = self.data[test_len:]
                    self.labels = self.labels[test_len:]
                else:
                    self.data = self.data[:test_len]
                    self.labels = self.labels[:test_len]
        
        else:

            self.data = np.load(os.path.join(root, domain+"_imgs.npy"))
            self.labels = np.load(os.path.join(root, domain+"_labels.npy"))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.labels[index]          

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)
         
        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.X)

