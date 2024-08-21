# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:40:54 2020

@author: Hang Min

Loading the CT image and skull segmentation mask 
"""
import torch
from torch.utils.data.dataset import Dataset
from data.utils import importimage_3D, importimage_3D_image_skull, importimage_3D_skull
import numpy as np

class load_data(Dataset):
    
    def __init__(self,  image_path, case_partition_spreadsheet, im_shape=(128,128,64), if_aug=0):
        self.images = []
        self.labels1 = []
        self.labels2 = []
      
        image, label1, label2 = importimage_3D (image_path, case_partition_spreadsheet, im_shape, if_aug)
            
        self.images = self.images + image
        self.labels1 = self.labels1 + label1
        self.labels2 = self.labels2 + label2
        
    def __getitem__(self, index):
        image = self.images[index]

        label1 = self.labels1[index]
        label2 = np.expand_dims(self.labels2[index], axis=0)
        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
   
        image = image.type(torch.FloatTensor)
        label1 = label1.type(torch.FloatTensor)
        label2 = label2.type(torch.FloatTensor)
        
        return image, label1, label2

    def __len__(self):
        return len(self.images)
    
    
class load_data_image_skull(Dataset):
    
    def __init__(self,  image_path, case_partition_spreadsheet, seg_path=None, im_shape=(128,128,64), if_aug=0):
        self.images = []
        self.labels1 = []
        self.labels2 = []
      
        image, label1, label2 = importimage_3D_image_skull (image_path, case_partition_spreadsheet, seg_path, im_shape, if_aug)
            
        self.images = self.images + image
        self.labels1 = self.labels1 + label1
        self.labels2 = self.labels2 + label2
        
    def __getitem__(self, index):
        image = self.images[index]

        label1 = self.labels1[index]
        label2 = np.expand_dims(self.labels2[index], axis=0)
        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
   
        image = image.type(torch.FloatTensor)
        label1 = label1.type(torch.FloatTensor)
        label2 = label2.type(torch.FloatTensor)
        
        return image, label1, label2

    def __len__(self):
        return len(self.images)
    
    
class load_data_skull(Dataset):
    
    def __init__(self,  image_path, case_partition_spreadsheet, seg_path=None, im_shape=(128,128,64), if_aug=0):
        self.images = []
        self.labels1 = []
        self.labels2 = []
      
        image, label1, label2 = importimage_3D_skull (image_path, case_partition_spreadsheet, seg_path, im_shape, if_aug)
            
        self.images = self.images + image
        self.labels1 = self.labels1 + label1
        self.labels2 = self.labels2 + label2
        
    def __getitem__(self, index):
        image = self.images[index]

        label1 = self.labels1[index]
        label2 = np.expand_dims(self.labels2[index], axis=0)
        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
   
        image = image.type(torch.FloatTensor)
        label1 = label1.type(torch.FloatTensor)
        label2 = label2.type(torch.FloatTensor)
        
        return image, label1, label2

    def __len__(self):
        return len(self.images)


