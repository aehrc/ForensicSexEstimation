# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:11:51 2023

@author: Hang Min
"""

'''
Only perform augmentations including flip and rotation on x-y plane

'''

import imgaug.augmenters as iaa
import numpy as np

def Flip(image):
  
    return np.flip(image, axis=1)


def rotate_pos(image, r):
    
    seq_img = iaa.Rotate((0, 10), order=1, random_state=r, cval=-1024)
    
    imgs_aug = seq_img.augment_images([image])
    
    return imgs_aug[0]

def rotate_neg(image, r):
    
    seq_img = iaa.Rotate((-10, 0), order=1, random_state=r, cval=-1024)
   
    imgs_aug = seq_img.augment_images([image])
    
    return imgs_aug[0]


class augmentation_only_image():
    """
    #   we try a few augmentations:
   
    """    
    def __init__(self, image, rts):
        self.image = image       
        self.row = self.image.shape[0]
        self.col = self.image.shape[1]
        self.depth = self.image.shape[2]   
        self.rts = rts # random states, an array with 5 elements all interger
    
    def aug_0 (self):
    #  flip
        aug_img = Flip(self.image)
    
        return aug_img
 

    def aug_1 (self):
        
        aug_img = rotate_pos(self.image, self.rts[0])
              
        return aug_img
    
    def aug_2 (self):
        
        aug_img = rotate_neg(self.image, self.rts[1])
              
        return aug_img
    
    
       
#   Make selections, in this way, we do not have to calculate all the augmentaion before making a selection. 
#   We can directly only compute the selected augmentations.
 
        
    def selection(self, i):
                     
        method_name = 'aug_'+str(i)
        method = getattr(self, method_name, lambda :'Invalid')
        
        return method()
    
    def select_aug(self, select):
        
        selected_aug = []
        
        for j in select:
            
            AUG = self.selection(j)
                        
            selected_aug.append(AUG)
            
        return selected_aug
        