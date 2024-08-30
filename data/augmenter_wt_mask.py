# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:40:09 2021

@author: Hang Min

Only perform augmentations including flip and rotation on x-y plane
"""

import imgaug.augmenters as iaa
import numpy as np

def flipit(images, labels):
    
    return [np.flip(images[0], axis=1)], [np.flip(labels[0], axis=1)]

def rotate_pos(images, labels, r):
    
    seq_img = iaa.Rotate((0, 10), order=1, cval=-1024, random_state=r, name="MyRotatePos")

    seq_masks = iaa.Rotate((0, 10), order=0, random_state=r, name="MyRotatePos")
    
    seq_img = seq_img.localize_random_state()

    seq_img_i = seq_img.to_deterministic()
    seq_masks_i = seq_masks.to_deterministic()
    
    seq_masks_i = seq_masks_i.copy_random_state(seq_img_i, matching="name")
      
    imgs_aug = []
    
    for image in images:
        
        aug_im = seq_img_i.augment_images(images=[image])
        imgs_aug = imgs_aug + aug_im
            
    masks_aug = []
    
    for label in labels:
        
        aug_msk = seq_masks_i.augment_images(images=[label])
        
        masks_aug = masks_aug + aug_msk
    
    # masks_aug = seq_masks_i.augment_images(images=label)
    
    return imgs_aug, masks_aug


def rotate_neg(images, labels, r):
    
    seq_img = iaa.Rotate((-10, 0), order=1, cval=-1024, random_state=r, name="MyRotateNeg")

    seq_masks = iaa.Rotate((-10, 0), order=0, random_state=r, name="MyRotateNeg")
    
    seq_img = seq_img.localize_random_state()

    seq_img_i = seq_img.to_deterministic()
    seq_masks_i = seq_masks.to_deterministic()
    
    seq_masks_i = seq_masks_i.copy_random_state(seq_img_i, matching="name")
      
    imgs_aug = []
    
    for image in images:
        
        aug_im = seq_img_i.augment_images(images=[image])
        imgs_aug = imgs_aug + aug_im
            
    masks_aug = []
    
    for label in labels:
        
        aug_msk = seq_masks_i.augment_images(images=[label])
        
        masks_aug = masks_aug + aug_msk
    
    # masks_aug = seq_masks_i.augment_images(images=label)
    
    return imgs_aug, masks_aug
      
    
class augmentation_wt_mask():
    
    def __init__(self, images, label, rts):
        self.images = images
        self.label = label
        self.rts = rts
    
    def aug_0 (self):
    #  flip
    
        aug_imgs, label = flipit(self.images, self.label)
        
        return aug_imgs, label

    # rotation pos          
    def aug_1 (self):
        
        aug_imgs, label = rotate_pos(self.images, self.label, self.rts[0])
        
        return aug_imgs, label
    
    # rotation neg
    def aug_2 (self):
        
        aug_imgs, label = rotate_neg(self.images, self.label, self.rts[1])
        
        return aug_imgs, label
   
       
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
    
    
