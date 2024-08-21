# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:11:37 2024

@author: Hang Min

Crop the skull region based on the skull segmentation
"""


import os
import sys
# Add the parent directory to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
import SimpleITK as sitk
from natsort import natsorted
import argparse
from data.utils import pad_and_crop_image_centered


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', help='the path to the image folder', type=str, default="../Cranial CT data/Cranial CT nifti isotropic")
    parser.add_argument('--skull_seg_folder', help='the path to the skull segmentation folder', type=str, default="../Cranial CT data/Cranial CT isotropic segmentations")
    parser.add_argument('--cropped_image_folder', help='the path to the cropped images', type=str, default="../Cranial CT data/Cranial CT nifti isotropic crop")
    parser.add_argument('--cropped_skull_seg_folder', help='the path to the cropped skull segmentation', type=str, default="../Cranial CT data/Cranial CT isotropic segmentations crop")
    
    
    args = parser.parse_args()    
    image_folder = args.image_folder
    skull_seg_folder = args.skull_seg_folder
    
    case_names = natsorted(os.listdir(image_folder))
    
    for i, case_name in enumerate(case_names):
        
        image = sitk.ReadImage(os.path.join(image_folder, case_name))
        base_name = case_name.rsplit('.nii.gz', 1)[0]
        skull_mask = sitk.ReadImage(os.path.join(skull_seg_folder, base_name, 'skull.nii.gz'))
        
        # Pad and crop both the input image and binary mask around the bounding box center with a 256x256x256 ROI
        cropped_image, cropped_mask = pad_and_crop_image_centered(image, skull_mask, roi_size=(256, 256, 256), padding_value=-1024)

        save_crop_image_folder = args.cropped_image_folder
        os.makedirs(save_crop_image_folder, exist_ok=True)
        save_crop_mask_folder = args.cropped_skull_seg_folder
        os.makedirs(save_crop_mask_folder, exist_ok=True)
        
        # Save or process the cropped images
        sitk.WriteImage(cropped_image, os.path.join(save_crop_image_folder, case_name))
        sitk.WriteImage(cropped_mask, os.path.join(save_crop_mask_folder, case_name))