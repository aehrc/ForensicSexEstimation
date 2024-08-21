# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:11:37 2024

@author: Hang Min

Crop the skull region based on the skull segmentation
"""
import os
import sys
import SimpleITK as sitk
from natsort import natsorted
# Add the parent directory of 'data' to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
from data.utils import pad_and_crop_image_centered
import time
import argparse

if __name__ == '__main__':
    
    start = time.time()  
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_folder", type=str, help="the skull CT folder path", default = '../Cranial CT data/Cranial CT nifti isotropic')
    parser.add_argument("--skull_seg_folder", type=str, help="the path where the skull mask are stored", default = '../Cranial CT data/Cranial CT isotropic segmentations')
    parser.add_argument("--save_crop_image_folder", type=str, help="the path where the cropped image will be saved", default = '../Cranial CT data/Cranial CT nifti isotropic crop')
    parser.add_argument("--save_crop_mask_folder", type=str, help="the path where the cropped skull mask will be saved", default = '../Cranial CT data/Cranial CT isotropic segmentations crop')
    
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

        save_crop_image_folder = args.save_crop_image_folder
        os.makedirs(save_crop_image_folder, exist_ok=True)
        save_crop_mask_folder = args.save_crop_mask_folder
        os.makedirs(save_crop_mask_folder, exist_ok=True)
        
        # Save or process the cropped images
        sitk.WriteImage(cropped_image, os.path.join(save_crop_image_folder, case_name))
        sitk.WriteImage(cropped_mask, os.path.join(save_crop_mask_folder, case_name))

    end = time.time()
    elapsed = end - start
    elapsed = round(elapsed, 2)
    print(
        "Running dicom convertion took "
        + str(elapsed)
        + " secs or "
        + str(elapsed / 60)
        + " mins in total"
    )