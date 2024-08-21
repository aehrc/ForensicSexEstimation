# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:11:43 2023

@author: Hang Min

Resample image to 1*1*1mm isotropic
"""
import sys
import SimpleITK as sitk
import numpy as np
from natsort import natsorted
import os
import shutil
import argparse
import time
# Add the parent directory of 'data' to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
from data.utils import isotropic_resampling

if __name__ == "__main__":
    start = time.time()  
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--CT_nifti_path", type=str, help="the path to store converted nifti data", default = 'Cranial CT nifti')
    parser.add_argument("--CT_resampled_path", type=str, help="the path where resampled CT data are stored", default = 'Cranial CT nifti isotropic')
    args = parser.parse_args()
        
    pixelType = sitk.sitkFloat32
    
    case_folder = args.CT_nifti_path
    case_names = natsorted(os.listdir(case_folder))
    
    resampled_img_path = args.CT_resampled_path
    os.makedirs(resampled_img_path, exist_ok=True)
        
    for i, case_name in enumerate(case_names):
                
        print(case_name)
                        
        img = sitk.ReadImage(os.path.join(case_folder, case_name), pixelType)
        img_array = sitk.GetArrayFromImage(img)
        img_array[img_array<-1024.0] = -1024.0
        img_array[img_array>3071.0] = 3071.0
       
        img_thresholded = sitk.GetImageFromArray(img_array)
        img_thresholded.CopyInformation(img)
        
        resampled_img = isotropic_resampling(img_thresholded, new_spacing = (1.0, 1.0, 1.0))                
        
        sitk.WriteImage(sitk.Cast(resampled_img, sitk.sitkInt16), os.path.join(resampled_img_path, case_name))
                                                           
    end = time.time()
    elapsed = end - start
    elapsed = round(elapsed, 2)
    print(
        "Running resampling took "
        + str(elapsed)
        + " secs or "
        + str(elapsed / 60)
        + " mins in total"
    )
        
        
        
        