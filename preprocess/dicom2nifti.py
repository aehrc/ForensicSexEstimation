# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:49:56 2023

@author: Hang Min

Convert CT dicom into nifti
"""
import sys
import os
import SimpleITK as sitk
from natsort import natsorted
import argparse
import time
# Add the parent directory of 'data' to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)

from data.utils import read_dcm_series


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Convert dicom CT data to nifti')    
    # Add the argument
    parser.add_argument("--CT_dicom_path", type=str, help="the of the CT dicom data", default = 'Cranial CT dicom')
    parser.add_argument("--CT_nifti_path", type=str, help="the path to store converted nifti data", default = 'Cranial CT nifti')
    # Parse the arguments
    args = parser.parse_args()
    
    start = time.time()  
    
    dicom_path = args.CT_dicom_path
    nifti_path = args.CT_nifti_path
    
    if not os.path.exists(nifti_path): 
        os.makedirs(nifti_path)
                    
    case_names = natsorted(os.listdir(dicom_path))
    
    for i, case_name in enumerate(case_names):
        
        image = read_dcm_series(os.path.join(dicom_path, case_name))
        
        print(case_name)
        
        nifti_image_path = os.path.join(nifti_path, case_name + '.nii.gz')
        sitk.WriteImage(image, nifti_image_path)
        
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
        
        
        
