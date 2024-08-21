# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:41:16 2024

@author: Hang Min
"""
import sys
import os
import subprocess
# Add the parent directory of 'data' to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)


# Define the paths
ct_dicom_path = '../Cranial CT data/CT dicom data'
ct_nifti_path = '../Cranial CT data/CT nifti data'
ct_isotropic_path = '../Cranial CT data/Cranial CT nifti isotropic/'
skull_seg_folder = '../Cranial CT data/Cranial CT isotropic segmentations original/'
ct_cropped_path = '../Cranial CT data/Cranial CT nifti isotropic crop/'
skull_cropped_path = '../Cranial CT data/Cranial CT isotropic segmentations crop'

# 1. Command to run the dicom convertion script
command = [
    'python', '../preprocess/dicom2nifti.py',
    '--CT_dicom_path', ct_dicom_path,
    '--CT_nifti_path', ct_nifti_path
]

# # Run the script
result = subprocess.run(command, capture_output=True, text=True)

# Print the output and errors (if any)
print("Output:\n", result.stdout)
print("Errors:\n", result.stderr)

# 2. Command to run the isotropic resampling 
command = [
    'python', '../preprocess/image_resampling.py',
    '--CT_nifti_path', ct_nifti_path,
    '--CT_resampled_path', ct_isotropic_path
]

result = subprocess.run(command, capture_output=True, text=True)

# Print the output and errors (if any)
print("Output:\n", result.stdout)
print("Errors:\n", result.stderr)



# 3. Run total segmentator CT segmentation
# Before running this, the total segmentator bash script needs to have execute permissions by doing: chmod +x example.sh
# This is in a bash file to be added.
# Run the shell script
total_segmentator_script = './run_total_segmentator.sh'
result = subprocess.run([total_segmentator_script], shell=True, check=True)

# If you need to capture the output
output = subprocess.run([total_segmentator_script], shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Print the output
print(output.stdout.decode())
print(output.stderr.decode())


# 4. Command to crop the image and skull mask 
command = [
    'python', '../preprocess/skull_region_cropping.py',
    '--image_folder', ct_isotropic_path,
    '--skull_seg_folder', skull_seg_folder,
    '--save_crop_image_folder', ct_cropped_path,
    '--save_crop_mask_folder', skull_cropped_path
]

result = subprocess.run(command, capture_output=True, text=True)

# Print the output and errors (if any)
print("Output:\n", result.stdout)
print("Errors:\n", result.stderr)
