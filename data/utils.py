
import os
# from natsort import natsorted
from skimage.transform import resize
import numpy as np
# import random
import SimpleITK as sitk
from data.augmenter_wt_mask import augmentation_wt_mask
from data.augmenter_only_image import augmentation_only_image

total_num_aug = 3
np.random.seed(1)

def read_dcm_series(dcm_folder: str):
    """
    Read dicom series
    :param dcm_folder: original dicom folder
    :return: sitk Image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def isotropic_resampling(image_to_resample, new_spacing = (1.0, 1.0, 1.0)):
    # Set the new spacing to 1x1x1 for isotropic resampling
    # new_spacing = (1.0, 1.0, 1.0)
    
    # Calculate the new size of the image
    original_spacing = image_to_resample.GetSpacing()
    original_size = image_to_resample.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    
    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_to_resample.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image_to_resample.GetOrigin())
    
    # Use an identity transform since we only want to change the spacing
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    
    # Set the default pixel value for new voxels that might fall outside the original image
    resampler.SetDefaultPixelValue(-1024.0)  # Common choice for background or air in CT
    
    # Execute the resampling
    resampled_image = resampler.Execute(image_to_resample)
    
    return resampled_image

def pad_and_crop_image_centered(input_image, binary_mask, roi_size=(256, 256, 256), padding_value=-1024):
    # Compute the bounding box of the binary mask
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(binary_mask > 0)  # Ensure binary format
    bounding_box = label_shape_analysis.GetBoundingBox(1)  # Assuming the mask label is 1
    
    # Calculate the center of the bounding box
    bbox_center = [bounding_box[i] + bounding_box[i + 3] // 2 for i in range(3)]  # For 3D: (x, y, z)
    
    # Calculate the start index for the ROI based on the center
    start_index = [int(c - s / 2) for c, s in zip(bbox_center, roi_size)]
    
    # Calculate padding needs
    image_size = input_image.GetSize()
    pad_lower = [-min(0, si) for si in start_index]
    pad_upper = [max(0, si + rs - isz) for si, rs, isz in zip(start_index, roi_size, image_size)]
    
    # Apply padding if needed
    if any(pad_lower) or any(pad_upper):
        padded_image = sitk.ConstantPad(input_image, pad_lower, pad_upper, padding_value)
        padded_mask = sitk.ConstantPad(binary_mask, pad_lower, pad_upper, 0)  # Pad mask with 0s
    else:
        padded_image = input_image
        padded_mask = binary_mask
    
    # Update start_index based on the padding to be non-negative
    start_index_padded = [max(0, si) for si in start_index]
    
    # Crop the padded images
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(roi_size)
    roi_filter.SetIndex(start_index_padded)
    cropped_image = roi_filter.Execute(padded_image)
    cropped_mask = roi_filter.Execute(padded_mask)
    
    return cropped_image, cropped_mask

def Normalize_zero_mean_unit_variance(image):
    norm_image = (image - image.mean())/image.std()
    return norm_image

def Normalize_zero_one(image):
    norm_image = (image - np.min(image))/(np.max(image)-np.min(image))
    norm_image[norm_image>1]=1
    norm_image[norm_image<0]=0
    return norm_image

def Normalize_customize (img, given_mean, given_std):
    
#    normalize the image into a given mean and std
 
    norm_img = Normalize_zero_mean_unit_variance(img)
    norm_img = (norm_img + given_mean)* given_std
 
    return norm_img

def unique(list1): 
      
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    return unique_list

    
def strfind(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def importimage_3D ( image_path, anno_df, im_shape=(128,128,64), if_aug=0):
   
    img_data=[]
    lab_data1 = []# 7 Walker scores (bilateral) annotation
    lab_data2 = []# biological sex annotation
    case_names = anno_df['Sample ID'].tolist()
    for i, case_name in enumerate(case_names):
      
        '''
        import the image
        '''
        image_name = os.path.join(image_path, case_name + '.nii.gz')
        img_sitk = sitk.ReadImage(image_name)
        img = np.transpose(sitk.GetArrayFromImage(img_sitk), (1, 2, 0))# z, y, x -> y, x, z
        img = img.astype(np.float32)
        img[img<-1024.0] = -1024.0
        img[img>3071.0] = 3071.0
                                
        img_resize = resize(img, im_shape, order = 3 , clip=True) # resize
        # Normalize the array to the range [-1, 1]
        img_norm = 2 * (img_resize - (-1024)) / (3071 - (-1024)) - 1# normalize to [-1,1]
        img_norm = np.expand_dims(img_norm, axis=0) 
        img_data.append(img_norm)
        
        '''
        import the label from the dataframe
        '''
       
        lab1 = anno_df.iloc[i,1:8].values.astype(np.float32)
        lab2 = np.float32(anno_df.iloc[i,8]=='F')
                  
        lab_data1.append(lab1)
        lab_data2.append(lab2)
        
        
        if if_aug > 0:

            random_states = np.random.randint(0, 100, size=2)
            print('random_states: ', random_states)
            aug_index = np.arange(0,total_num_aug,1)
            slc = aug_index.tolist()
            
            AUG = augmentation_only_image(img, random_states)
            aug_items = AUG.select_aug(slc)

            for ii in range(0, len(aug_items)):
                
                i_aug = resize(aug_items[ii], im_shape, order = 3, clip=True)
                i_aug_norm = 2 * (i_aug - (-1024)) / (3071 - (-1024)) - 1
                i_aug_norm = np.expand_dims(i_aug_norm, axis=0)              
                img_data.append(i_aug_norm)
                
                lab_data1.append(lab1)
                lab_data2.append(lab2)
                                
    return img_data, lab_data1, lab_data2
    
def importimage_3D_image_skull (image_path, anno_df, seg_path=None, im_shape=(128,128,64), if_aug=0):
   
    img_data=[]
    lab_data1 = []# 7 Walker scores (bilateral) annotation
    lab_data2 = []# biological sex annotation
    case_names = anno_df['Sample ID'].tolist()
    for i, case_name in enumerate(case_names):
      
        '''
        import the image
        '''
        image_name = os.path.join(image_path, case_name + '.nii.gz')
        img_sitk = sitk.ReadImage(image_name)
        
        img = np.transpose(sitk.GetArrayFromImage(img_sitk), (1, 2, 0))
        img = img.astype(np.float32)
        img[img<-1024.0] = -1024.0
        img[img>3071.0] = 3071.0
                        
        img_resize = resize(img, im_shape, order = 3 , clip=True) # resize
        # Normalize the array to the range [-1, 1]
        img_norm = 2 * (img_resize - (-1024)) / (3071 - (-1024)) - 1# normalize to [-1,1]
        img_norm = np.expand_dims(img_norm, axis=0) 
                
        '''
        Generate the skull bone mask
        
        '''
        
        bone_mask_sitk = sitk.ReadImage(os.path.join(seg_path, case_name + '.nii.gz'))
        bone_mask_sitk = sitk.BinaryMorphologicalClosing(bone_mask_sitk, [5,5,5])
            
        bone_mask = np.transpose(sitk.GetArrayFromImage(bone_mask_sitk), (1, 2, 0))
        bone_mask = bone_mask.astype(np.float32)
        bone_mask_resize = resize(bone_mask, im_shape, order = 0 , clip=True)
        bone_mask_resize[bone_mask_resize>=0.5] = 1
        bone_mask_resize[bone_mask_resize<0.5] = 0
        bone_mask_resize = np.expand_dims(bone_mask_resize, axis=0)
        
        img_data.append(np.concatenate((img_norm, bone_mask_resize), axis = 0))
        
        '''
        import the label
        '''
       
        lab1 = anno_df.iloc[i,1:8].values.astype(np.float32)
        lab2 = np.float32(anno_df.iloc[i,8]=='F')
                  
        lab_data1.append(lab1)
        lab_data2.append(lab2)
        
        
        if if_aug > 0:

            random_states = np.random.randint(0, 100, size=2)
            print('random_states: ', random_states)
            aug_index = np.arange(0,total_num_aug,1)
            slc = aug_index.tolist()
            
            AUG = augmentation_wt_mask([img], [bone_mask], random_states)
            aug_items = AUG.select_aug(slc)

            for ii in range(0, len(aug_items)):
                
                img_aug = resize(aug_items[ii][0][0], im_shape, order = 3, clip=True)
                img_aug_norm = 2 * (img_aug - (-1024)) / (3071 - (-1024)) - 1
                img_aug_norm = np.expand_dims(img_aug_norm, axis=0)              
                                
                msk_aug = resize(aug_items[ii][1][0], im_shape, order = 0, clip=True)
                msk_aug[msk_aug>=0.5] = 1
                msk_aug[msk_aug<0.5] = 0
                msk_aug_ = np.expand_dims(msk_aug, axis=0)
                
                img_data.append(np.concatenate((img_aug_norm, msk_aug_), axis = 0))
                
                lab_data1.append(lab1)
                lab_data2.append(lab2)
                                
    return img_data, lab_data1, lab_data2    
        
                
def importimage_3D_skull (image_path, anno_df, seg_path=None, im_shape=(128,128,64), if_aug=0):
   

    img_data=[]
    lab_data1 = []# 7 Walker scores (bilateral) annotation
    lab_data2 = []# biological sex annotation
    case_names = anno_df['Sample ID'].tolist()
    for i, case_name in enumerate(case_names):
      
        '''
        import the image
        '''
        image_name = os.path.join(image_path, case_name + '.nii.gz')
        img_sitk = sitk.ReadImage(image_name)
        
        img = np.transpose(sitk.GetArrayFromImage(img_sitk), (1, 2, 0))
        img = img.astype(np.float32)
        img[img<-1024.0] = -1024.0
        img[img>3071.0] = 3071.0
                        
        img_resize = resize(img, im_shape, order = 3 , clip=True) 
     
        img_norm = np.expand_dims(img_resize, axis=0) 
                
        bone_mask_sitk = sitk.ReadImage(os.path.join(seg_path, case_name + '.nii.gz'))
        bone_mask_sitk = sitk.BinaryMorphologicalClosing(bone_mask_sitk, [5,5,5])
            
        bone_mask = np.transpose(sitk.GetArrayFromImage(bone_mask_sitk), (1, 2, 0))
        bone_mask = bone_mask.astype(np.float32)
        bone_mask_resize = resize(bone_mask, im_shape, order = 0 , clip=True)
        bone_mask_resize[bone_mask_resize>=0.5] = 1
        bone_mask_resize[bone_mask_resize<0.5] = 0
        bone_mask_resize = np.expand_dims(bone_mask_resize, axis=0)
        
        bone_mask_area = np.zeros(img_norm.shape).astype(np.float32)
        bone_mask_area[bone_mask_resize==1] = img_norm[bone_mask_resize==1]
        img_data.append(bone_mask_area)
        
        '''
        import the label
        '''
       
        lab1 = anno_df.iloc[i,1:8].values.astype(np.float32)
        lab2 = np.float32(anno_df.iloc[i,8]=='F')
                  
        lab_data1.append(lab1)
        lab_data2.append(lab2)
        
        
        if if_aug > 0:

            random_states = np.random.randint(0, 100, size=2)
            print('random_states: ', random_states)
            aug_index = np.arange(0,total_num_aug,1)
            slc = aug_index.tolist()
            
            AUG = augmentation_wt_mask([img], [bone_mask], random_states)

            aug_items = AUG.select_aug(slc)

            for ii in range(0, len(aug_items)):
                
                img_aug = resize(aug_items[ii][0][0], im_shape, order = 3, clip=True)
             
                img_aug_norm = np.expand_dims(img_aug, axis=0)              
                                
                msk_aug = resize(aug_items[ii][1][0], im_shape, order = 0, clip=True)
                msk_aug[msk_aug>=0.5] = 1
                msk_aug[msk_aug<0.5] = 0
                msk_aug_ = np.expand_dims(msk_aug, axis=0)
                
                bone_mask_area_aug = np.zeros(img_aug_norm.shape)
                bone_mask_area_aug[msk_aug_==1] = img_aug_norm[msk_aug_==1]
                
                img_data.append(bone_mask_area_aug)
                
                lab_data1.append(lab1)
                lab_data2.append(lab2)
                
                
    return img_data, lab_data1, lab_data2   
            
def generate_weights_for_walker_score(walker_pred_array):
    # Check for columns that are all zeros
    columns_all_zero = np.all(walker_pred_array == 0, axis=0)
    
    # Initialize a result array with all ones
    result_array = np.ones(walker_pred_array.shape[1], dtype=int)
    
    # Assign 0 to the position in result_array that corresponds to all-zero columns
    result_array[columns_all_zero] = 0

    return result_array

def average_walkder_scores(walker_score_all_test_pred, walker_score_all_weights):
    walker_pred = np.sum(walker_score_all_test_pred, axis =0)
    
    walker_score_all_weights = [
    [int(value) for value in sublist.strip('[]').split()] 
    for sublist in walker_score_all_weights
    ]

    # print(walker_score_all_weights)
    sum_walker_weights = np.sum(np.array(walker_score_all_weights), axis = 0)
    average_walker_pred = walker_pred/sum_walker_weights
    
    return average_walker_pred    

def round_to_nearest_walker_score(array):
    # Define the set of numbers to compare with
    reference_numbers = np.array([1, 2, 3, 4, 5])
    
    # Find the closest number from the reference set for each element in the array
    closest_numbers = np.array([reference_numbers[np.abs(reference_numbers - x).argmin()] for x in array.ravel()])
    
    # Reshape the array back to its original shape
    closest_numbers_reshaped = closest_numbers.reshape(array.shape)

    return closest_numbers_reshaped        
  
def mininum_distance_roc(tprs, fprs, thresholds):
    metric = np.sqrt((1-tprs)**2 + fprs**2)
    # index = np.argmin(metric)
    min_value = np.min(metric)  # Find the minimum value in the metric array
    min_indices = np.where(metric == min_value)[0]  # Find all indices where the metric equals the minimum value
    
    return thresholds[min_indices[-1]], tprs[min_indices[-1]], fprs[min_indices[-1]]  