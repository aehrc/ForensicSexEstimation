# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:32:27 2024

@author: Hang Min
Test on sample data
"""

import os
import sys
# Add the parent directory to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
import numpy as np
import SimpleITK as sitk
import time
import argparse
import torch
from torch.utils.data import DataLoader
from networks.resnet import ResNet_two_output, ResNet_auxiliary, ResNet_single_output
import pandas as pd
from data.utils import average_walkder_scores, round_to_nearest_walker_score
from testing.predict import test_model_two_ouputs, test_model_single_output, assign_value_to_key
from natsort import natsorted
from torch.utils.data import TensorDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdl_name', help='name of the output model', type=str, default="ResNet.pt")
    parser.add_argument('--mdl_type', help='the type of models: ResNet_two_output, ResNet_auxiliary and ResNet_single_output', type=str, default="ResNet_auxiliary")
    parser.add_argument('--num_metrics', help='number of Walker traits (bilateral)', type=int, default=7)
    parser.add_argument('--num_classes', help='number of biological sex classes, 0 (male) or 1 (female). Since this is binary classification, it will be set as 1.', type=int, default=1)
            
    args = parser.parse_args()        
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Mapping of model types to functions
    model_functions = {
        "ResNet_two_output": ResNet_two_output,
        "ResNet_auxiliary": ResNet_auxiliary,
        "ResNet_single_output": ResNet_single_output
    }
    
    model_function = model_functions.get(args.mdl_type)
    
    start = time.time() 
    
    data_path = os.path.join(parent_folder_path, 'Cranial CT data', 'sample_data')
    
    case_names = natsorted(os.listdir(data_path))
    
    image_data = []
    
    # placeholders simply to fill in the variable position in dataset
    label1_placeholder = []
    label2_placeholder = []
    
    for case_name in case_names:
        
        image = sitk.ReadImage(os.path.join(data_path, case_name))
        
        image_np = np.transpose(sitk.GetArrayFromImage(image), (1, 2, 0)) 
        
        image_np = np.expand_dims(image_np, axis=0)
        image_data.append(image_np)
        
        # placeholders simply to fill in the variable position in dataset, can be any value
        label1_placeholder.append(0)
        label2_placeholder.append(0)
        
    image_data = np.array(image_data)    
    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.from_numpy(image_data).float()
   
    label1_placeholder = np.array(label1_placeholder)
    label1_placeholder = torch.from_numpy(label1_placeholder).float()
    label2_placeholder = np.array(label2_placeholder)
    label2_placeholder = torch.from_numpy(label2_placeholder).float()
    
    # Create a TensorDataset with image data and fake labels
    dataset = TensorDataset(image_tensor, label1_placeholder, label2_placeholder)
       
    # Create a DataLoader with batch size of 1 (or any batch size you need)
    test_dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = model_function(channel=1, filters=[32, 64, 128, 256], num_metrics=args.num_metrics, num_classes=args.num_classes)
        
    Test_Pred1s = []
    Test_Pred2s = []
    # Walker traits labels
    keys = ['GLA\n(1 g)', 'SUP-L\n(2 om)', 'SUP-R\n(3 om)', 'MEN\n(4 em)', 'MAS-L\n(5 ms)', 'MAS-R\n(6 ms)', 'NUC\n(7 nu)']
    # print('Walker traits: ', keys)
    
    
    if args.mdl_type == 'ResNet_two_output' or args.mdl_type == 'ResNet_auxiliary':
        if_walker_score = True
        for fold in range(0,5):
           
            weight_path = os.path.join(parent_folder_path, 'model__' + args.mdl_type + '__' +'skull', 'exp' + str(fold), args.mdl_name)
            
            model.to(device)
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))#, map_location=torch.device('cpu')
            model.eval()
                         
            test_pred1s, test_pred2s = test_model_two_ouputs (test_dataLoader, model, device)
                                        
            test_pred1s = np.array(test_pred1s)
            test_pred2s = np.array(test_pred2s)
            
            Test_Pred1s.append(test_pred1s)
            Test_Pred2s.append(test_pred2s)
               
    elif args.mdl_type == 'ResNet_single_output': 
        if_walker_score = False
        for fold in range(0,5):
            weight_path = os.path.join(parent_folder_path, 'model__' + args.mdl_type + '__' + 'skull', 'exp' + str(fold), args.mdl_name)
            
            model.to(device)
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))#, map_location=torch.device('cpu')
            model.eval()
                                 
            test_pred2s = test_model_single_output (test_dataLoader, model, device)
            
            test_pred2s = np.array(test_pred2s)
            Test_Pred2s.append(test_pred2s)
                                        
    else:
        raise ValueError("Unrecognized model type!")
        
        
    '''
    Load the output spreadsheet and calculate the final walker score and sex estimation
    '''
    # load the probability thresholds and walker score weights (caculated on validation set) for each fold
    
    thresh_weights_excel_path = os.path.join(parent_folder_path, 'model__' + args.mdl_type + '__' + 'skull', 'Thresholds_Weights.xlsx')
    excel_file = pd.ExcelFile(thresh_weights_excel_path)
    prob_thresholds = pd.read_excel(thresh_weights_excel_path, sheet_name = 'Prob Thresholds')
    if 'Walker Weights' in excel_file.sheet_names:
        walker_weights = pd.read_excel(thresh_weights_excel_path, sheet_name='Walker Weights', dtype={'ColumnName': float})
        print("Sheet 'Walker Weights' loaded successfully.")
    else:
        print("Sheet 'Walker Weights' does not exist in the Excel file.")
        
    all_test_predicts = []
    walker_score_all_test_pred = []
    
    for fold in range(0,5):
        
        pred = np.array(Test_Pred2s[fold])
        threshold = prob_thresholds['Prob Thresholds'].tolist()[fold]
        predict = np.int8(pred.copy()>threshold)
        all_test_predicts.append(predict)
               
    if if_walker_score:
        
        avg_walker_score_all_pred = average_walkder_scores(Test_Pred1s, walker_weights['Walker Weights'].tolist())
        walker_score_final_pred = round_to_nearest_walker_score(avg_walker_score_all_pred)
        
    vote_test_predict = np.float32(np.mean(all_test_predicts, axis = 0)>0.5)
    pred_str = np.array(['M']*len(vote_test_predict))
    pred_str[vote_test_predict==1]='F'
    
    final_test_results = {'Sample ID': case_names,
                          'Pred sex': list(pred_str),                                 
                          }
    
    print(final_test_results)