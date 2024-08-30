# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:41:03 2024

@author: Hang Min

Calculate test performance
"""
import os
import sys
# Add the parent directory to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
import time
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve, recall_score


if __name__ == '__main__':
    
    start = time.time() 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdl_type', help='the type of models: ResNet_two_output, ResNet_auxiliary and ResNet_single_output', type=str, default="ResNet_two_output")
    parser.add_argument('--input_mode', help='the type of input: whole CT image, image and skull mask, or skull', type=str, default='skull')
    
    args = parser.parse_args()    
    
    test_output_path = os.path.join(parent_folder_path, 'model__' + args.mdl_type +'__'+ args.input_mode, 'test', 'test_pred_output.xlsx')
    test_performance_path = os.path.join(parent_folder_path, 'model__' + args.mdl_type +'__'+ args.input_mode, 'test', 'Final performance.xlsx')
    case_partition_path = '../data_partition/Case_partition.xlsx'
    test_gt_df = pd.read_excel(case_partition_path, sheet_name = 'Test')
    test_gt = np.array(test_gt_df['Reported Sex'].tolist())
    test_gt = np.float32(test_gt=='F')
    
    all_AUCs = []
    
    for f in range(0,5):
        test_output = pd.read_excel(test_output_path, sheet_name = 'Test'+str(f))
        test_prob = np.array(test_output['Pred prob'].tolist())
        fprs, tprs, thresholds = roc_curve(test_gt, test_prob)
        AUC = auc(fprs, tprs)
        
        all_AUCs.append(AUC)
        
    average_auc = np.mean(all_AUCs)
    std_auc = np.std(all_AUCs)
    
    test_pred = np.array(pd.read_excel(test_performance_path, sheet_name = 'Performance')['Pred sex'].tolist())
    test_pred = np.float32(test_pred=='F')
    
    accuracy = accuracy_score(test_gt, test_pred)
    
    tn, fp, fn, tp = confusion_matrix(test_gt, test_pred).ravel()
    
    # Sensitivity (Recall)
    sensitivity = recall_score(test_gt, test_pred)
    
    # Specificity
    specificity = tn / (tn + fp)
    
    metrics = {
    'Average AUC': [average_auc],
    'AUC Std Dev': [std_auc],
    'Accuracy': [accuracy],
    'Sensitivity': [sensitivity],
    'Specificity': [specificity]
    }
    
    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Specify the output file path
    output_metrics_path = os.path.join(parent_folder_path, 'model__' + args.mdl_type +'__'+ args.input_mode, 'test', 'Evaluation metrics.xlsx')
    
    # Save the DataFrame to an Excel file
    metrics_df.to_excel(output_metrics_path, index=False)
    
    print(f"Metrics have been saved to {output_metrics_path}")
