# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:02:19 2023

@author: Hang Min
"""
import sys
import torch

def test_model_two_ouputs (dataLoader, model, device):
      
    pred1s = []# Walker scores
    pred2s = []# Cranial sex
    with torch.no_grad():
    
        for n, (image, _, _) in enumerate(dataLoader):
        
            '''
            Unpack the data and start testing
            '''
            
            image = image.to(device)
            
            pred1, pred2 = model(image)       
            
            pred1s = pred1s + list(pred1.detach().cpu().numpy())
            pred2s = pred2s + list(pred2.detach().cpu().numpy()[:,0])
            
    return pred1s, pred2s

def test_model_single_output (dataLoader, model, device):
      
    pred2s = []# Cranial sex
    with torch.no_grad():
    
        for n, (image, _, _) in enumerate(dataLoader):
        
            '''
            Unpack the data and start testing
            '''
            
            image = image.to(device)
            
            pred2 = model(image)       
            
            pred2s = pred2s + list(pred2.detach().cpu().numpy()[:,0])
            
    return pred2s

def assign_value_to_key (Dict, keys, content):
    
    for i, k in enumerate(keys):
        Dict[k] = list(content[:, i])
        
    return Dict