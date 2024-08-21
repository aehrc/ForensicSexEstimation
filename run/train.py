# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:02:33 2024

@author: Hang Min
"""

import os
import sys
# Add the parent directory to the Python path
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder_path)
import time
import argparse
import torch
from torch.utils.data import DataLoader
from networks.resnet import ResNet_two_output, ResNet_auxiliary, ResNet_single_output
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from data.Custom_data_loader import load_data
from data.Custom_data_loader import load_data_image_skull
from data.Custom_data_loader import load_data_skull
from training.trainer import train_model_two_outputs, train_model_single_output

if __name__ == '__main__':
    
    start = time.time() 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdl_name', help='name of the output model', type=str, default="ResNet.pt")
    parser.add_argument('--mdl_type', help='the type of models: ResNet_two_output, ResNet_auxiliary and ResNet_single_output', type=str, default="ResNet_two_output")
    parser.add_argument('--input_mode', help='the type of input: whole CT image, image and skull mask, or skull', type=str, default='skull')
    # parser.add_argument('--number_output', help='whether to generate 2 outputs (sex and walker scores), or 1 output (just sex)', type=int, default=2)
    parser.add_argument('--fold', help='cross validation folder', type=int, default=0)
    parser.add_argument('--num_epoch', help='number of training epochs', type=int, default=100)
    parser.add_argument('--batch', help='batch size', type=int, default=4)
    parser.add_argument('--num_metrics', help='number of Walker traits (bilateral)', type=int, default=7)
    parser.add_argument('--num_classes', help='number of biological sex classes, 0 (male) or 1 (female). Since this is binary classification, it will be set as 1.', type=int, default=1)
    parser.add_argument('--aug', help='number of augmentation option', type=int, default=3)
    # parser.add_argument('--log_dir', help='tensorboard save path', default='tensorboard/ResNet/')
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
        
    args = parser.parse_args()    
    learning_rate = args.lr
    batch = args.batch
    n_epochs = args.num_epoch
    
    # Mapping of model types to functions
    model_functions = {
        "ResNet_two_output": ResNet_two_output,
        "ResNet_auxiliary": ResNet_auxiliary,
        "ResNet_single_output": ResNet_single_output
    }
    
    model_function = model_functions.get(args.mdl_type)
    model_save_folder = os.path.join(parent_folder_path, 'model__' + args.mdl_type + '__' + args.input_mode, 'exp' + str(args.fold))
    if not os.path.exists(model_save_folder):        
        os.makedirs(model_save_folder)
        
    model_name = os.path.join(model_save_folder, args.mdl_name)
    
    '''
    Build model
    '''    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
    Start training the network
    '''    
    data_path = '/datasets/work/hb-radiationtqa/work/Cranial CT data/Cranial CT nifti isotropic crop/'
    seg_path = '/datasets/work/hb-radiationtqa/work/Cranial CT data/Cranial CT isotropic segmentations crop/'
    
    train_spreadsheet = pd.read_excel('/datasets/work/hb-c-radiation/work/Python_env/myenv_ct_gender/cranial_ct_gender_classification/data_process/Case_partition.xlsx', 'Train'+str(args.fold))
    
    val_spreadsheet = pd.read_excel('/datasets/work/hb-c-radiation/work/Python_env/myenv_ct_gender/cranial_ct_gender_classification/data_process/Case_partition.xlsx', 'Val'+str(args.fold))

    if  args.input_mode == 'image':
        # Use image only input
        print('Using whole image as input:')
        train_dataset = load_data(data_path, train_spreadsheet, im_shape = (128,128,128), if_aug = args.aug)
        
        train_dataLoader = DataLoader(
            dataset = train_dataset,
            batch_size = batch,
            shuffle = True
        )
        print('Number of training images: ', len(train_dataset))
             
        val_dataset = load_data(data_path, val_spreadsheet, im_shape = (128,128,128), if_aug = 0)
        
        val_dataLoader = DataLoader(
            dataset = val_dataset,
            batch_size = batch,
            shuffle = False
        )
    
        print('Number of validation images: ', len(val_dataset))
                       
        model  = model_function(channel=1, filters=[32, 64, 128, 256], num_metrics=args.num_metrics, num_classes=args.num_classes)
               
        tensorboard_writer = SummaryWriter(log_dir= model_save_folder)
        
        # Carry out training
        print(model)
               
        '''
        #tensorboard callbacks
        '''
        print("Training ...")
        
        model.to(device)
        
        if args.mdl_type == 'ResNet_two_output' or args.mdl_type == 'ResNet_auxiliary':           
            train_model_two_outputs(model, train_dataLoader, val_dataLoader, batch, n_epochs, learning_rate, model_name, tensorboard_writer, args.num_classes, device)
            
        elif args.mdl_type == 'ResNet_single_output':         
            train_model_single_output(model, train_dataLoader, val_dataLoader, batch, n_epochs, learning_rate, model_name, tensorboard_writer, args.num_classes, device)
            
        else:
            raise ValueError("Unrecognized model type!")
            
    elif args.input_mode == 'image_skull':
        # using image + bone mask
        print('Using image and skull mask as 2 channel inputs, skull mask generated by total segmentator:')
        train_dataset = load_data_image_skull(data_path, train_spreadsheet, seg_path=seg_path, im_shape = (128,128,128), 
                                                 if_aug = args.aug)
        
        train_dataLoader = DataLoader(
            dataset = train_dataset,
            batch_size = batch,
            shuffle = True
        )
        
        print('Number of training images: ', len(train_dataset))
             
        val_dataset = load_data_image_skull(data_path, val_spreadsheet, seg_path=seg_path, im_shape = (128,128,128), 
                                               if_aug = 0)
        
        val_dataLoader = DataLoader(
            dataset = val_dataset,
            batch_size = batch,
            shuffle = False
        )
        
        print('Number of validation images: ', len(val_dataset))
                       
        model = model_function(channel=2, filters=[32, 64, 128, 256], num_metrics=args.num_metrics, num_classes=args.num_classes)
                  
        tensorboard_writer = SummaryWriter(log_dir = model_save_folder)
        
        # Carry out training
        print(model)
             
        '''
        #tensorboard callbacks
        '''
        print("Training ...")
        
        model.to(device)
         
        if args.mdl_type == 'ResNet_two_output' or args.mdl_type == 'ResNet_auxiliary':           
            train_model_two_outputs(model, train_dataLoader, val_dataLoader, batch, n_epochs, learning_rate, model_name, tensorboard_writer, args.num_classes, device)
            
        elif args.mdl_type == 'ResNet_single_output':         
            train_model_single_output(model, train_dataLoader, val_dataLoader, batch, n_epochs, learning_rate, model_name, tensorboard_writer, args.num_classes, device)
            
        else:
            raise ValueError("Unrecognized model type!")
        
    elif args.input_mode == 'skull':
        # Using image and bone mask intersection (bone mask region only)
        print('Using skull region intersection between image and skull mask as input, skull mask generated by total segmentator:')
        train_dataset = load_data_skull(data_path, train_spreadsheet, seg_path=seg_path, im_shape = (128,128,128), 
                                                if_aug = args.aug)
        
        train_dataLoader = DataLoader(
            dataset = train_dataset,
            batch_size = batch,
            shuffle = True
        )
        
        print('Number of training images: ', len(train_dataset))
             
        val_dataset = load_data_skull(data_path, val_spreadsheet, seg_path=seg_path, im_shape = (128,128,128), 
                                              if_aug = 0)
        
        val_dataLoader = DataLoader(
            dataset = val_dataset,
            batch_size = batch,
            shuffle = False
        )
        
        print('Number of validation images: ', len(val_dataset))
                               
        model = model_function(channel=1, filters=[32, 64, 128, 256], num_metrics=args.num_metrics, num_classes=args.num_classes)
              
        tensorboard_writer = SummaryWriter(log_dir= model_save_folder)
        
        # Carry out training
        print(model)
               
        '''
        #tensorboard callbacks
        '''
        print("Training ...")
        
        model.to(device)
         
        if args.mdl_type == 'ResNet_two_output' or args.mdl_type == 'ResNet_auxiliary':           
            train_model_two_outputs(model, train_dataLoader, val_dataLoader, batch, n_epochs, learning_rate, model_name, tensorboard_writer, args.num_classes, device)
            
        elif args.mdl_type == 'ResNet_single_output':         
            train_model_single_output(model, train_dataLoader, val_dataLoader, batch, n_epochs, learning_rate, model_name, tensorboard_writer, args.num_classes, device)
            
        else:
            raise ValueError("Unrecognized model type!")
        
    else:
        
        raise ValueError("The input type is not recognized!")

  
    end = time.time()
    elapsed = end - start
    elapsed = round(elapsed, 2)
    print("Running took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")