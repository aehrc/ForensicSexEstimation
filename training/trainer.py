# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:04:01 2024

@author: Hang Min

Run training of the network
"""

import torch
from torch import nn
import numpy as np
import torch.optim as optim
from networks.callbacks import EarlyStopping
from sklearn import metrics


def train_model_two_outputs(net, train_dataloader, val_dataloader, batchSize, n_epochs, lr_rate, model_name, tensorboard_writer, num_class, device):
    # train the model to predict both walker scores and cranial sex
    loss1 = nn.MSELoss()
    loss2 = nn.BCELoss()
       
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_rate, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr = 1e-7, verbose=True)    
    # early stopping is basically disabled here, since we want the model to train for all epochs and save the best performing one on validation data.
    early_stopping = EarlyStopping(patience=n_epochs, verbose=True,  delta=0, path = model_name)
           
    for epoch in range(1, n_epochs + 1):
        
        print('Traning epoch: ' + str(epoch))
        
        # set the network to training
        
        Train_loss = []
        Train_loss1 = []  
        Train_loss2 = []  
        
        net.train()
        for step, (image, label1, label2) in enumerate(train_dataloader): 
            
            optimizer.zero_grad()
            
            image = image.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
                  
            pred1, pred2 = net.forward(image)   
            
            train_loss1 = loss1(pred1, label1)
            train_loss2 = loss2(pred2, label2)
            train_loss = train_loss1 + train_loss2
            
            Train_loss.append(train_loss.item())
            Train_loss1.append(train_loss1.item())
            Train_loss2.append(train_loss2.item())
                   
            print('[epoch %5d, step: %d, MSE loss = %f, BCE loss = %f, combined loss = %f '
                    % (epoch, step+1,  train_loss1.item(), train_loss2.item(), train_loss.item()))
                        
            train_loss.backward()           
            optimizer.step()
            
        tensorboard_writer.add_scalar("MSE_loss/train", np.average(Train_loss1), epoch)
        tensorboard_writer.add_scalar("BCE_loss/train", np.average(Train_loss2), epoch)
        tensorboard_writer.add_scalar("Combined_loss/train", np.average(Train_loss), epoch)
        
        print('validation at epoch: ' + str(epoch)) 
       
        net.eval() 
            
        Val_auc = 0
        Val_pred = []
        Val_label = []
        Val_loss1 = []
        Val_loss2 = []
        Val_loss = []
      
        with torch.no_grad():
        
            for j, (val_image, val_label1, val_label2) in enumerate(val_dataloader):
                               
                val_image = val_image.to(device)
              
                val_label1 = val_label1.to(device)
                val_label2 = val_label2.to(device)
               
                val_pred1, val_pred2 = net(val_image) #.forward      
                               
                val_loss1 = loss1(val_pred1, val_label1)# the order should be pred first and then label
                val_loss2 = loss2(val_pred2, val_label2)
                val_loss = val_loss1 + val_loss2
                
                Val_loss1.append(val_loss1.item())
                Val_loss2.append(val_loss2.item())
                Val_loss.append(val_loss.item())
                
                Val_pred = Val_pred + list(val_pred2.detach().cpu().numpy()[:,0])
                Val_label = Val_label + list(val_label2.detach().cpu().numpy()[:,0])
                                                                     
        print('Prediction probability on validation set: ', Val_pred)
        print('length of the pred prob: ', len(Val_pred))
        print('Predicted label on validation set: ', Val_label)
        print('length of the pred label: ',len(Val_label))
                
        Validation_loss = np.average(Val_loss)
        
        print('Validation MSE loss = %f, Validation BCE loss = %f, combined loss = %f '
                % (np.average(Val_loss1), np.average(Val_loss2), Validation_loss))
               
        # Calculating the roc metric on validation set and AUC
        fpr, tpr, thresholds = metrics.roc_curve(np.array(Val_label), np.array(Val_pred))
        Val_auc = metrics.auc(fpr, tpr)
        
        print("Validation AUC = {}".format(Val_auc))
                               
        early_stopping(Validation_loss, net)
        
        scheduler.step(Validation_loss)
        
        tensorboard_writer.add_scalar("MSE_loss/val", np.average(Val_loss1), epoch)
        tensorboard_writer.add_scalar("BCE_loss/val", np.average(Val_loss2), epoch)
        tensorboard_writer.add_scalar("combined_loss/val", Validation_loss, epoch)
        tensorboard_writer.add_scalar("AUC/val", Val_auc, epoch)
            
        print('Epoch {}, lr {}'.format(
        epoch, optimizer.param_groups[0]['lr']))
                
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    net.load_state_dict(torch.load(model_name))
    
    print('Finish Training')

    return  net

def train_model_single_output(net, train_dataloader, val_dataloader, batchSize, n_epochs, lr_rate, model_name, tensorboard_writer, num_class, device):
    # Only train the model to predict cranial sex
    loss2 = nn.BCELoss()
       
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_rate, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr = 1e-7, verbose=True)    
    early_stopping = EarlyStopping(patience=n_epochs, verbose=True,  delta=0, path = model_name)
           
    for epoch in range(1, n_epochs + 1):
        
        print('Traning epoch: ' + str(epoch))
                
        Train_loss = []
             
        net.train()
        for step, (image, _, label2) in enumerate(train_dataloader): 
            
            optimizer.zero_grad()
            
            image = image.to(device)
            label2 = label2.to(device)
                  
            pred2 = net.forward(image)   
            
            train_loss2 = loss2(pred2, label2)
            
            Train_loss.append(train_loss2.item())
                   
            print('[epoch %5d, step: %d, BCE loss = %f '
                    % (epoch, step+1,  train_loss2.item()))
                        
            train_loss2.backward()           
            optimizer.step()
            
        tensorboard_writer.add_scalar("BCE_loss/train", np.average(Train_loss), epoch)
            
        print('validation at epoch: ' + str(epoch)) 
       
        net.eval() 
            
        Val_auc = 0
        Val_pred = []
        Val_label = []
        Val_loss = []
      
        with torch.no_grad():
        
            for j, (val_image, _, val_label2) in enumerate(val_dataloader):
                               
                val_image = val_image.to(device)              
                val_label2 = val_label2.to(device)
               
                val_pred2 = net(val_image) 
                               
                val_loss2 = loss2(val_pred2, val_label2)
                
                Val_loss.append(val_loss2.item())
                
                Val_pred = Val_pred + list(val_pred2.detach().cpu().numpy()[:,0])
                Val_label = Val_label + list(val_label2.detach().cpu().numpy()[:,0])
                                                                     
        print('Prediction probability on validation set: ', Val_pred)
        print('length of the pred prob: ', len(Val_pred))
        print('Predicted label on validation set: ', Val_label)
        print('length of the pred label: ',len(Val_label))
                
        Validation_loss = np.average(Val_loss)
        
        print(' Validation BCE loss = %f '
                % (Validation_loss))
               
        # Calculating the roc metric on validation set and AUC
        fpr, tpr, thresholds = metrics.roc_curve(np.array(Val_label), np.array(Val_pred))
        Val_auc = metrics.auc(fpr, tpr)
        
        print("Validation AUC = {}".format(Val_auc))
                               
        early_stopping(Validation_loss, net)
        
        scheduler.step(Validation_loss)
        
        tensorboard_writer.add_scalar("BCE_loss/val", Validation_loss, epoch)
        tensorboard_writer.add_scalar("AUC/val", Val_auc, epoch)
            
        print('Epoch {}, lr {}'.format(
        epoch, optimizer.param_groups[0]['lr']))
                
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    net.load_state_dict(torch.load(model_name))
    
    print('Finish Training')

    return  net


