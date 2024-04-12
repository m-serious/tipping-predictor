#!/usr/bin/env python
# coding: utf-8


import os
import GNN_RNNmodel#gnn_model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.nn import GCNConv, GINConv
import scipy.sparse as sp
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
import networkx as nx
import math
import copy
import random
import scipy.integrate as spi
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time as ti



def create_dataset( dataset_class): 
    dataset = [] 
    print(f'start building {dataset_class}_dataset!')

    filepath = f"data_scaled/real/{dataset_class}/"
    filenames = os.listdir( filepath )

    for filename in os.listdir( filepath ):
        results = np.load( filepath+filename )
        x = results['x']
        y = results['y']

        edge_list = np.array([[],[]])
        edge_index = torch.LongTensor(edge_list)

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        data = Data(x=x, edge_index=edge_index, y=y) 
        dataset.append(data)
    return dataset



torch.cuda.set_device(0)

seed = 1024
for time in range(10):
    train_dataset = create_dataset( 'train' )
    test_dataset = create_dataset( 'test' )

    print(len(train_dataset), len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net().to(device)
    model = GNN_RNNmodel.GIN_GRU( node_features= 20 , hidden_dim = 256, out_dim = 32, num_layers = 6 )
    model.load_state_dict(torch.load('prediction/CTPiCS/'+f"model_parameter-5.pkl"))
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) 
    # scheduler = StepLR(optimizer, step_size=150*math.ceil((len(train_dataset))/256), gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=50, verbose=True, threshold=1e-5, min_lr=1e-5)  # patience=50*math.ceil((len(train_dataset))/256)
    train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True) 

    
    model.train() 
    loss_func = nn.MSELoss() 
    # loss_compute = weight_MSEloss().to(device)
    mae_best, test, best_epoch = 5, 1, 0
    loss_epoch = np.zeros(300)
    for epoch in range(300): 
        loss_all = 0
        for data in train_loader: 
            # print(data.edge_index)
            # print(data.batch.shape)
            # print(data.x.shape)
            data = data.to('cuda')
            optimizer.zero_grad() 
            output = model(data) 
            y = data.y 
            # print(label)
            loss = loss_func(output, y) 
    #         loss = loss_compute(output, y, w_loss)
            loss.backward() 
            loss_all += loss.item() * len(y) 
            optimizer.step() 
        scheduler.step(mae_best)
        if (epoch+1)%50==0: 
            print(f"Epoch{epoch+1}learning rate:{optimizer.param_groups[0]['lr']}" )
        loss_epoch[epoch] = loss_all / len(train_dataset)

        if (epoch+1) % 5 == 0:
            print(f'epoch:{epoch+1}, loss:{loss_epoch[epoch]}') 

        if (epoch+1) % 10 == 0:
            model.eval()

            val_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False) 
            labels, preds = [], []
            for data in val_loader: 
                label = data.y.numpy()
                labels += list(label)
                data = data.to('cuda')
                pred = model(data).cpu().detach().numpy() 
                preds += list(pred)
            mae = mean_absolute_error(labels, preds)

            print(f'Test dataset, mae loss:', mae)
            if mae < mae_best:
                mae_best = mae
#                 test = mae_test
                best_epoch = epoch
                np.savez(f'prediction/CTPiCS/2real/acc-{time}.npz', label=labels, preds = preds)  # hidden_nodes
                torch.save(model.state_dict(), f"prediction/CTPiCS/2real/model_parameter-{time}.pkl")
                print(f'Accuracy has been updated, and new model has been saved!')
            print(f'times={time}, epoch:{epoch+1}, until now，the best mae loss on Test dataset:{mae_best}, best epoch:{best_epoch+1}')

            model.train()    

        if epoch - best_epoch >= 200:
            print('!!Since best_mae is not updated for too long, break for next training!')
            break


    # save loss
    np.savez(f'prediction/CTPiCS/2real/loss-{time}.npz', loss=loss_epoch)




