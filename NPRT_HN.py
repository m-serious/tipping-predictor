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

torch.cuda.set_device(1)

seed = 1024
for time in range(5):
    for removal_ratio in np.arange(0, 1, 0.09):
        removal_ratio = round(removal_ratio,2)
        
        dataset = [] 
        print('start building dataset!')
        counts = np.arange(1000)
        np.random.seed(seed)
        np.random.shuffle(counts)
        flagval, flagtest = 1, 1
        for count in counts:
            for j in range(55):
                if not os.path.exists( f'data_neuron/data{count}_{j}.npz' ):
                    break

                results = np.load(f'data_neuron/data{count}_{j}.npz')
                x = results['x']
#                 print(len(x))
                p = results['p']
                y = results['y']
#                 w_loss = results['w_loss']

              
                removal_num = int( removal_ratio * len(x) )
                rest = np.random.choice(len(x), len(x)-removal_num, replace=False)
                x = x[rest]
            
                edge_list = np.array([[],[]])
                edge_index = torch.LongTensor(edge_list)

                x = torch.FloatTensor(x)

                y = torch.FloatTensor(y)
                p = torch.FloatTensor(p)

                data = Data(x=x, edge_index=edge_index, p=p, y=y) 
                dataset.append(data)
                if len(dataset) % 2000 == 0:
                    print(f"already process {len(dataset)} data")
            if len(dataset) > 8450 and flagval:
                datasetvt = copy.deepcopy(dataset)
                flagval = 0
                print(f"val&test dataset is ok, has {len(datasetvt)} data!")
        print(f"already process {len(dataset)} data")
        print('dataset is ready!')

        random.seed(seed)
        random.shuffle(datasetvt)      
        test_dataset = datasetvt[-int(len(datasetvt)*2/3):]
        val_dataset = datasetvt[:len(datasetvt)-int(len(datasetvt)*2/3)]
        print(len(test_dataset), len(val_dataset))
        train_dataset = dataset[len(test_dataset)+len(val_dataset):]


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNN_RNNmodel.GIN_GRU( node_features= 20 , hidden_dim = 256, out_dim = 32, num_layers = 6 ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005) 
        # scheduler = StepLR(optimizer, step_size=150*math.ceil((len(train_dataset))/256), gamma=0.5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=50, verbose=True, threshold=1e-5, min_lr=1e-5)  # patience=50*math.ceil((len(train_dataset))/256)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) 


        model.train() 
        loss_func = nn.MSELoss() 
        # loss_compute = weight_MSEloss().to(device)
        mae_best, test, best_epoch = 1, 1, 0
        loss_epoch = np.zeros(600)
        for epoch in range(600): 
            loss_all = 0
            
            for data in train_loader:
                # print(data.edge_index)
                # print(data.batch.shape)
                # print(data.x.shape)
                data = data.to('cuda')
                optimizer.zero_grad() 
                output = model(data)
                y = data.y 
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

                val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
                labels, preds = [], []
                for data in val_loader: 
                    label = data.y.numpy() 
                    labels += list(label)
                    data = data.to('cuda')
                    pred = model(data).cpu().detach().numpy() 
                    preds += list(pred)
                mae = mean_absolute_error(labels, preds)

                test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False) 
                labels, preds = [], []
                for data in test_loader:
                    label = data.y.numpy()
                    labels += list(label)
                    data = data.to('cuda')
                    pred = model(data).cpu().detach().numpy()
                    preds += list(pred)
                mae_test = mean_absolute_error(labels, preds)

                print(f'Val dataset, mae loss:', mae)
                print(f'Test dataset, mae loss:', mae_test)
                if mae < mae_best:
                    mae_best = mae
                    test = mae_test
                    best_epoch = epoch
                    np.savez(f'result/hidden_nodes/acc-{removal_ratio}-{time}.npz', label=labels, preds = preds)
                    torch.save(model.state_dict(), f"result/hidden_nodes/model_parameter-{removal_ratio}-{time}.pkl")
                    print(f'Accuracy has been updated, and new model has been saved!')
                print(f'times={time}, nodes_removal_ratio={removal_ratio}, epoch:{epoch+1}, until now，the best mae loss on Val dataset:{mae_best}，Test dataset:{test}, best epoch:{best_epoch+1}')

                model.train()    
                
            if epoch - best_epoch >= 300:
                print('!!Since best_mae is not updated for too long, break for next training!')
                break


        # save loss
        np.savez(f'result/hidden_nodes/loss-{removal_ratio}-{time}.npz', loss=loss_epoch)




