#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


def create_dataset( dataset_class, data_num, seed): 
    dataset = [] # data数据对象的list集合
    print(f'start building {dataset_class}_dataset!')
    for system in ['neuron', 'biomass', 'veg_turb']:
        filepath = f"data_scaled/{system}/{dataset_class}/"
        filenames = os.listdir( filepath )
        seed = seed        
        data_ids = np.arange(len(filenames))
        np.random.seed(seed)
        np.random.shuffle(data_ids)
        
        for count in range(data_num):
        # 邻接矩阵转换成COO稀疏矩阵及转换
            results = np.load( filepath+filenames[data_ids[count]] )
            x = results['x']
            y = results['y']

            edge_list = np.array([[],[]])
            edge_index = torch.LongTensor(edge_list)

            # 节点及节点特征数据转换
            x = torch.FloatTensor(x)

            # 图标签数据转换
            y = torch.FloatTensor(y)

            data = Data(x=x, edge_index=edge_index, y=y) # 构建新型data数据对象
            dataset.append(data)
            if len(dataset) % 2500 == 0:
                print(f"already process {len(dataset)} {system} data for {dataset_class} dataset!")
    return dataset


# In[ ]:


torch.cuda.set_device(1)

seed = 1024
for time in range(2, 7):#[0,1]:
    train_dataset = create_dataset( 'train', 10500, seed)
    val_test_dataset = create_dataset( 'val_test', 4500, seed)

    random.seed(seed)
    random.shuffle(val_test_dataset)      
    test_dataset = val_test_dataset[-int(len(val_test_dataset)*2/3):]
    val_dataset = val_test_dataset[:len(val_test_dataset)-int(len(val_test_dataset)*2/3)]
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # 构建模型实例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net().to(device)
    model = GNN_RNNmodel.GIN_GRU( node_features= 20 , hidden_dim = 256, out_dim = 32, num_layers = 6 ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # 优化器，模型参数优化计算
    # scheduler = StepLR(optimizer, step_size=150*math.ceil((len(train_dataset))/256), gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=50, verbose=True, threshold=1e-5, min_lr=1e-5)  # patience=50*math.ceil((len(train_dataset))/256)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) # 加载训练数据集，训练数据中分成每批次20个图片data数据

    # 训练模型
    model.train() # 表示模型开始训练
    loss_func = nn.MSELoss() # 均方根误差损失函数
    # loss_compute = weight_MSEloss().to(device)
    mae_best, test, best_epoch = 1, 1, 0
    loss_epoch = np.zeros(1000)
    for epoch in range(1000): # 训练所有训练数据集20次
        loss_all = 0
        # 一轮epoch优化的内容
        for data in train_loader: # 每次提取训练数据集一批5张data图片数据赋值给data
            # data是batch_size图片的大小
            # print(data.edge_index)
            # print(data.batch.shape)
            # print(data.x.shape)
            data = data.to('cuda')
            optimizer.zero_grad() # 梯度清零
            output = model(data) # 前向传播，把一批训练数据集导入模型并返回输出结果，输出结果的维度是[20,2]
            y = data.y # 20张图片数据的标签集合，维度是[20]
    #         w_loss = data.w_loss
            # print(label)
            loss = loss_func(output, y) # 损失函数计算
    #         loss = loss_compute(output, y, w_loss)
            loss.backward() #反向传播
            loss_all += loss.item() * len(y) # 将最后的损失值汇总
            optimizer.step() # 更新模型参数
        scheduler.step(mae_best)
        if (epoch+1)%50==0:  # len(data)<128 & epoch%50==0:
#                 print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
            print(f"第{epoch+1}个epoch的学习率为{optimizer.param_groups[0]['lr']}" )
        loss_epoch[epoch] = loss_all / len(train_dataset)
    #     tmp = (loss_all / len(train_dataset)) # 算出损失值或者错误率

        if (epoch+1) % 5 == 0:
            print(f'epoch:{epoch+1}, loss:{loss_epoch[epoch]}')  # 显示出来的应该就是第5个，常规所认知的，比如第1200

        # 测试模型
        if (epoch+1) % 10 == 0:
            model.eval()

            val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False) # 加载训练数据集，训练数据中分成每批次20个图片data数据
            labels, preds = [], []
            for data in val_loader: # 每次提取训练数据集一批5张data图片数据赋值给data
                label = data.y.numpy() # 获取测试集的图片标签
                labels += list(label)
                data = data.to('cuda')
                pred = model(data).cpu().detach().numpy() # 将数据导入之前构造好的模型，返回输出结果维度是[20,2]
                preds += list(pred)
            mae = mean_absolute_error(labels, preds)

            test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False) # 加载训练数据集，训练数据中分成每批次20个图片data数据
            labels, preds = [], []
            for data in test_loader: # 每次提取训练数据集一批5张data图片数据赋值给data
                label = data.y.numpy() # 获取测试集的图片标签
                labels += list(label)
                data = data.to('cuda')
                pred = model(data).cpu().detach().numpy() # 将数据导入之前构造好的模型，返回输出结果维度是[20,2]
                preds += list(pred)
            mae_test = mean_absolute_error(labels, preds)

            print(f'本次测试，在验证集上的绝对值误差为:', mae)
            print(f'本次测试，在测试集上的绝对值误差为:', mae_test)
            if mae < mae_best:
                mae_best = mae
                test = mae_test
                best_epoch = epoch
                np.savez(f'prediction/CTPiCS/acc-{time}.npz', label=labels, preds = preds)  # hidden_nodes
                torch.save(model.state_dict(), f"prediction/CTPiCS/model_parameter-{time}.pkl")
                print(f'刷新误差率，模型已更新！')
            print(f'times={time}, epoch:{epoch+1}, 当前所有测试，在验证集上的绝对值误差最好为:{mae_best}，对应在测试集上为:{test}, best epoch:{best_epoch+1}')

            model.train()    

#             if epoch - best_epoch >= 300:
#                 print('!!Since best_mae is not updated for too long, break for next training!')
#                 break


    # save loss
    np.savez(f'prediction/CTPiCS/loss-{time}.npz', loss=loss_epoch)




