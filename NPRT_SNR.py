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

torch.cuda.set_device(0)


seed = 1024
for time in [4]:#range(5):
    for SNR in [15,10,5]:#np.arange(65, 39, -5):
        SNR = int(SNR)
#         noise_strength = round(10**noise_strength,3)
        
        dataset = [] # data数据对象的list集合
        print('start building dataset!')
        counts = np.arange(1000)
        np.random.seed(seed)
        np.random.shuffle(counts)
        flagval, flagtest = 1, 1
        for count in counts:
            for j in range(55):
                if not os.path.exists( f'data_neuron/data{count}_{j}.npz' ):
                    break
            # 数据转换
            # 邻接矩阵转换成COO稀疏矩阵及转换
                results = np.load(f'data_neuron/data{count}_{j}.npz')
                x = results['x']
                p = results['p']
                y = results['y']
#                 w_loss = results['w_loss']

                # 数据处理
                N1, N2 = x.shape[0], x.shape[1]
                noise = np.random.randn(N1, N2)
                noise_power = np.sum(noise*noise,axis=1)/N2
                Ps = np.sum(x*x,axis=1)/N2
                Pn = Ps/10**(SNR/10)
                n = noise*np.sqrt(Pn/noise_power).reshape(N1,1)                 
                
#                 noise_power = np.sum(noise*noise)/(N1*N2)
#                 Ps = np.sum(x*x)/(N1*N2)
#                 Pn = Ps/10**(SNR/10)
#                 n = noise*np.sqrt(Pn/noise_power)    
    
#                 noise = noise_strength*np.random.randn(x.shape[0], x.shape[1])
                x += n
            
                edge_list = np.array([[],[]])
                edge_index = torch.LongTensor(edge_list)

                # 节点及节点特征数据转换
                x = torch.FloatTensor(x)

                # 图标签数据转换
                y = torch.FloatTensor(y)
                p = torch.FloatTensor(p)
#                 w_loss = torch.FloatTensor(w_loss)


                # 构建数据集:一张图500个左右节点，每个节点30个特征，边的Coo稀疏矩阵，一个图预测指标p
#                 data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, p=p, y=y, w_loss=w_loss) # 构建新型data数据对象
                data = Data(x=x, edge_index=edge_index, p=p, y=y) # 构建新型data数据对象
                dataset.append(data)
                if len(dataset) % 2000 == 0:
                    print(f"already process {len(dataset)} data")
            if len(dataset) > 8450 and flagval:
                datasetvt = copy.deepcopy(dataset)
                flagval = 0
                print(f"val&test dataset is ok, has {len(datasetvt)} data!")
        print(f"already process {len(dataset)} data")
        print('dataset is ready!')

        # 切分数据集，分成训练和测试两部分
#         print("End : %s" % ti.ctime())
#         ti.sleep( 4000 )
        random.seed(seed)
        random.shuffle(datasetvt)      
        test_dataset = datasetvt[-int(len(datasetvt)*2/3):]
        val_dataset = datasetvt[:len(datasetvt)-int(len(datasetvt)*2/3)]
        print(len(test_dataset), len(val_dataset))
        train_dataset = dataset[len(test_dataset)+len(val_dataset):]

    
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
        loss_epoch = np.zeros(600)
        for epoch in range(600): # 训练所有训练数据集20次
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
                    np.savez(f'result/1000graph/SNR/1/acc-{SNR}-{time}.npz', label=labels, preds = preds)
                    torch.save(model.state_dict(), f"result/1000graph/SNR/1/model_parameter-{SNR}-{time}.pkl")
                    print(f'刷新误差率，模型已更新！')
                print(f'times={time}, SNR={SNR}, epoch:{epoch+1}, 当前所有测试，在验证集上的绝对值误差最好为:{mae_best}，对应在测试集上为:{test}, best epoch:{best_epoch+1}')

                model.train()    
                
            if epoch - best_epoch >= 300:
                print('!!Since best_mae is not updated for too long, break for next training!')
                break


        # save loss
        np.savez(f'result/1000graph/SNR/1/loss-{SNR}-{time}.npz', loss=loss_epoch)




