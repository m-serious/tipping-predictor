#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import networkx as nx
import numpy as np
import random
import scipy.integrate as spi
import multiprocessing as mp
from pyhdf.SD import *
from collections import defaultdict
# import torch
# from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
          


location, district = 'Af', 'h20v08'
filePath = f'real_data/{location}/{district}/'
filenames = os.listdir(filePath)

transdict = defaultdict(list)
for file in filenames:
#     print(file)
    trans = file.split('_')
    transdict[ (int(trans[1]), int(trans[2])) ].append(file)

cp_linear_step, w_width = 2.5, 20
keys = list(transdict.keys())  # [ key for key in transdict]
for i in range(len(keys)):
    for j in range( len(transdict[keys[i]]) ):
        x = np.load( filePath+f'{transdict[keys[i]][j]}/SI/x.npy' )
        cont_para = np.load( filePath+f'{transdict[keys[i]][j]}/SI/cont_para.npy' )    
        cp_linear = np.arange(cont_para[-1], cont_para[0], cp_linear_step)[::-1]
        
        len_cp = len(cp_linear)
        x_linear = np.zeros(( len_cp, len(x[0]) ))
        for k in range(len_cp-1):
            inter_id = np.where(cont_para > cp_linear[k])[0][-1]
            x_linear[k] = (cp_linear[k]-cont_para[inter_id])/(cont_para[inter_id+1]-cont_para[inter_id]) * (x[inter_id+1]-x[inter_id]) + x[inter_id]

        x_linear[-1] = x[-1]
        save_path = f'real_data/transects/{i}_{j}/'
        if not os.path.exists( save_path ):
            os.makedirs( save_path ) 
        for lt in range(1, len_cp+1-w_width):
            np.savez( save_path+f'{district}_{transdict[keys[i]][j]}_{lt}.npz', ts = x_linear[-lt-w_width:-lt].T, y = cont_para[-1], cp_max = cp_linear[0] ) 

    


