import networkx as nx
import numpy as np
import random
import time
import scipy.integrate as spi
import multiprocessing as mp
import os

# os.environ["MKL_NUM_THREADS"] = '4'
# os.environ["NUMEXPR_NUM_THREADS"] = '4'
# os.environ["OMP_NUM_THREADS"] = '4'

# Define network and hyper-parameters
# md = 5
# n = 500
mu, delta = 10, 1

# define dynamics
def diff(xr, Ar):
    dd = -xr + Ar.dot(1 / (1 + np.exp(mu-delta*xr)))   
    return dd

def rk4( theta, Ar, h, diff ):  # [node, hidden_channel]
#     h = 0.01
    k1 = diff(theta, Ar)
    k2 = diff(theta + h * k1 / 2, Ar)
    k3 = diff(theta + h * k2 / 2, Ar)
    k4 = diff(theta + h * k3, Ar)
    theta = theta + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return theta


def simulation(count):
    
    time_start = time.time()    
    _, decimals = str(time_start).split('.')  # 取当前时间（float）的小数部分作为种子
    seed = int(decimals)
    np.random.seed(seed)
    
    md = 4.5 + 0.01*np.random.randint(-150,150,size=1)
    n = 100 + np.random.randint(-90, 100,size=1)
    md, n = md[0], int(n)

    # g = scale_free( n, md, seed )
    g = nx.erdos_renyi_graph( n=n, p=md/(n-1), directed=True, seed=int(seed) ) #<k> << n
    g.remove_nodes_from(list(nx.isolates(g)))
    mapping = dict(zip(g, range(0, len(g))))
    g = nx.relabel_nodes(g, mapping)
    
    w = np.random.rand( len(g.edges()) ) + (np.random.rand()*10+10) #uniform (0,1)+0.2,  mean value=0.5+0.2=0.7;  w>0

    source, target = zip(*g.edges())
    weight_edges = zip( source, target, w )
    g.add_weighted_edges_from( weight_edges )
    N = len(g)

    #generate weight adjacency matrix
    A = nx.to_numpy_array( g ) #element: out_link
    for s,t, w in g.edges( data=True ):
        A[s,t] = w['weight']
    A = A.T #elemnt: in_link

    x = np.random.rand( len(g.nodes()) ) * 10 + 8 #init
     
    ## Simulate the system
#     ni = 101
    ni = 100
    rx = np.zeros((ni,N))
#     t = np.arange(0., 3, 0.01)
    h = 0.01
    t = np.arange(0., 15, h)
    p = 1 # control the weight
    dom_eval_re = np.zeros(ni)

#     xi, info_dict = spi.odeint(diff, x, t, args=(A, mu, delta),
#                               full_output=True,
#                               hmin=1e-14,
#                               mxhnil=0)
    xi = np.zeros(( len(t), N ))
    xi[0] = rk4( x, A, h, diff )
    pi = np.arange(p, p-0.01, -0.01/len(t))
    for j in range(len(t)-1):
        At = A * pi[j+1]
        xi[j+1] = rk4( xi[j], At, h, diff )
    
    rx[0] = xi[-1]

    # Compute jacobian matrix
    xx = np.tile(rx[0],(N,1))
#     jacobian = A*(np.exp(mu-xx)/(1+np.exp(mu-xx))**2)
    jacobian = At*(np.exp(mu-xx)/(1+np.exp(mu-xx))**2)
    row, col = np.diag_indices_from(jacobian)
    jacobian[row,col] = -1

    # Compute eigenvalues
    evals = np.linalg.eigvals(jacobian)

    # Compute the real part of the dominant eigenvalue (smallest magnitude)
    re_evals = [lam.real for lam in evals]
    ind_max = np.array(re_evals).argmax()
    dom_eval_re[0] = max(re_evals)
    
    print("Number:", count)
    print(f"N:{N}, edges:{len(g.edges())}, md:{md}")
    print("initial average:", np.average(xi[-1]))


    ## Compute changing_lambda(descending order)
#     t = np.arange(0., 3, 0.01)
    t = np.arange(0., 10, h)
    for p in np.arange(0.99, -0.01, -0.01):
        i = int((1-p)/0.01-1)
        
#         Ai = A * p # not changing the original data
#         xi, info_dict = spi.odeint(diff, rx[i], t, args=(Ai, mu, delta),
#                               full_output=True,
#                               hmin=1e-14,
#                               mxhnil=0)
        pi = np.arange(p, p-0.01, -0.01/len(t))
        xi = np.zeros(( len(t), N ))
        xi[0] = rk4( rx[i], A*pi[0], h, diff )
        for j in range(len(t)-1):
            At = A * pi[j+1]
            xi[j+1] = rk4( xi[j], At, h, diff )
        i = i+1
        rx[i] = xi[-1]

        # Compute jacobian matrix
        xx = np.tile(rx[i],(N,1))
#         jacobian = Ai*(np.exp(mu-xx)/(1+np.exp(mu-xx))**2)
        jacobian = At*(np.exp(mu-xx)/(1+np.exp(mu-xx))**2)
        row, col = np.diag_indices_from(jacobian)
        jacobian[row,col] = -1

        # Compute eigenvalues
        evals = np.linalg.eigvals(jacobian)

        # Compute the real part of the dominant eigenvalue (smallest magnitude)
        re_evals = [lam.real for lam in evals]
        ind_max = np.array(re_evals).argmax()
        dom_eval_re[i] = max(re_evals)
        if i == (ni-1):
            break

#     xarray = np.arange(0, 1.01, 0.01)
    rxt = rx.T
    print("Number:", count)
    print("final average:", np.average(xi[-1]))
    
    state = np.average(rxt, axis=0)
    A = A.T
    
    m = np.argmax(dom_eval_re[20:95])
#     pt = (m+20)*0.01
    pt = (m+20+1)*0.01
    save_data_number = 0
    # detect judgement
    if state[m+20] - state[m+21] > 5:
        print(f"already found bifurcation in count: {count}, start saving data!")
        print(f"dev:{dom_eval_re[20:95].max()}, argdev:{m+20}, delta:{state[m+20] - state[m+21]}")
        for i in range(55):
#             p = pt+i*0.01
            np.savez(f'transient_conti/data{count}_{i}.npz', x = rx[m-i:m+20-i].T, y=np.array([pt]), w_loss=np.array([i+1]))
            if m-i == 0:
                break
        save_data_number = i + 1
        print(f"count: {count} saves {i+1} data")
    else:
#         save_data_number = 0
        print(f"no bifurcation found in count: {count}, dev:{dom_eval_re[20:95].max()}, argdev:{m+20}, delta:{state[m+20] - state[m+21]}")

    return save_data_number


def parallel(core=20):
#     global p
    pool = mp.Pool(core)

    results = []
    for count in np.arange(0, 1000, 1):
        results.append(pool.apply_async(simulation, args=(count,)))
    pool.close()
    pool.join()
    
    listnumber = []
    data_number = 0
    
    for i in results:
        r1 = i.get()
        data_number += r1
        listnumber.append(r1)
        
    return data_number, listnumber
#     return
    
    
data_number0, listnumber0 = parallel()
print(f"totally generate {data_number0} data!")
listnumber1 = np.array(listnumber0)
np.savez('transient_conti/data_number.npz', data_number = listnumber1)


# simulation(0)
















