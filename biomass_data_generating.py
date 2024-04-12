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


def scale_free( n, md, seed ):

    np.random.seed(seed)

    m = n * md
    gamma_in = 2.5
    gamma_out = 2.5

    alpha_in = 1/(gamma_in-1)
    alpha_out = 1/(gamma_out-1)

    w_in = np.ones(n)
    w_out = np.ones(n)
    edges = list()

    for i in range(n):
        w_in[i] = 1 / (1 + i)**alpha_in
        w_out[i] = 1 / (1 + i)**alpha_out


    w_in = np.random.permutation(w_in)
    w_out = np.random.permutation(w_out)
    w_in = w_in / np.sum(w_in)
    w_out = w_out / np.sum(w_out)

    l = 0
    while l < m:

        s = np.random.choice(range(n), p=w_out)

        t = np.random.choice(range(n), p=w_in)

        if s != t:
            edge = (s, t)
            if edge not in edges:
                edges.append(edge)
                l += 1

    # print(edges)
    g = nx.DiGraph()
#     g.add_nodes_from( range(n) )
    g.add_edges_from( edges )
    mapping = dict(zip(g, range(0, len(g))))  # delete isolated nodes
    g = nx.relabel_nodes(g, mapping)   # renumber the nodes

    
    return g


# define dynamics
def diff_func(xr, Ar, r, D, cr):
    dd = r*xr*(1-xr/10) - cr*xr**2/(xr**2+1) - D*(np.sum(Ar*(xr.reshape(xr.shape[0],1)-xr),axis=1)) + 0.5*np.random.normal(size=len(xr))

    return dd


def rk4( theta, Ar, r, D, cr, h, diff_func ):  # [node, hidden_channel]
#     h = 0.01
    k1 = diff_func(theta, Ar, r, D, cr)
    k2 = diff_func(theta + h * k1 / 2, Ar, r, D, cr)
    k3 = diff_func(theta + h * k2 / 2, Ar, r, D, cr)
    k4 = diff_func(theta + h * k3, Ar, r, D, cr)
    theta = theta + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return theta



def simulation(count):
    
    time_start = time.time()    
    _, decimals = str(time_start).split('.')  # Take the fractional portion of the current time as seed
    seed = int(decimals)
    np.random.seed(seed)
    
    md = 4 + 0.01*np.random.randint(-200,200,size=1)
    n = 200 + np.random.randint(-195,200,size=1) 
    md, n = md[0], int(n)

    g = scale_free( n, md, seed )
    g.remove_nodes_from(list(nx.isolates(g)))
    mapping = dict(zip(g, range(0, len(g))))
    g = nx.relabel_nodes(g, mapping)
    N = len(g)

    w = np.ones( len(g.edges()) )
    source, target = zip(*g.edges())
    weight_edges = zip( source, target, w )
    g.add_weighted_edges_from( weight_edges )
    
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    print(f'graph:{count}, nodes:{nodes}, edges:{edges}, md:{edges/nodes}')

    #generate weight adjacency matrix
    A = nx.to_numpy_array( g ) #element: out_link
    for s,t, w in g.edges( data=True ):
        A[s,t] = w['weight']
    A = A.T #elemnt: in_link

    x = np.random.rand( N ) * 15  #init
    r, D = 0.3*np.random.random_sample()+0.7, 0.5*np.random.random_sample()+0.5
    print(f'graph:{count}, r:{r}, D:{D}')
     
    ## Simulate the system
    ni = 101
    rx = np.zeros((ni,N))
    h = 0.01
    t = np.arange(0., 15, h)
    c = 1 # grazing rate
    dom_eval_re = np.zeros(ni)

    xi = np.zeros(( len(t), N ))
    xi[0] = rk4( x, A, r, D, c, h, diff_func )
    for j in range(len(t)-1):
        xi[j+1] = rk4( xi[j], A, r, D, c, h, diff_func )
    rx[0] = xi[-1]

    # Compute jacobian matrix
    xx = np.tile(rx[0],(N,1))
    jacobian = 0.5*A*xx
    row, col = np.diag_indices_from(jacobian)
    jacobian[row,col] = 1 - xi[-1]/5 - c*2*xi[-1]/(xi[-1]**2+1)**2 - 0.5*xi[-1]*np.sum(A, axis=1)

    # Compute eigenvalues
    evals = np.linalg.eigvals(jacobian)

    # Compute the real part of the dominant eigenvalue (smallest magnitude)
    re_evals = [lam.real for lam in evals]
    ind_max = np.array(re_evals).argmax()
    dom_eval_re[0] = max(re_evals)
    
    print("Number:", count)
    print(f"N:{N}, edges:{len(g.edges())}, md:{edges/nodes}")
    print("initial average:", np.average(xi[-1]))


    ## Compute changing_lambda
    t = np.arange(0., 10, h)
    for c in np.arange(1.02, 3.01, 0.02):
        num = int((c-1)/0.02 - 1)
        xi = np.zeros(( len(t), N ))
        xi[0] = rk4( rx[num], A, r, D, c, h, diff_func )
        for j in range(len(t)-1):
            xi[j+1] = rk4( xi[j], A, r, D, c, h, diff_func )

        num += 1
        rx[num] = xi[-1]

        # Compute jacobian matrix
        xx = np.tile(rx[num],(N,1))
        jacobian = 0.5*A*xx
        row, col = np.diag_indices_from(jacobian)
        jacobian[row,col] = 1 - xi[-1]/5 - c*2*xi[-1]/(xi[-1]**2+1)**2 - 0.5*xi[-1]*np.sum(A, axis=1)

        # Compute eigenvalues
        evals = np.linalg.eigvals(jacobian)

        # Compute the real part of the dominant eigenvalue (smallest magnitude)
        re_evals = [lam.real for lam in evals]
        ind_max = np.array(re_evals).argmax()
        dom_eval_re[num] = max(re_evals)

    xarray = np.arange(1, 3.01, 0.02) 
    rxt = rx.T
    time_ode = time.time()
    print("Number:", count)
    print("final average:", np.average(xi[-1]), f'use {time_ode-time_start} seconds!')
    
    state = np.average(rxt, axis=0)
    A = A.T
    
#     Save data
    save_data_number = 0
    tipping_ID = np.argmax(dom_eval_re[20:-(len(state[state<1])-3)]) + 20
    if state[tipping_ID+1]<1 and state[tipping_ID-1]>3:
        print(f"already found bifurcation in count:{count}, start saving data!!!")
        print(f"dev:{dom_eval_re[tipping_ID]}, argdev:{tipping_ID}, Tpoint:{xarray[tipping_ID]}, delta:{state[tipping_ID-1] - state[tipping_ID+1]}")
        
        for i in range(60):
            np.savez( f'data/data{count}_{i}.npz', x = rx[tipping_ID-20-i:tipping_ID-i].T, A = A, y=np.array([xarray[tipping_ID]]), w_loss=np.array([i+1]) )                        
            if tipping_ID-20-i == 0:
                break
        save_data_number = i + 1
        print(f"count:{count} saves {i+1} data")
    else:
        print(f"no bifurcation found in count:{count}, dev:{dom_eval_re[tipping_ID]}, argdev:{tipping_ID}, delta:{state[tipping_ID-1] - state[tipping_ID+1]}")

    return save_data_number
        


def parallel(core=60):
#     global p
    pool = mp.Pool(core)

    results = []
    for count in np.arange(0, 600, 1):
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
np.savez('data_number.npz', data_number = listnumber1)


# simulation(0)
















