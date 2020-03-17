import numpy as np
import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.sparse import csgraph

def matrix_perturbation(A,A_new):
    size = len(A)
    # ------define----

    delta_A=A_new-A

    D=csgraph.laplacian(A)+A

    D_new = csgraph.laplacian(A_new)+A_new

    delta_D=D_new-D

    L=csgraph.laplacian(A)

    L_new = csgraph.laplacian(A_new)

    delta_L=L_new-L

    # -----eigen_value----

    u, V = la.eig(A)

    V = np.matrix(V)

    # ----the change of u------

    delta_u = V.T.dot(delta_L).dot(V)-np.diag(u).dot(V.T).dot(delta_D).dot(V)

    u_new = u+np.diag(delta_u)

    # ----the change of v------

    weight_0_1 = (V.T[1].dot(delta_A).dot(V.T[0].T)-u[0]*(V.T[1]).dot(delta_D).dot(V.T[0].T))/(u[0]-u[1])

    weights_i_j =  ((V.T.dot(delta_A).dot(V)).T-np.diag(u).dot(V.T).dot(delta_D).dot(V))

    np.fill_diagonal(weights_i_j,0)

    # np.sum(np.multiply(np.tile(np.array(weights_i_j[...,np.newaxis]),[1,1,3]),(np.array(V)).T),axis=1).T

    weights_i_i = np.multiply(np.tile(np.diag(-0.5*V.T.dot(delta_D).dot(V)),[size,1]),V)

    # delta_v_0 = -0.5*V.T[0].dot(delta_D).dot(V.T[0].T)[0,0]*(V.T[0].T)+weight_0_1*V.T[1].T

    V_new = V + weights_i_i + weights_i_j

    return u_new,V_new

def get_graph_emb(raw_actions):
    graph_embs = []
    for i in range(len(raw_actions)):
        L=[]
        G = nx.Graph()
        G.add_nodes_from([0,1,2,3,4,5,6])
        G.add_weighted_edges_from([(i,j,0.01) for i in range(1,8) for j in range(1,8) if i<j])
        for j in range(len(raw_actions[i])):
            (start_player, end_player) = raw_actions[i][j].split(':')[0].split('->')
            G.add_weighted_edges_from([(int(start_player), int(end_player), 1)])
            A=nx.to_numpy_matrix(G)
            u, V = la.eig(csgraph.laplacian(A))
            L.append(V[1])
        while len(L)<maxlen and len(L)<10000:
            L.append([0,]*len(L[0]))
        graph_embs.append(L)
    return graph_embs
