import os
import math
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import roc_auc_score
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple

GPR_CHOLESKY_LOWER = True
SIZE=392
THRESHOLD=SIZE*5


def detect_cycle(adj: np.ndarray) -> Tuple[bool, Optional[List[int]]]:
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square 2D numpy array")

    n = adj.shape[0]
    # 0 = unvisited, 1 = visiting (in recursion stack), 2 = done
    state = np.zeros(n, dtype=np.int8)
    parent = np.full(n, -1, dtype=int)

    def build_cycle(u: int, v: int) -> List[int]:
        path = [v]
        cur = u
        while cur != v and cur != -1:
            path.append(cur)
            cur = parent[cur]
        path.append(v)
        path.reverse()  # make it v ... u v in forward order
        return path

    def dfs(u: int) -> Optional[List[int]]:
        state[u] = 1  # visiting
        # iterate neighbors v with edge u->v
        # Works for 0/1, bool, or weighted adjacency (nonzero means edge exists).
        for v in range(n):
            if adj[u, v] != 0:
                if state[v] == 0:
                    parent[v] = u
                    cyc = dfs(v)
                    if cyc is not None:
                        return cyc
                elif state[v] == 1:
                    # back edge found => cycle
                    return build_cycle(u, v)

        state[u] = 2  # done
        return None

    for i in range(n):
        if state[i] == 0:
            cyc = dfs(i)
            if cyc is not None:
                return True, cyc

    return False, None

def get_mdl_train(x,x_parents):
    #print("x:\n"+str(x))
    #print("x_parents:\n"+str(x_parents))
    if x_parents.empty:
        x_parents=pd.DataFrame(np.ones(np.shape(x)))
    
        x_parents = np.asarray(x_parents).reshape(-1,1)
        kernel=ConstantKernel(1.0, (1e-3, 1e3))+ WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
        gpr=GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x_parents,x)

        # Model Complexity parameter alpha and MDL model score
        mdl_pen_train = 0

        # MDL data score/log likelihood
        mdl_lik_train = -gpr.log_marginal_likelihood_value_

        # MDL normalization term
        mdl_norm_train = 0 # 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**-2 * K))

    else:
        x_parents = np.asarray(x_parents)
        if x_parents.ndim==1:
            #print(f'one dim x_parents: {x_parents}')
            x_parents=x_parents.reshape(-1,1)
        kernel=RBF(1.0)+WhiteKernel(noise_level=1.0)
        gpr=GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x_parents,x)
        K=gpr.kernel_(x_parents)
        sigma=1

        # Model Complexity parameter alpha and MDL model score
        mat = np.eye(x_parents.shape[0]) * sigma**2 + K
        alpha = np.linalg.solve(mat, x)
        gpr.alpha_=alpha
        mdl_pen_train=alpha.T @ K @ alpha # alpha.T @ mat @ alpha is used in original code

        # MDL data score/log likelihood
        mdl_lik_train = -gpr.log_marginal_likelihood_value_
        # precompute for self.mdl_score():
        L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        gpr.L_ = L

        # MDL normalization term
        mdl_norm_train = 0 # set to 0 because of infinity and fair
        #mdl_norm_train =  1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**-2 * K))
        #if np.isinf(mdl_norm_train):
        #    mdl_norm_train = 0
    return mdl_lik_train + mdl_pen_train + mdl_norm_train


def get_pair_scores_skeleton(x_data,x_parents,y_data,y_parents,mute=True):
    # input: data of little system related to node X and Y (X<Y), cf==False if no confounder Z
    # output: scores of X->Y, Y->X, X<-Z->Y, no edge, confidence
    score_XY, score_YX, score_cf, score_no, confidence = 0,0,0,0,0


    # gpr_x: node x and its parents
    mdl_x=get_mdl_train(x_data,x_parents)
    # gpr_xy: node x and its parents and new parent y
    xp_yd=pd.concat([x_parents, y_data],axis=1)
    xp_yd.columns=xp_yd.columns.astype(str)
    mdl_xy=get_mdl_train(x_data,xp_yd)
    # gpr_y: node y and its parents
    mdl_y=get_mdl_train(y_data,y_parents)
    # gpr_yx: node y and its parents and new parent x
    yp_xd=pd.concat([y_parents, x_data],axis=1)
    yp_xd.columns=yp_xd.columns.astype(str)
    mdl_yx=get_mdl_train(y_data,yp_xd)

    score_xy = mdl_x+mdl_xy # score of x causes y
    score_yx = mdl_y+mdl_yx # score of y causes x
    score_z = mdl_xy+mdl_yx
    score_no = mdl_x+mdl_y
    if mute == False:
        print("\n\nscore_xy:"+str(score_xy)+"\nscore_yx:"+str(score_yx)+"\nscore_z:"+str(score_z)+"\nscore_no:"+str(score_no))

    confidence=abs(score_xy-score_yx)

    return score_xy, score_yx, score_z, score_no, confidence


def get_graph_mdl(graph, data):
    # input: data and causal graph
    # output: mdl score for the whole graph
    score=0
    dim=graph.shape[0]

    if not isinstance(graph, np.ndarray):
        graph = graph.to_numpy()

    for i in range(dim):
        i_parents = np.where((graph[:,i]==1)|(graph[:,i]==3))[0]
        score += get_mdl_train(data.iloc[:,i], data.iloc[:,i_parents.tolist()])

    return score


def skeleton(graph,data):
    length, dim = data.shape
    mdl_score=math.inf

    score=np.zeros((2,dim,dim)) #(0,X,Y) for X->Y, (0,Y,X) for Y->X, (1,min(X,Y), max(X,Y)) for Y<-Z->X, (1,max(X,Y),min(X,Y)) for no edge
    confidence=np.zeros((dim,dim))      # MAX(X->Y, Y->X) - MIN(X->Y, Y->X)

    initial=True
    counter=0
    MAX_ITER=20

    # get the skeleton    
    while(initial and counter<MAX_ITER):
        counter=counter+1
        for i in range(dim):
            for j in range(i+1,dim):
                # compute the score in different cases in each pair of the nodes (i<j)
                #print("\ni:"+str(i)+"\tj:"+str(j))
                graph_tmp=graph.copy()
                graph_tmp[i,j]=0
                graph_tmp[j,i]=0
                i_parents = np.where(graph_tmp[:,i]>0)[0]
                j_parents = np.where(graph_tmp[:,j]>0)[0]
                score[0,i,j], score[0,j,i], score[1,i,j], score[1,j,i], confidence[i,j] = get_pair_scores_skeleton(data.iloc[:,i],data.iloc[:,i_parents.tolist()],data.iloc[:,j],data.iloc[:,j_parents.tolist()],mute=True)

        graph_record=graph.copy()
        # start to modify graph according to scores
        for i in range(dim):
            for j in range(i+1,dim):
                delta=score[0,i,j]-score[0,j,i]
                if score[1,j,i]<min(score[0,i,j],score[0,j,i]):
                    graph[i,j]=0
                    graph[j,i]=0
                elif delta>THRESHOLD:
                    graph[i,j]=0
                    graph[j,i]=1
                elif delta<(0-THRESHOLD):
                    graph[i,j]=1
                    graph[j,i]=0
                else:
                    graph[i,j]=3
                    graph[j,i]=3
                    confidence[i,j]=0
        if (graph_record==graph).all():
            initial=False

    return graph,score,confidence


def break_cycle(graph_sk,data):
    graph=graph_sk.copy()
    graph_noz = np.where(graph==3,0,graph)
    cycle, cyc_list = detect_cycle(graph_noz)
    while cycle:
        cyc_size = len(cyc_list)-1
        loss = np.zeros((cyc_size,2))
        for i in range(cyc_size):
            graph_tmp = graph.copy()
            graph_tmp[cyc_list[i],cyc_list[i+1]]=0
            graph_tmp[cyc_list[i+1],cyc_list[i]]=0
            loss[i,0]=get_graph_mdl(graph_tmp, data) - get_graph_mdl(graph, data) # loss of deleting the edge

            graph_tmp[cyc_list[i],cyc_list[i+1]]=3
            graph_tmp[cyc_list[i+1],cyc_list[i]]=3
            loss[i,1]=get_graph_mdl(graph_tmp, data) - get_graph_mdl(graph, data) # loss of replacing the edge to latent factor
        flat_idx = np.argmin(loss)
        idx_r,idx_c=np.unravel_index(flat_idx, loss.shape)
        if idx_c==0:
            graph[cyc_list[i],cyc_list[i+1]]=0
            graph[cyc_list[i+1],cyc_list[i]]=0
        elif idx_c==1:
            graph[cyc_list[i],cyc_list[i+1]]=3
            graph[cyc_list[i+1],cyc_list[i]]=0

        graph_noz = np.where(graph==3,0,graph)
        cycle, cyc_list = detect_cycle(graph_noz)
    return graph


if __name__ == '__main__':
    dir_name='data/real/auto_mpg/'
    result_dir = dir_name+'results/'
    try:
        os.makedirs(result_dir)
        print(f"Directory '{result_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory'{result_dir}' already exist.")

    data = pd.read_csv(dir_name+'auto_mpg.csv')
    length, dim = data.shape
    graph_sk,score,confidence = skeleton(np.zeros((dim,dim)),data)
    esti_graph = break_cycle(graph_sk,data)

    np.savetxt(result_dir+'confidence.csv',confidence.astype(int),delimiter=",",fmt="%d")
    np.savetxt(result_dir+'DAG.csv',esti_graph.astype(int),delimiter=",",fmt="%d")
