import os
import pandas as pd
import numpy as np
import cdt.metrics as cdt_metrics
import networkx as nx


if __name__  == '__main__':
    size = 200
    dir_name = 'data/art/multi/multi_size'+str(size)+'n6z1poly3/'
    dir_result = dir_name+'results/'
    sets=20
    
    z_hit=np.zeros(sets)
    e_hit=np.zeros(sets)
    shd=np.zeros(sets)
    sid=np.zeros(sets)
    accuracy=np.zeros(sets)
    accuracy_all=np.zeros(sets)

    mdl_for=np.zeros(sets)
    mdl_bac=np.zeros(sets)
    mdl_null=np.zeros(sets)

    for ii in range(sets):
         print(f'====DATASET {ii}====')
         graph_true = pd.read_csv(dir_name+str(ii)+'_DAG_noZ.csv',header=None).to_numpy()
         graph_esti = pd.read_csv(dir_result+str(ii)+'_DAG.csv',header=None).to_numpy()


         z_pos = np.argwhere(graph_true==3)[0]
         if graph_esti[z_pos[0],z_pos[1]]==3:
             z_hit[ii]=1
         e_poses = np.argwhere(graph_true==1)
         count = np.sum(graph_true==1)
         counter=0
         for e_pos in e_poses:
             if graph_esti[e_pos[0],e_pos[1]]==1:
                 counter = counter+1
         e_hit[ii]=counter/count
         if np.isnan(e_hit[ii]):
             e_hit[ii]=1
             

         true_clean = (graph_true==1).astype(int)
         esti_clean = (graph_esti==1).astype(int)
         accuracy[ii] = np.mean(true_clean==esti_clean)

         true_full = (graph_true>0).astype(int)
         esti_full = (graph_esti>0).astype(int)
         accuracy_all[ii] = np.mean(true_full==esti_full)


    print(f'Z_hit: {z_hit}')
    print(f'average: {np.mean(z_hit)}')
    print(f'E_hit: {e_hit}')
    print(f'average: {np.mean(e_hit)}')
    print(f'Sim1: {accuracy}')
    print(f'average:{np.mean(accuracy)}')
    print(f'Sim2: {accuracy_all}')
    print(f'average:{np.mean(accuracy_all)}')
