import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import networkx as nx
import numpy as np
from tqdm import tqdm
import argparse
import random
from models import GNN,GPSModel,GraphTransformer,Graphormer
from data_loader import load_data_outage, load_graph_data
from logger import Logger

def Average(lst):
    return sum(lst) / len(lst)

def Topo_Fe_TimeSeries_MP(node_voltage, branchFlow, Filtration_voltage, Filtration_Flow,N,E):
    betti_0 = []
    for p in range(len(Filtration_voltage)):
        # Precompute active nodes based on threshold F_voltage[p]
        Active_node_v = np.where(node_voltage >= Filtration_voltage[p])[0]
        for q in range(len(Filtration_Flow)):
            if Active_node_v.size == 0:
                betti_0.append(0)
                continue

            # Create directed graph
            G = nx.Graph()
            G.add_nodes_from(Active_node_v)

            # Find edges where branch flow exceeds threshold F_Flow[q]
            indices = np.where(branchFlow >= Filtration_Flow[q])[0]
            edges_to_add = [(int(N.index(E[s][0])), int(N.index(E[s][1]))) for s in indices]

            # Filter edges to include only active nodes
            edges_to_add = [(a, b) for a, b in edges_to_add if a in Active_node_v and b in Active_node_v]
            G.add_edges_from(edges_to_add)
            betti_0.append(nx.number_connected_components(G))
    return betti_0


def main():
    parser = argparse.ArgumentParser(description="Betti Extraction (line failure)")
    parser.add_argument('--model', type=str, choices='MLP')
    parser.add_argument('--data_type', type=str, default='LineFailures')
    parser.add_argument('--bus', type=str, default='8500bus')

    args = parser.parse_args()
    A, node_fe,edge_fe, Class,N,E = load_data_outage(args.data_type,args.bus)
    F_Flow = [100, 30, 23, 15, 5, 2, -5]
    F_voltage = [1.0,.99,0.98, 0.97, 0.96, 0.68, 0.67, 0.66, 0.35, 0.34, 0.33,0]
    betti=[]
    S_betti=[]
    for i in tqdm(range(5000)):
        AverageVoltage = np.array([Average(list) for list in node_fe[i]])
        betti_0=Topo_Fe_TimeSeries_MP(AverageVoltage, edge_fe[i], F_voltage, F_Flow, N, E)
        betti.append(betti_0)
        S_betti.append(betti_0[6::7])

    torch.save(torch.tensor(betti), f'MP_features_{args.bus}_{args.data_type}.pt')
    torch.save(torch.tensor(S_betti), f'SP_features_{args.bus}_{args.data_type}.pt')
    # print(betti)
    # print(S_betti)
if __name__ == "__main__":
    main()
