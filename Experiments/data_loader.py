import networkx as nx
import pickle
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
# Define optimized timeseries extractor
def make_timeseries(P0Data, Nodes, Edges, total_time_step, scenario):
    data = P0Data[scenario]
    bus_voltage_series = data['BusVoltage series']
    branch_flow_series = data['BranchFlow series']

    time_series_voltage = [
        [bus_voltage_series[node][t] for node in Nodes]
        for t in range(total_time_step)
    ]
    time_series_BFlow = [
        [branch_flow_series[edge][t] for edge in Edges]
        for t in range(total_time_step)
    ]

    return time_series_voltage, time_series_BFlow


def load_dataset_detection(data_type, bus_name):
    # Construct paths
    base_path = os.path.join('datasets', data_type, bus_name)
    graph_path = os.path.join(base_path, f"{bus_name}Ex.gml")
    pickle_path = os.path.join(base_path, f"{data_type}_{bus_name}.pkl")

    # Load graph
    G = nx.read_gml(graph_path)
    N = list(G.nodes)
    E = list(G.edges)
    A = nx.to_numpy_array(G)

    # Load data
    with open(pickle_path, 'rb') as f:
        POData = pickle.load(f)

    # Step 1: Load all scenarios from the streamed file i.e 8500 node EVCS datsets
    #POData = []
    # with open( pickle_path, 'rb') as f:
    #     while True:
    #         try:
    #             scenario = pickle.load(f)
    #             #POData += scenario
    #             POData.append(scenario)
    #         except EOFError:
    #             break

    N_scenarios = len(POData)
    my_dict=POData[0]['BusVoltage series']
    time_steps=len(my_dict[next(iter(my_dict))])
    node_voltage = []
    branch_flow = []
    #for i in tqdm(range(N_scenarios), desc="Processing scenarios"):
    for i in range(N_scenarios):
        voltage, flow = make_timeseries(POData, N, E, time_steps, i)
        node_voltage.append(voltage)
        branch_flow.append(flow)
    if data_type=='PVAttacks':
        Class_labels = [0 if not POData[i]["Targeted PVs"] else 1 for i in range(N_scenarios)]
    elif data_type=='EVCSAttacks':
        Class_labels = [0 if not POData[i]["Targeted Stations"] else 1 for i in range(N_scenarios)]
    elif data_type=='SensorAttacks':
        Class_labels = [0 if not POData[i]["Targeted Buses"] else 1 for i in range(N_scenarios)]
    else:
        print("Data type is not valid")


    return G, node_voltage, branch_flow, Class_labels


def generate_multilabels(targeted_stations_list):
    """
    Converts a list of lists of targeted stations into multi-label binary vectors.

    Args:
        targeted_stations_list (List[List[str]]): A list where each element is a list of station names (e.g., ["PV1", "PV2"]).

    Returns:
        class_labels (List[List[int]]): A list of binary vectors indicating presence of each station.
        all_stations (List[str]): Sorted list of all unique stations (used as label indices).
    """
    # Step 1: Get all unique stations
    all_stations = sorted(set(station for stations in targeted_stations_list for station in stations))

    # Step 2: Create a mapping from station name to index
    station_to_index = {station: idx for idx, station in enumerate(all_stations)}

    # Step 3: Generate binary label vectors
    class_labels = []
    for stations in targeted_stations_list:
        label_vector = [0] * len(all_stations)
        for station in stations:
            label_vector[station_to_index[station]] = 1
        class_labels.append(label_vector)

    return class_labels, all_stations
def load_dataset_localization(data_type, bus_name):
    # Construct paths
    base_path = os.path.join('datasets', data_type, bus_name)
    graph_path = os.path.join(base_path, f"{bus_name}Ex.gml")
    pickle_path = os.path.join(base_path, f"{data_type}_{bus_name}.pkl")

    # Load graph
    G = nx.read_gml(graph_path)
    N = list(G.nodes)
    E = list(G.edges)
    A = nx.to_numpy_array(G)

    # Load data
    with open(pickle_path, 'rb') as f:
        POData = pickle.load(f)



    # Step 1: Load all scenarios from the streamed file i.e 8500 node EVCS datsets
    #POData = []
    # with open( pickle_path, 'rb') as f:
    #     while True:
    #         try:
    #             scenario = pickle.load(f)
    #             #POData += scenario
    #             POData.append(scenario)
    #         except EOFError:
    #             break

    N_scenarios = len(POData)
    my_dict=POData[0]['BusVoltage series']
    time_steps=len(my_dict[next(iter(my_dict))])
    node_voltage = []
    branch_flow = []
    #for i in tqdm(range(N_scenarios), desc="Processing scenarios"):
    for i in range(N_scenarios):
        voltage, flow = make_timeseries(POData, N, E, time_steps, i)
        node_voltage.append(voltage)
        branch_flow.append(flow)
    if data_type=='PVAttacks':
        targeted_stations_list = [POData[i]['Targeted PVs'] for i in range(N_scenarios)]
        # Call the function:
        Class_labels, all_PVs = generate_multilabels(targeted_stations_list)
    elif data_type=='EVCSAttacks':
        targeted_stations_list = [POData[i]['Targeted Stations'] for i in range(N_scenarios)]
        # Call the function:
        Class_labels, all_PVs = generate_multilabels(targeted_stations_list)
    elif data_type=='SensorAttacks':
        targeted_stations_list = [POData[i]['Targeted Buses'] for i in range(N_scenarios)]
        # Call the function:
        Class_labels, all_PVs = generate_multilabels(targeted_stations_list)
    else:
        print("Data type is not valid")


    return G, node_voltage, branch_flow, Class_labels


class TimeSeriesGraphDataset(Dataset):
    def __init__(self, adj_matrix, node_features, edge_features, labels):
        """
        Args:
            adj_matrix: [N, N] numpy or scipy sparse matrix
            node_features: list of [T, N] arrays — time series of node voltages
            edge_features: list of [T, E, 1] arrays — time series of edge branch flows
            labels: list of integer labels
        """
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = sp.coo_matrix(adj_matrix)

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj_matrix)

        self.x_list = [torch.tensor(x, dtype=torch.float) for x in node_features]  # [T, N, 1]
        self.edge_attr_list = [torch.tensor(e, dtype=torch.float) for e in edge_features]        # [T, E, 1]
        self.y_list = [torch.tensor(y, dtype=torch.long) for y in labels]

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return {
            'x': self.x_list[idx],              # Node features: [T, N, 1]
            'edge_attr': self.edge_attr_list[idx],  # Edge features: [T, E, 1]
            'edge_index': self.edge_index,      # [2, num_edges] (shared across all samples)
            'edge_weight': self.edge_weight,    # [num_edges] (optional)
            'y': self.y_list[idx]               # Label
        }



def load_data_outage(data_type,bus_name):
    base_path = os.path.join('datasets', data_type, bus_name)
    graph_path = os.path.join(base_path, f"{bus_name}Ex.gml")
    pickle_path = os.path.join(base_path, f"{data_type}_{bus_name}.pkl")
    #G=nx.read_gml("8500busEx.gml")
    G = nx.read_gml(graph_path)
    A = nx.to_numpy_array(G)
    N = list(G.nodes)
    E= list(G.edges)
    with open(pickle_path, 'rb') as f:
        POData = pickle.load(f)
    N_scenarios=len(POData)
    node_fe=[]
    for i in range(N_scenarios):
        values = np.array([
    POData[i]['BusVoltages'].get(node, np.array([-1, -1, -1])) for node in N
])
        scaler = MinMaxScaler()

        # Fit and transform
        x_scaled = scaler.fit_transform(values)
        node_fe.append(x_scaled)
    edge_fe = []
    for i in range(N_scenarios):
        E_values = np.array([
            POData[i]['BranchFlows'].get(edge, -1) for edge in E])
        edge_fe.append(E_values)
    Class=[]
    for i in range(N_scenarios):
        Class.append(POData[i]["Outage"])
    i=0
    while i < len(Class):
        if Class[i] == 'No':
            Class[i] = 0
        if Class[i] == 'Yes':
            Class[i] = 1
        i += 1
    return A, node_fe,edge_fe, Class,N,E


def load_graph_data(adj_matrix, node_features_list, graph_labels):
    """
    Args:
        adj_matrix: numpy array or scipy sparse matrix of shape [N, N] (shared across all graphs)
        node_features_list: list of NumPy arrays or torch tensors of shape [N, F]
        graph_labels: list or array of graph-level labels
    Returns:
        List of torch_geometric.data.Data objects
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = sp.coo_matrix(adj_matrix)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)

    data_list = []
    for i in range(len(node_features_list)):
        x = torch.tensor(node_features_list[i], dtype=torch.float)
        y = torch.tensor([graph_labels[i]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list

def get_voltage_array(entry, N):
    """
    Returns a (len(N), 3) numpy array of 3D voltage features for the given nodes in N.
    Values are selected from either 'Sensor BusVoltages' or 'Unknown BusVoltages'.
    Assumes each node is in exactly one of them.
    """
    sensor = entry['Sensor BusVoltages']
    unknown = entry['Unknown BusVoltages']

    voltages = []
    for node in N:
        if node in sensor:
            voltages.append(sensor[node])
        elif node in unknown:
            voltages.append(unknown[node])
        else:
            raise KeyError(f"Node {node} not found in either Sensor or Unknown BusVoltages")

    return np.array(voltages)

def load_data_StateEstimate(data_type, bus_name):
    base_path = os.path.join('datasets', data_type, bus_name)
    graph_path = os.path.join(base_path, f"{bus_name}Ex.gml")
    pickle_path = os.path.join(base_path, f"{data_type}_{bus_name}.pkl")
    # G=nx.read_gml("8500busEx.gml")
    G = nx.read_gml(graph_path)
    A = nx.to_numpy_array(G)
    N = list(G.nodes)
    E = list(G.edges)
    with open(pickle_path, 'rb') as f:
        POData = pickle.load(f)
    N_scenarios = len(POData)
    knownbus = list(POData[0]['Sensor BusVoltages'].keys())
    known_mask = torch.tensor([node in knownbus for node in N])

    node_fe = []
    for i in range(N_scenarios):
        values = get_voltage_array(POData[i], N)
        node_fe.append(values)
    edge_fe = []
    for i in range(N_scenarios):
        E_values = np.array([
            POData[i]['Sensor BranchFlows'].get(edge, -1) for edge in E])
        edge_fe.append(E_values)
    return A, node_fe, edge_fe, known_mask

def make_graph_data(adj_matrix, node_features_list):
    """
    Args:
        adj_matrix: numpy array or scipy sparse matrix of shape [N, N] (shared across all graphs)
        node_features_list: list of NumPy arrays or torch tensors of shape [N, F]
        graph_labels: list or array of graph-level labels
    Returns:
        List of torch_geometric.data.Data objects
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = sp.coo_matrix(adj_matrix)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)

    data_list = []
    for i in range(len(node_features_list)):
        x = torch.tensor(node_features_list[i], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    return data_list