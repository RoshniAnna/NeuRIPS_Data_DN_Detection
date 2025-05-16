"""
In this file the line failures (outages) are simulated
"""

import os
import pandas as pd
from GraphBuild import * 
import matplotlib.pyplot as plt
import random
import pickle

### Building circuit

# Initialize Circuit
FolderName = os.path.dirname(os.path.realpath("__file__"))
DSSfile = r""+ FolderName+ "\Master.dss"
Ckt_obj = CircuitSetup(DSSfile)  #creating a DSS object instance

#-- Equivalent Graph
G_original =  build_graph(Ckt_obj)
nx.readwrite.gml.write_gml(G_original,"8500NodeEx.gml") #Graph undirected with edge features and node features which are constant
node_list=list(G_original.nodes())
edge_list=list(G_original.edges())

### Read Sensors
with open('bus_sensors.pkl', 'rb') as f1:
     bus_sensors = pickle.load(f1)

with open('branch_sensors.pkl', 'rb') as f2:
     branch_sensors = pickle.load(f2)

### Loadshapes Restructure
input_file = 'LoadShape1.xlsx'
sheet_name = 'LoadShape1'  # Change to the sheet you want to split
df = pd.read_excel(input_file, sheet_name=sheet_name)

points_per_day = 24  # Assuming 1-hour resolution (24 points = 24 hours)
# Convert to flat list
loadshape_values = df.iloc[:, 0].values.tolist()
# Check total number of complete days available
num_days = len(loadshape_values) // points_per_day
# Slice the loadshape into daily
LoadShapes = []
for i in range(num_days):
    start_idx = i * points_per_day
    end_idx = (i + 1) * points_per_day
    daily_shape = loadshape_values[start_idx:end_idx]
    LoadShapes.append(daily_shape)


## Define function to generate outage scenarios    

MAX_RADIUS_FRAC = 1 / 3
MAX_OUTAGE_PERC = 0.4    
def generate_outage_edges(G, max_rad_frac=MAX_RADIUS_FRAC, max_percfail=MAX_OUTAGE_PERC):
    nd = random.choice(list(G.nodes()))
    # max_rad = nx.diameter(G)
    rad = math.ceil(random.uniform(1, 15))  # ensure radius >= 1
    Gsub = nx.ego_graph(G, nd, radius=rad, undirected=False)
    sub_edges = list(Gsub.edges())
    if not sub_edges:
        return generate_outage_edges(G)  # Retry if no edges in subgraph
    out_perc = random.uniform(0.1, max_percfail)
    N_out = max(1, math.ceil(len(sub_edges) * out_perc))
    out_edges = random.sample(sub_edges, k=N_out)
    return out_edges


Scenarios  = []
scid = 0
### Scenario generation
NSc = 20000 # parameter indicating no.of scenarios for each

# # Normal case
# for idx in range(NSc//2):
#     print(scid)
#     loadshape_day = random.choice(LoadShapes)
#     ldmult = random.choice(loadshape_day)
#     Ckt_obj.dss.Solution.LoadMult(ldmult)
#     Ckt_obj.dss.Solution.Solve()
#     V_node_Sc = {}
#     for bus in bus_sensors:
#             V=Bus(Ckt_obj,bus).Vmag
#             V_node_Sc[bus] = V  
            
#     flow_branch_Sc = {}  
#     for (u,v) in branch_sensors:
#         branch_label = G_original[u][v]['Label']
#         branch_device = G_original[u][v]['Device']
#         branch_elem = f"{branch_device}.{branch_label}"
#         branch_pflow = Branch(Ckt_obj,branch_elem).flow
#         flow_branch_Sc[(u,v)] = np.sum(branch_pflow)
#     Scenarios.append({'Index':scid, 'Outage':'No', 'Outage Lines': [], 'Outage phase': [], 'BusVoltages':V_node_Sc, 'BranchFlows':flow_branch_Sc})
#     scid = scid + 1


# Outage case
for idx in range(48):
    print(scid)
    Ckt_obj = CircuitSetup(DSSfile)
    loadshape_day = random.choice(LoadShapes)
    ldmult = random.choice(loadshape_day)
    Ckt_obj.dss.Solution.LoadMult(ldmult)
    Ckt_obj.dss.Solution.Solve()
    outage_edges = generate_outage_edges(G_original)
    outage_lines = []
    outage_types = []
    
    for from_node, to_node in outage_edges:
        edge_data = G_original.get_edge_data(from_node, to_node)
        outageelem = f"{edge_data['Device']}.{edge_data['Label']}"
        if edge_data['Device'] == 'line':
            outage_lines.append(outageelem)
            Ckt_obj.dss.Circuit.SetActiveElement(outageelem)
            active_phases = Ckt_obj.dss.CktElement.NodeOrder()[:len(Ckt_obj.dss.CktElement.NodeOrder())//2]
            phase_indic = random.randint(1,len(active_phases))
            if phase_indic == 3: #3 phase
                    Ckt_obj.dss.CktElement.Open(1,0)
                    outage_types.append('3 ph')                
            if phase_indic == 1: #1 phase
                    phx = random.choice(active_phases)
                    Ckt_obj.dss.CktElement.Open(1,phx)
                    outage_types.append('1 ph')                
            if phase_indic == 2: #2 phase
                    [phx,phy] = random.sample(active_phases, 2)
                    Ckt_obj.dss.CktElement.Open(1,phx)
                    Ckt_obj.dss.CktElement.Open(1,phy)
                    outage_types.append('2 ph')  
            Ckt_obj.dss.Solution.Solve()
            
    V_node_Sc = {}
    for bus in bus_sensors:
            V=Bus(Ckt_obj,bus).Vmag
            V_node_Sc[bus] = V  
            
    flow_branch_Sc = {}  
    for (u,v) in branch_sensors:
        branch_label = G_original[u][v]['Label']
        branch_device = G_original[u][v]['Device']
        branch_elem = f"{branch_device}.{branch_label}"
        branch_pflow = Branch(Ckt_obj,branch_elem).flow
        flow_branch_Sc[(u,v)] = np.sum(branch_pflow)
    
    Scenarios.append({'Index':scid, 'Outage':'Yes', 'Outage Lines': outage_lines, 'Outage phase': outage_types, 'BusVoltages':V_node_Sc, 'BranchFlows':flow_branch_Sc})
    scid = scid + 1
    
# Shuffle list    
random.shuffle(Scenarios)

# Write list
with open(FolderName+ '/LineFailures_8500_Attack5.pkl', 'wb') as file:
    pickle.dump(Scenarios, file)