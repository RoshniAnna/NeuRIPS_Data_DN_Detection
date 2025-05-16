
"""
In this file the data for state estimation is generated.
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
DSSfile = r""+ FolderName+ "\IEEE123Master.dss"
Ckt_obj = CircuitSetup(DSSfile)  #creating a DSS object instance

#-- Equivalent Graph
G_original =  build_graph(Ckt_obj)
nx.readwrite.gml.write_gml(G_original,"123busEx.gml") #Graph undirected with edge features and node features which are constant
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

Scenarios  = []
scid = 0
### Scenario generation
NSc = 15000 # parameter indicating no.of scenarios for each

for idx in range(NSc):
    print(scid)
    loadshape_day = random.choice(LoadShapes)
    ldmult = random.choice(loadshape_day)
    Ckt_obj.dss.Solution.LoadMult(ldmult)
    Ckt_obj.dss.Solution.Solve()
    V_node_Sc = {}
    V_unknown = {}
    for bus in bus_sensors:
        V=Bus(Ckt_obj,bus).Vmag
        V_node_Sc[bus] = V  
            
    for bus in node_list:
        if bus not in bus_sensors:
            V=Bus(Ckt_obj,bus).Vmag
            V_unknown[bus] = V  
            
    flow_branch_Sc = {}  
    for (u,v) in branch_sensors:
        branch_label = G_original[u][v]['Label']
        branch_device = G_original[u][v]['Device']
        branch_elem = f"{branch_device}.{branch_label}"
        branch_pflow = Branch(Ckt_obj,branch_elem).flow
        flow_branch_Sc[(u,v)] = np.sum(branch_pflow)
        
    Scenarios.append({'Index':scid, 'Sensor BusVoltages':V_node_Sc, 'Sensor BranchFlows':flow_branch_Sc, 'Unknown BusVoltages':V_unknown})
    scid = scid + 1
   
# Shuffle list    
random.shuffle(Scenarios)

# Write list
with open(FolderName+ '/StateEstimate_123.pkl', 'wb') as file:
    pickle.dump(Scenarios, file)