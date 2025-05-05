
"""
In this file the different types of attacks on grid connected PV Systems is simulated
"""

import os
import pandas as pd
from GraphBuild import * 
import matplotlib.pyplot as plt
import random
import pickle

### Loading PV information
PVs = [{'no':1, 'bus':'890', 'numphase':3, 'phaseconn':'.1.2.3', 'size':146, 'kV':4.16},
       {'no':2, 'bus':'806', 'numphase':3, 'phaseconn':'.1.2.3', 'size':144, 'kV':24.9},
       {'no':3, 'bus':'816', 'numphase':3, 'phaseconn':'.1.2.3', 'size':200, 'kV':24.9}]

### Building circuit with PV

#--Initialize Circuit
FolderName = os.path.dirname(os.path.realpath("__file__"))
DSSfile = r""+ FolderName+ "\ieee34Mod1.dss"
Ckt_obj = CircuitSetup(DSSfile, PVs)  #creating a DSS object instance

#-- Equivalent Graph
G_original =  build_graph(Ckt_obj)
nx.readwrite.gml.write_gml(G_original,"34busEx.gml") #Graph undirected with edge features and node features which are constant
node_list=list(G_original.nodes())
edge_list=list(G_original.edges())

### Loadshapes Restructure

#-- Converting demand to daily variations
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
    
#-- Converting irradiance to daily variations
irradiance_df = pd.read_csv('5mins Irradiance Data (NSRDB).csv', skiprows=2) # Load the file, skip first metadata row
irradiance_df.columns
ghi_values = irradiance_df['GHI'].values
# Settings
points_per_day = 288  # 5-minute resolution
num_days = len(ghi_values) // points_per_day
# Slice into daily irradiance profiles
IrradianceShapes = []

for i in range(num_days):
    start_idx = i * points_per_day
    end_idx = (i + 1) * points_per_day
    daily_ghi = ghi_values[start_idx:end_idx]
    IrradianceShapes.append(daily_ghi)
    
# Initialize profiles    
Ckt_obj.dss.Text.Command(f"New LoadShape.IrradAttack")
Ckt_obj.dss.Text.Command(f"New LoadShape.LoadVar")
Ckt_obj.dss.Text.Command(f"New LoadShape.IrradVar")

#------------------------------------------------------------------------------------
### Define function to simulate attacks and extract time series information
def inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_idx, end_idx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day):
    
    # Edit the LoadShape in opendss
    Ckt_obj.dss.Text.Command(f"Edit LoadShape.LoadVar npts={len(loadshape_day)} interval=1 mult=(" + ' '.join(map(str, loadshape_day)) + ")")
    Ckt_obj.dss.Text.Command(
        f"Edit LoadShape.IrradVar npts={len(irradiance_day)} minterval=5 mult=({' '.join(map(str, irradiance_day))})"
    )

    # Step 2: Normalize it
    Ckt_obj.dss.Text.Command("Edit LoadShape.IrradVar action=normalize")
    
    # Change irradiance during the attack window
    if  attack_type == 0 or attack_type==2:
        attack_irradiance = irradiance_day.copy()
        for i in range(start_idx, end_idx):
            attack_irradiance[i] *= attack_mult
        # Edit the LoadShape in opendss
        Ckt_obj.dss.Text.Command(
            f"Edit LoadShape.IrradAttack npts={len(attack_irradiance)} minterval=5 mult=({' '.join(map(str, attack_irradiance))})"
        )
        # Step 2: Normalize it
        Ckt_obj.dss.Text.Command("Edit LoadShape.IrradAttack action=normalize")
        
    # Assign Load shapes
    Ckt_obj.dss.Text.Command("BatchEdit Load..* daily=LoadVar")
    if attack_type == 0 or attack_type == 2:
            for gen in PVs:
                if gen in attacked_PVs: 
                    Ckt_obj.dss.Text.Command(f"Edit PVSystem.PV{str(gen['no'])} daily=IrradAttack irradiance=1 tdaily=")
                else:
                    Ckt_obj.dss.Text.Command(f"Edit PVSystem.PV{str(gen['no'])} daily=IrradVar irradiance=1 tdaily=")
            Ckt_obj.dss.Solution.Solve()
            
    else:
            Ckt_obj.dss.Text.Command("BatchEdit PVSystem..* daily=IrradVar irradiance=1 tdaily=")
            Ckt_obj.dss.Solution.Solve()
    
    # Time-series simulation        
    V_node_Sc = {bus: [] for bus in node_list}
    flow_branch_Sc= {br: [] for br in edge_list}
    powers_Sc = {pv['no']:[] for pv in PVs}
    
    Ckt_obj.dss.Text.Command("Set mode=daily")
    Ckt_obj.dss.Text.Command("Set stepsize=30m")
    Ckt_obj.dss.Text.Command("Set number=1")

    t= 0
    while t<24:
            Ckt_obj.dss.Solution.Solve()
            if attack_type == 1:
                if start_idx * 5 / 60 <= t < end_idx * 5 / 60:
                    for pv in attacked_PVs:
                        Ckt_obj.dss.Text.Command(f"Edit PVSystem.PV{pv['no']} pf={pf_attack}")  # Force poor power factor
                else:
                    for pv in attacked_PVs:
                        Ckt_obj.dss.Text.Command(f"Edit PVSystem.PV{pv['no']} pf=0.95")  # Restore normal
                        
            # Get Node Voltages
            for bus in node_list:
                V=Bus(Ckt_obj,bus).Vmag
                V_node_Sc[bus].append(V)

            # Get PV powers  
            i = Ckt_obj.dss.PVsystems.First()
            while i>0:
                    sno = PVs[i-1]['no']
                    powers_Sc[sno].append(-1*sum(Ckt_obj.dss.CktElement.Powers()[::2])) # Circuit element power
                    i = Ckt_obj.dss.PVsystems.Next()
            
            # Get branchflows  
            for (u,v) in edge_list:
                branch_label = G_original[u][v]['Label']
                branch_device = G_original[u][v]['Device']
                branch_elem = f"{branch_device}.{branch_label}"
                branch_pflow = Branch(Ckt_obj,branch_elem).flow
                branchflow = np.sum(branch_pflow)
                flow_branch_Sc[(u,v)].append(branchflow)                    

            t = Ckt_obj.dss.Solution.DblHour()
    return  V_node_Sc, powers_Sc, flow_branch_Sc

#------------------------------------------------------------------------------------
Scenarios  = []
scid  = 0
### Scenario generation
NSc = 500 # parameter indicating no.of scenarios for each

# Normal case
for idx in range(NSc):
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    attacked_PVs = []
    attack_mult = 1
    pf_attack = 0.95 #normal
    start_idx = end_idx = 0
    attack_type  = -1
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_idx, end_idx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    Scenarios.append({'Index':scid, 'Anomalous':'No', 'Targeted PVs': [], 'Attack Type': 'Nil','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1
    
# Type 0 : Power Curtailment Attack    
for idx in range(NSc):
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    # Select PV(s) which are under attack
    num_attacked = random.randint(1, len(PVs)) # no. of PVs under attack
    attacked_PVs = random.sample(PVs, k=num_attacked) # index of attacked PVs
    # Define PV attack window
    # active window: 10am to 4pm → 120 to 192 (5-min steps)
    start_idx = random.randint(120, 192)  # 10:00 to 16:00  
    max_steps = min(48, 288 - start_idx)    # max 4 hours = 48 steps
    attack_len = random.randint(12, max_steps)  # minimum 1 hour (12 steps)
    end_idx = start_idx + attack_len    
    attack_mult = round(random.uniform(0.1, 0.7), 2) # TYPE 0
    pf_attack = 0.95 #normal
    attack_type = 0
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_idx, end_idx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    PV_targets =[f"PV {pv['no']}" for pv in attacked_PVs]
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted PVs': PV_targets, 'Attack Type': 'PCA','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1   

# Type 1 : Power Factor Attack    
for idx in range(NSc):
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    # Select PV(s) which are under attack
    num_attacked = random.randint(1, len(PVs)) # no. of PVs under attack
    attacked_PVs = random.sample(PVs, k=num_attacked) # index of attacked PVs
    # Define PV attack window
    # active window: 10am to 4pm → 120 to 192 (5-min steps)
    start_idx = random.randint(120, 192)  # 10:00 to 16:00  
    max_steps = min(48, 288 - start_idx)    # max 4 hours = 48 steps
    attack_len = random.randint(12, max_steps)  # minimum 1 hour (12 steps)
    end_idx = start_idx + attack_len  
    attack_mult = 1 # normal
    pf_attack = round(random.uniform(0.5, 0.7), 2) #TYPE 1
    attack_type = 1
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_idx, end_idx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    PV_targets =[f"PV {pv['no']}" for pv in attacked_PVs]
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted PVs': PV_targets, 'Attack Type': 'PFA','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1   

# Type 2 : Denial Of Service Attack    
for idx in range(NSc):
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    # Select PV(s) which are under attack
    num_attacked = random.randint(1, len(PVs)) # no. of PVs under attack
    attacked_PVs = random.sample(PVs, k=num_attacked) # index of attacked PVs
    # Define PV attack window
    # active window: 10am to 4pm → 120 to 192 (5-min steps)
    start_idx = random.randint(120, 192)  # 10:00 to 16:00  
    max_steps = min(48, 288 - start_idx)    # max 4 hours = 48 steps
    attack_len = random.randint(12, max_steps)  # minimum 1 hour (12 steps)
    end_idx = start_idx + attack_len    
    attack_mult = 0 # TYPE 2
    pf_attack = 0.95 #normal
    attack_type = 2
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_idx, end_idx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    PV_targets =[f"PV {pv['no']}" for pv in attacked_PVs]
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted PVs': PV_targets, 'Attack Type': 'DOS','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1   
    
# Shuffle list    
random.shuffle(Scenarios)

# Write list
with open(FolderName+ './PVAttacks_34.pkl', 'wb') as file:
    pickle.dump(Scenarios, file)
