
"""
In this file the different types of attacks on grid connected PV Systems is simulated
"""

import os
import pandas as pd
from GraphBuild import * 
import matplotlib.pyplot as plt
import random
import pickle
import uuid
import datetime

run_id = uuid.uuid4().hex[:8]
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# === SLURM array support ===
job_indx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))  # SLURM provides 0,1,2,...
total_jobs = 10  # match your SLURM --array=0-9
NSc = 10000
NSc_per_type = 2500  # total per attack type
job_size_per_type = NSc_per_type // total_jobs  # 250 per job

print(f"Running SLURM array job {job_indx}: total {job_size_per_type} scenarios per attack type", flush=True)

### Loading PV information
PVs = [{'no':1, 'bus':'l3104136', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1000,'kV':12.47, 'KVA':1000},
        {'no':2, 'bus':'l2895449', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1000, 'kV':12.47, 'KVA':1000},
        {'no':3, 'bus':'l3010560', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1050, 'kV':12.47, 'KVA':1050},
        {'no':4, 'bus':'l2876797', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1050, 'kV':12.47, 'KVA':1050},
        {'no':5, 'bus':'l2876814', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1100, 'kV':12.47, 'KVA':1100},
        {'no':6, 'bus':'l3081380', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1200, 'kV':12.47, 'KVA':1200},
        {'no':7, 'bus':'l2766718', 'numphase':3, 'phaseconn':'.1.2.3', 'size':1500, 'kV':12.47, 'KVA':1500}]

### Building circuit with PV

#--Initialize Circuit
FolderName = os.path.dirname(os.path.realpath("__file__"))
DSSfile = os.path.join(FolderName, "Master.dss")
Ckt_obj = CircuitSetup(DSSfile, PVs)  #creating a DSS object instance
output_dir = os.path.join(FolderName, "results")
os.makedirs(output_dir, exist_ok=True)

#-- Equivalent Graph
G_original =  build_graph(Ckt_obj)
nx.readwrite.gml.write_gml(G_original,"8500NodeEx.gml") #Graph undirected with edge features and node features which are constant
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
    start_tidx = i * points_per_day
    end_tidx = (i + 1) * points_per_day
    daily_shape = loadshape_values[start_tidx:end_tidx]
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
    start_ridx = i * points_per_day
    end_ridx = (i + 1) * points_per_day
    daily_ghi = ghi_values[start_ridx:end_ridx]
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
                branchflow = Branch(Ckt_obj,branch_elem).flow
                flow_branch_Sc[(u,v)].append(branchflow)                    

            t = Ckt_obj.dss.Solution.DblHour()
    return  V_node_Sc, powers_Sc, flow_branch_Sc

#------------------------------------------------------------------------------------
Scenarios  = []
scid = 0
# Normal case
for idx in range(job_size_per_type):
    print(f"Running SLURM array job {job_indx}: generating {job_size_per_type} Normal type", flush=True)
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    attacked_PVs = []
    attack_mult = 1
    pf_attack = 0.95 #normal
    start_nidx = end_nidx = 0
    attack_type  = -1
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_nidx, end_nidx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    Scenarios.append({'Index':scid, 'Anomalous':'No', 'Targeted PVs': [], 'Attack Type': 'Nil','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1
    
outfile = os.path.join(output_dir, f'{timestamp}_{run_id}_PVAttacks_8500_job{job_indx}_normal.pkl')
with open(outfile, 'wb') as file:
    pickle.dump(Scenarios, file)

Scenarios = [] 
scid = 0
# Type 0 : Power Curtailment Attack    
for idx in range(job_size_per_type):
    print(f"Running SLURM array job {job_indx}: generating {job_size_per_type} PCA type", flush=True)
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    # Select PV(s) which are under attack
    num_attacked = random.randint(1, len(PVs)) # no. of PVs under attack
    attacked_PVs = random.sample(PVs, k=num_attacked) # index of attacked PVs
    # Define PV attack window
    # active window: 10am to 4pm → 120 to 192 (5-min steps)
    start_pidx = random.randint(120, 192)  # 10:00 to 16:00  
    max_steps = min(48, 288 - start_pidx)    # max 4 hours = 48 steps
    attack_len = random.randint(12, max_steps)  # minimum 1 hour (12 steps)
    end_pidx = start_pidx + attack_len    
    attack_mult = round(random.uniform(0.1, 0.7), 2) # TYPE 0
    pf_attack = 0.95 #normal
    attack_type = 0
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_pidx, end_pidx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    PV_targets =[f"PV {pv['no']}" for pv in attacked_PVs]
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted PVs': PV_targets, 'Attack Type': 'PCA','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1

outfile = os.path.join(output_dir, f'{timestamp}_{run_id}_PVAttacks_8500_job{job_indx}_PCA.pkl')
with open(outfile, 'wb') as file:
    pickle.dump(Scenarios, file)

Scenarios = []
scid  = 0
# Type 1 : Power Factor Attack    
for idx in range(job_size_per_type):
    print(f"Running SLURM array job {job_indx}: generating {job_size_per_type} PFA type", flush=True)
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    # Select PV(s) which are under attack
    num_attacked = random.randint(1, len(PVs)) # no. of PVs under attack
    attacked_PVs = random.sample(PVs, k=num_attacked) # index of attacked PVs
    # Define PV attack window
    # active window: 10am to 4pm → 120 to 192 (5-min steps)
    start_fidx = random.randint(120, 192)  # 10:00 to 16:00  
    max_steps = min(48, 288 - start_fidx)    # max 4 hours = 48 steps
    attack_len = random.randint(12, max_steps)  # minimum 1 hour (12 steps)
    end_fidx = start_fidx + attack_len  
    attack_mult = 1 # normal
    pf_attack = round(random.uniform(0.5, 0.7), 2) #TYPE 1
    attack_type = 1
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_fidx, end_fidx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    PV_targets =[f"PV {pv['no']}" for pv in attacked_PVs]
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted PVs': PV_targets, 'Attack Type': 'PFA','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1
    
outfile = os.path.join(output_dir, f'{timestamp}_{run_id}_PVAttacks_8500_job{job_indx}_PFA.pkl')
with open(outfile, 'wb') as file:
    pickle.dump(Scenarios, file)
    
    
Scenarios = []
scid = 0

# Type 2 : Denial Of Service Attack    
for idx in range(job_size_per_type):
    print(f"Running SLURM array job {job_indx}: generating {job_size_per_type} DOS type", flush=True)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    irradiance_day =random.choice(IrradianceShapes)   # 288 points
    # Select PV(s) which are under attack
    num_attacked = random.randint(1, len(PVs)) # no. of PVs under attack
    attacked_PVs = random.sample(PVs, k=num_attacked) # index of attacked PVs
    # Define PV attack window
    # active window: 10am to 4pm → 120 to 192 (5-min steps)
    start_didx = random.randint(120, 192)  # 10:00 to 16:00  
    max_steps = min(48, 288 - start_didx)    # max 4 hours = 48 steps
    attack_len = random.randint(12, max_steps)  # minimum 1 hour (12 steps)
    end_didx = start_didx + attack_len    
    attack_mult = 0 # TYPE 2
    pf_attack = 0.95 #normal
    attack_type = 2
    V_node_Sc, powers_Sc, flow_branch_Sc = inject_pv_attack(Ckt_obj, PVs, attacked_PVs, start_didx, end_didx, attack_type, attack_mult, pf_attack, loadshape_day, irradiance_day)
    PV_targets =[f"PV {pv['no']}" for pv in attacked_PVs]
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted PVs': PV_targets, 'Attack Type': 'DOS','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'PV power series':powers_Sc})
    scid = scid + 1
    
outfile = os.path.join(output_dir, f'{timestamp}_{run_id}_PVAttacks_8500_job{job_indx}_DOS.pkl')
with open(outfile, 'wb') as file:
    pickle.dump(Scenarios, file)
    

