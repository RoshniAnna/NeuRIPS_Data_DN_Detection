
"""
In this file the different types of attacks on grid connected EVCS Systems are simulated.
"""

import os
from scipy.io import loadmat
import pandas as pd
from GraphBuild import * 
import matplotlib.pyplot as plt
import random
import pickle
import uuid
import datetime
import gc
import gzip

# === SLURM array support ===
job_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
total_jobs = 4  # Adjust based on --array=0-3
run_id = uuid.uuid4().hex[:8]
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')



# Loading EVCS information
mat_data = loadmat('chargerdata.mat') #Reading the charging station data
charger_data = mat_data['chargerdata']
stations_bus = 5

Chargers_Select = []
# Loop through each EV charger entry
for charger_name in charger_data.dtype.names:
    daily_time = charger_data[charger_name][0,0]['daily_time'][0][0].flatten()
    daily_charging_power = charger_data[charger_name][0,0]['daily_charging_power'][0][0].flatten()
    daily_charging_current = charger_data[charger_name][0,0]['daily_charging_current'][0][0].flatten()
    daily_charging_voltage = charger_data[charger_name][0,0]['daily_charging_voltage'][0][0].flatten()
    latitude = charger_data[charger_name][0,0]['Latitude'][0][0].flatten()
    longitude = charger_data[charger_name][0,0]['Longitude'][0][0].flatten()
    data = {'Latitude':latitude, 
            'Longitude':longitude, 
            'daily_charging_current':daily_charging_current,
            'daily_charging_power':daily_charging_power* stations_bus,
            'daily_charging_voltage':daily_charging_voltage,
            'daily_time': daily_time}
    Chargers_Select.append({'name':charger_name, 'data':data})

for d in Chargers_Select:
    d['PBase'] = (max(d['data']['daily_charging_power'])*1.05)/1000 # slighlty more than maximum and converting it to kW
    
# Charging Stations
StationsInfo  =[{'no': 1, 'bus': 'l3104136','numphase':3, 'phaseconn':'.1.2.3', 'kV':12.47, 'indx':0, 'kw':Chargers_Select[0]['PBase'], 'kwh':Chargers_Select[0]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 2, 'bus': 'l2895449','numphase':3, 'phaseconn':'.1.2.3', 'kV':12.47, 'indx':1, 'kw':Chargers_Select[1]['PBase'], 'kwh':Chargers_Select[1]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 3, 'bus': 'l3010560','numphase':3, 'phaseconn':'.1.2.3', 'kV':12.47, 'indx':2, 'kw':Chargers_Select[2]['PBase'], 'kwh':Chargers_Select[2]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 4, 'bus': 'l2876797','numphase':3, 'phaseconn':'.1.2.3', 'kV':12.47, 'indx':0, 'kw':Chargers_Select[0]['PBase'], 'kwh':Chargers_Select[0]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 5, 'bus': 'l2876814','numphase':3, 'phaseconn':'.1.2.3', 'kV':12.47, 'indx':1, 'kw':Chargers_Select[1]['PBase'], 'kwh':Chargers_Select[1]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 6, 'bus': 'l3081380','numphase':3, 'phaseconn':'.1.2.3', 'kV':12.47, 'indx':2, 'kw':Chargers_Select[2]['PBase'], 'kwh':Chargers_Select[2]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 7, 'bus': 'l2766718','numphase':3, 'phaseconn':'.1.2.3', 'kV':12.47, 'indx':1, 'kw':Chargers_Select[1]['PBase'], 'kwh':Chargers_Select[1]['PBase']*5.5*3600, 'stored':0, 'reserve':0}]
### Building the Circuit with EVCS

# Initialize Circuit
FolderName = os.path.dirname(os.path.realpath("__file__"))
DSSfile = os.path.join(FolderName, "Master.dss")
Ckt_obj = CircuitSetup(DSSfile, StationsInfo)  #creating a DSS object instance

#-- Equivalent Graph
G_original =  build_graph(Ckt_obj)
nx.readwrite.gml.write_gml(G_original,"8500busEx.gml") #Graph undirected with edge features and node features which are constant    
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
    



#------------------------------------------------------------------------------------
### Define function to simulate attacks and extract time series information

def Powerflow_Timeseries(Ckt_obj, loadshape_day, charging_profiles):
    Ckt_obj = CircuitSetup(DSSfile, StationsInfo)  #creating a DSS object instance
    # Initializing Load Demand shape
    Ckt_obj.dss.Text.Command(f"New LoadShape.LoadVar")
    
    # Initializing Storage Shapes
    for i in range(len(StationsInfo)):
        shape_name = "StorageShape"+str(StationsInfo[i]['no'])
        Ckt_obj.dss.Text.Command(f"New LoadShape.{shape_name}")  
        
    # Edit the LoadShape in opendss
    Ckt_obj.dss.Text.Command(f"Edit LoadShape.LoadVar npts={len(loadshape_day)} interval=1 mult=(" + ' '.join(map(str, loadshape_day)) + ")")
    # Assign Load shapes
    Ckt_obj.dss.Text.Command("BatchEdit Load..* daily=LoadVar")
    
    # Edit the Storage Shape in opendss
    for i in range(len(StationsInfo)):
        shape_name = "StorageShape"+str(StationsInfo[i]['no'])
        charge_P = charging_profiles[i]
        Ckt_obj.dss.Text.Command(f'Edit LoadShape.{shape_name} npts={len(charge_P)} pmult=({ " ".join(map(str, charge_P)) }) sinterval=1')
        # Assign Storage shapes
        elem_name = 'ChargeStatn'+str(StationsInfo[i]['no'])
        Ckt_obj.dss.Text.Command(f'Edit Storage.{elem_name} Daily={shape_name}')
          
    # Time-series simulation        
    V_node_Sc = {bus: [] for bus in node_list}
    flow_branch_Sc= {br: [] for br in edge_list}
    powers_Sc = {stations['no']:[] for stations in StationsInfo}
    
    Ckt_obj.dss.Text.Command("Set mode=daily")
    Ckt_obj.dss.Text.Command("Set stepsize=5m")
    Ckt_obj.dss.Text.Command("Set number=1")
    t= 0
    try:
      while t<24:
              Ckt_obj.dss.Solution.Solve()
              # Get Node Voltages
              for bus in node_list:
                  V=Bus(Ckt_obj,bus).Vmag
                  V_node_Sc[bus].append(V) 
                  
              # Get Storage powers  
              i = Ckt_obj.dss.Storages.First()
              while i>0:
                      sno = StationsInfo[i-1]['no']
                      powers_Sc[sno].append(sum(Ckt_obj.dss.CktElement.Powers()[::2])) # Circuit element power  
                      i= Ckt_obj.dss.Storages.Next()       
                      
              # Get branchflows  
              for (u,v) in edge_list:
                  branch_label = G_original[u][v]['Label']
                  branch_device = G_original[u][v]['Device']
                  branch_elem = f"{branch_device}.{branch_label}"
                  branch_pflow = Branch(Ckt_obj,branch_elem).flow
                  branchflow = np.sum(branch_pflow)
                  flow_branch_Sc[(u,v)].append(branchflow)
                      
              t = Ckt_obj.dss.Solution.DblHour()
              
    except Exception as e:
        print(f"Power flow failed: {e}", flush=True)
        return None, None, None
        
    return  V_node_Sc, powers_Sc, flow_branch_Sc
    
#------------------------------------------------------------------------------------

### Scenario generation
total_scenarios_per_type = 200  # total per type across all jobs
scenarios_per_job = total_scenarios_per_type // total_jobs
batch_size = 50
batch_count = 0
Scenarios = []
scid  = 0
skipped_count = 0
def write_batch():
    global batch_count, Scenarios
    if Scenarios:
        out_name = f"{timestamp}_{run_id}_EVCSAttacks_8500_job{job_index}_part{batch_count}.pkl.gz"
        out_path = os.path.join("results", out_name)
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(Scenarios, f)
        print(f"Batch {batch_count} written with {len(Scenarios)} scenarios.", flush=True)
        batch_count += 1
        Scenarios.clear()
        gc.collect()
        
     
# Normal case
for _ in range(scenarios_per_job):
    print(f"Normal case: Scenario {scid}", flush =True)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    charging_profiles = []
    for i in range(len(StationsInfo)):
        sindx = StationsInfo[i]['indx']
        charge_P = -1*(Chargers_Select[sindx]['data']['daily_charging_power']/(StationsInfo[i]['kw']*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
        charging_profiles.append(charge_P)
    V_node_Sc, powers_Sc, flow_branch_Sc = Powerflow_Timeseries(Ckt_obj, loadshape_day, charging_profiles)
    scid += 1
    if V_node_Sc is None:
      print(f"Skipping scenario {scid-1} due to simulation error.", flush=True)
      skipped_count += 1
      continue        
    Scenarios.append({'Index':scid-1, 'Anomalous':'No', 'Targeted Stations': [], 'Attack Type': 'Nil','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})

    if len(Scenarios) >= batch_size:
        write_batch()

peak_start = 9
peak_end = 19


# Type 1 attack: Peak increase

for _ in range(scenarios_per_job):
    print(f"Type 1: Scenario {scid}", flush =True)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(1.01,1.05)
    charging_profiles = []
    for i in range(len(StationsInfo)):
        sindx = StationsInfo[i]['indx']
        elem_name = 'ChargeStatn'+str(StationsInfo[i]['no'])
        charge_time = Chargers_Select[sindx]['data']['daily_time']
        charge_P=np.zeros(len(charge_time))
        if i in attacked_stations:
            attack_max  = StationsInfo[i]['kw']*attack_mult
            for dindx in range(len(charge_time)): #for each index of charging profile
                # If the charging time is within the attack duration (peak)
                    if (charge_time[dindx] > (peak_start*3600)) and (charge_time[dindx] < (peak_end*3600)):
                        charge_P[dindx] = -1*attack_mult*(Chargers_Select[sindx]['data']['daily_charging_power'][dindx]/(attack_max*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
                    else:
                        charge_P[dindx]=-1*(Chargers_Select[sindx]['data']['daily_charging_power'][dindx]/(attack_max*1000)) 
    
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={attack_max} kWrated={attack_max} model=1 Vminpu=0.1 Vmaxpu=10")
        else:
            charge_P = -1*(Chargers_Select[sindx]['data']['daily_charging_power']/(StationsInfo[i]['kw']*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={StationsInfo[i]['kw']} kWrated={StationsInfo[i]['kw']} model=1 Vminpu=0.1 Vmaxpu=10")    
        
        charging_profiles.append(charge_P)
    V_node_Sc, powers_Sc, flow_branch_Sc = Powerflow_Timeseries(Ckt_obj, loadshape_day, charging_profiles)
    scid += 1
    if V_node_Sc is None:
      print(f"Skipping scenario {scid-1} due to simulation error.", flush=True)
      skipped_count += 1
      continue 
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid-1, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 1', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})

    if len(Scenarios) >= batch_size:
        write_batch()

# Type 2 attack: Time shift and peak increase

for _ in range(scenarios_per_job):
    print(f"Type 2: Scenario {scid}", flush =True)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(1.01,1.05)
    time_shift = random.choice(range(1,4)) #time shift duration
    shift_choice = random.choice([0,1]) #time shift direction 0: left(-ve), 1: right(+ve)
    if shift_choice == 0:
        time_shift = -1*time_shift 
    charging_profiles = []
    for i in range(len(StationsInfo)):
        sindx = StationsInfo[i]['indx']
        elem_name = 'ChargeStatn'+str(StationsInfo[i]['no'])
        charge_time = Chargers_Select[sindx]['data']['daily_time']
        charge_P=np.zeros(len(charge_time))
        if i in attacked_stations:
            attack_max  = StationsInfo[i]['kw']*attack_mult
            for dindx in range(len(charge_time)): #for each index of charging profile
                charge_P[dindx]=-1*(Chargers_Select[sindx]['data']['daily_charging_power'][dindx]/(attack_max*1000))
            # Shifting the charging demand to the left or right
            charge_P[((peak_start+time_shift)*3600):((peak_end+time_shift)*3600)] = attack_mult*charge_P[(peak_start*3600):(peak_end*3600)]    
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={attack_max} kWrated={attack_max} model=1 Vminpu=0.1 Vmaxpu=10")
        else:
            charge_P = -1*(Chargers_Select[sindx]['data']['daily_charging_power']/(StationsInfo[i]['kw']*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={StationsInfo[i]['kw']} kWrated={StationsInfo[i]['kw']} model=1 Vminpu=0.1 Vmaxpu=10")    
        
        charging_profiles.append(charge_P)
    V_node_Sc, powers_Sc, flow_branch_Sc = Powerflow_Timeseries(Ckt_obj, loadshape_day, charging_profiles)
    scid += 1
    if V_node_Sc is None:
      print(f"Skipping scenario {scid-1} due to simulation error.", flush=True)
      skipped_count += 1
      continue 
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid-1, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 2', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})

    if len(Scenarios) >= batch_size:
        write_batch()
        
# Type 3 attack: Demand increase and Time shift

for _ in range(scenarios_per_job):
    print(f"Type 3: Scenario {scid}", flush =True)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(1.01,1.05)
    time_shift = random.choice(range(1,4)) #time shift duration
    shift_choice = random.choice([0,1]) #time shift direction 0: left(-ve), 1: right(+ve)
    if shift_choice == 0:
        time_shift = -1*time_shift 
    charging_profiles = []
    for i in range(len(StationsInfo)):
        sindx = StationsInfo[i]['indx']
        elem_name = 'ChargeStatn'+str(StationsInfo[i]['no'])
        charge_time = Chargers_Select[sindx]['data']['daily_time']
        charge_P=np.zeros(len(charge_time))
        if i in attacked_stations:
            attack_max  = StationsInfo[i]['kw']*attack_mult
            for dindx in range(len(charge_time)): #for each index of charging profile
                # If the charging time is within the attack duration (peak)
                    if (charge_time[dindx] > (peak_start*3600)) and (charge_time[dindx] < (peak_end*3600)):
                        charge_P[dindx] = -1*attack_mult*(Chargers_Select[sindx]['data']['daily_charging_power'][dindx]/(attack_max*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
                    else:
                        charge_P[dindx]=-1*(Chargers_Select[sindx]['data']['daily_charging_power'][dindx]/(attack_max*1000)) 
            # Shifting the charging demand to the right or left
            charge_P[((peak_start+time_shift)*3600):((peak_end+time_shift)*3600)] = charge_P[(peak_start*3600):(peak_end*3600)] 
    
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={attack_max} kWrated={attack_max} model=1 Vminpu=0.1 Vmaxpu=10")
        else:
            charge_P = -1*(Chargers_Select[sindx]['data']['daily_charging_power']/(StationsInfo[i]['kw']*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={StationsInfo[i]['kw']} kWrated={StationsInfo[i]['kw']} model=1 Vminpu=0.1 Vmaxpu=10")    
        
        charging_profiles.append(charge_P)
    V_node_Sc, powers_Sc, flow_branch_Sc = Powerflow_Timeseries(Ckt_obj, loadshape_day, charging_profiles)
    scid += 1
    if V_node_Sc is None:
      print(f"Skipping scenario {scid-1} due to simulation error.", flush=True)
      skipped_count += 1
      continue 
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid-1, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 3', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})

    if len(Scenarios) >= batch_size:
        write_batch()
        
# Type 4 attack: Time shift peak and additional peak 
for _ in range(scenarios_per_job):
    print(f"Type 4: Scenario {scid}", flush =True)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(1.01,1.05)
    attack_mult2 = random.uniform(1.01,1.02) # attack scaling factor for type 6
    time_shift = random.choice(range(1,4)) #time shift duration
    shift_choice = random.choice([0,1]) #time shift direction 0: left(-ve), 1: right(+ve)
    if shift_choice == 0:
        time_shift = -1*time_shift 
    charging_profiles = []
    for i in range(len(StationsInfo)):
        sindx = StationsInfo[i]['indx']
        elem_name = 'ChargeStatn'+str(StationsInfo[i]['no'])
        charge_time = Chargers_Select[sindx]['data']['daily_time']
        charge_P=np.zeros(len(charge_time))
        if i in attacked_stations:
            attack_max  = StationsInfo[i]['kw']*attack_mult
            for dindx in range(len(charge_time)): #for each index of charging profile
                # If the charging time is within the attack duration (peak)
                    if (charge_time[dindx] > (peak_start*3600)) and (charge_time[dindx] < (peak_end*3600)):
                        charge_P[dindx] = -1*attack_mult*(Chargers_Select[sindx]['data']['daily_charging_power'][dindx]/(attack_max*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
                    else:
                        charge_P[dindx]=-1*(Chargers_Select[sindx]['data']['daily_charging_power'][dindx]/(attack_max*1000)) 
            # Shifting the charging demand to the right or left
            charge_P[((peak_start+time_shift)*3600):((peak_end+time_shift)*3600)] = attack_mult2*charge_P[(peak_start*3600):(peak_end*3600)] 
    
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={attack_max} kWrated={attack_max} model=1 Vminpu=0.1 Vmaxpu=10")
        else:
            charge_P = -1*(Chargers_Select[sindx]['data']['daily_charging_power']/(StationsInfo[i]['kw']*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
            Ckt_obj.dss.Text.Command(f"Edit Storage.{elem_name} kW={StationsInfo[i]['kw']} kWrated={StationsInfo[i]['kw']} model=1 Vminpu=0.1 Vmaxpu=10")    
        
        charging_profiles.append(charge_P)
    V_node_Sc, powers_Sc, flow_branch_Sc = Powerflow_Timeseries(Ckt_obj, loadshape_day, charging_profiles)
    scid += 1
    if V_node_Sc is None:
      print(f"Skipping scenario {scid-1} due to simulation error.", flush=True)
      skipped_count += 1
      continue 
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid-1, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 4', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})

    if len(Scenarios) >= batch_size:
        write_batch()

write_batch()       

print(f"No of skipped scenarios:{skipped_count}") 

