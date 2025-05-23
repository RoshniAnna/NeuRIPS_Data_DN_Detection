
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
            'daily_charging_power':daily_charging_power * stations_bus,
            'daily_charging_voltage':daily_charging_voltage,
            'daily_time': daily_time}
    Chargers_Select.append({'name':charger_name, 'data':data})

for d in Chargers_Select:
    d['PBase'] = (max(d['data']['daily_charging_power'])*1.05)/1000 # slighlty more than maximum and converting it to kW

# Charging Stations
StationsInfo  =[{'no': 1, 'bus': '25','numphase':3, 'phaseconn':'.1.2.3', 'kV':4.16, 'indx':0, 'kw':Chargers_Select[0]['PBase'], 'kwh':Chargers_Select[0]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 2, 'bus': '40','numphase':3, 'phaseconn':'.1.2.3', 'kV':4.16, 'indx':1, 'kw':Chargers_Select[1]['PBase'], 'kwh':Chargers_Select[1]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 3, 'bus': '54','numphase':3, 'phaseconn':'.1.2.3', 'kV':4.16, 'indx':2, 'kw':Chargers_Select[2]['PBase'], 'kwh':Chargers_Select[2]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 4, 'bus': '62','numphase':3, 'phaseconn':'.1.2.3', 'kV':4.16, 'indx':0, 'kw':Chargers_Select[0]['PBase'], 'kwh':Chargers_Select[0]['PBase']*5.5*3600, 'stored':0, 'reserve':0},
                {'no': 5, 'bus': '76','numphase':3, 'phaseconn':'.1.2.3', 'kV':4.16, 'indx':2, 'kw':Chargers_Select[2]['PBase'], 'kwh':Chargers_Select[2]['PBase']*5.5*3600, 'stored':0, 'reserve':0}]

### Building the Circuit with EVCS

# Initialize Circuit
FolderName = os.path.dirname(os.path.realpath("__file__"))
DSSfile = os.path.join(FolderName, "IEEE123Master.dss")
Ckt_obj = CircuitSetup(DSSfile, StationsInfo)  #creating a DSS object instance

#-- Equivalent Graph
G_original =  build_graph(Ckt_obj)
nx.readwrite.gml.write_gml(G_original,"123busEx.gml") #Graph undirected with edge features and node features which are constant    
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
        
    return  V_node_Sc, powers_Sc, flow_branch_Sc
    
#------------------------------------------------------------------------------------

Scenarios  = []
scid  = 0
### Scenario generation
NSc = 250 # parameter indicating no.of scenarios for each

# Normal case
for idx in range(NSc):
    print(scid)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    charging_profiles = []
    for i in range(len(StationsInfo)):
        sindx = StationsInfo[i]['indx']
        charge_P = -1*(Chargers_Select[sindx]['data']['daily_charging_power']/(StationsInfo[i]['kw']*1000)) # converting the charging power to kW, normalizing wrt to max Power and -ve (indicate charging)
        charging_profiles.append(charge_P)
    V_node_Sc, powers_Sc, flow_branch_Sc = Powerflow_Timeseries(Ckt_obj, loadshape_day, charging_profiles)        
    Scenarios.append({'Index':scid, 'Anomalous':'No', 'Targeted Stations': [], 'Attack Type': 'Nil','BusVoltage series':V_node_Sc,'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})
    scid = scid + 1

peak_start = 9
peak_end = 19


# Type 1 attack: Peak increase

for idx in range(NSc):
    print(scid)
    # Edit load shapes (Load and PV) for each scenario
    
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(5,7)
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
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 1', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})
    scid = scid + 1


# Type 2 attack: Time shift and peak increase

for idx in range(NSc):
    print(scid)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(5,7)
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
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 2', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})
    scid = scid + 1

# Type 3 attack: Demand increase and Time shift

for idx in range(NSc):
    print(scid)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(5,7)
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
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 3', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})
    scid = scid + 1

# Type 4 attack: Time shift peak and additional peak 
for idx in range(NSc):
    print(scid)
    # Edit load shapes (Load and PV) for each scenario
    loadshape_day = random.choice(LoadShapes) # 24 points
    # Select Storages(s) which are under attack
    num_attacked = random.randint(1, len(StationsInfo)) # no. of PVs under attack
    attacked_stations = random.sample(range(len(StationsInfo)), k=num_attacked) # index of attacked PVs
    attack_mult = random.uniform(5,7)
    attack_mult2 = random.uniform(1.01,2.5) # attack scaling factor for type 6
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
    EVCS_targets = [f"EVCS {StationsInfo[atk]['no']}" for atk in attacked_stations]       
    Scenarios.append({'Index':scid, 'Anomalous':'Yes', 'Targeted Stations': EVCS_targets, 'Attack Type': 'Type 4', 'BusVoltage series':V_node_Sc, 'BranchFlow series':flow_branch_Sc,'EVCS power series':powers_Sc})
    scid = scid + 1
        
# Shuffle list    
random.shuffle(Scenarios)

# Write list
with open(os.path.join(FolderName, 'EVCSAttacks_123_1.pkl'), 'wb') as file:
    pickle.dump(Scenarios, file)   
