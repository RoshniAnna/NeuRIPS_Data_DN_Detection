# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:02:08 2025

@author: dro210000
"""

import os
import math
import csv
import pickle
import statistics
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from random import sample,choices,choice,uniform, randint,shuffle
from DSSCircuit_Interface import *

FolderName=os.path.dirname(os.path.realpath("__file__"))
G_original = nx.read_gml(""+ FolderName+ r"\34busEx.gml")
nodes=list(G_original.nodes())
edges=list(G_original.edges())

DSSfile=r""+ FolderName+ "\ieee34Mod2.dss" 
d_obj= DSS(DSSfile) #create a circuit object
d_obj.compile_ckt_dss() #compiling the circuit
d_obj.dssText.command = "Set Maxiterations=200" 
d_obj.dssText.command = "Set maxcontroliter=200"
d_obj.AllowForms=False 
# d_obj.dssText.command = "Set controlmode=OFF"

#Creating Array of Loadshapes
LoadShape = []
input_file = 'LoadShape1.xlsx'
sheet_name = 'LoadShape1'  # Change to the sheet you want to split
df = pd.read_excel(input_file, sheet_name=sheet_name)

# Number of parts to split the file into
num_files = 365  

# Calculate the number of rows per file
rows_per_file = math.ceil(len(df) / num_files)

# Split and save to new Excel files
for i in range(num_files):
    start_row = i * rows_per_file
    end_row = min((i + 1) * rows_per_file, len(df))
    
    # Create a new DataFrame for each chunk
    chunk_df = df[start_row:end_row]
    
    # Convert dataframe to list
    loadshape = []
    value = chunk_df.values.tolist()
    for x in range(len(value)):
        data = float(value[x][0])
        loadshape.append(data)    
    LoadShape.append(loadshape)
    
#Loading PV information
PVs = [{'no':1, 'bus':'890', 'numphase':3, 'phaseconn':'.1.2.3', 'size':146, 'kV':4.16, 'KVA':146},
        {'no':2, 'bus':'806', 'numphase':3, 'phaseconn':'.1.2.3', 'size':144, 'kV':24.9, 'KVA':144},
        {'no':3, 'bus':'816', 'numphase':3, 'phaseconn':'.1.2.3', 'size':200, 'kV':24.9, 'KVA':200}]
temp =[25, 25, 25, 25, 25, 25, 25, 25, 40, 50, 60, 65, 65, 65, 57, 50, 35, 30,  25, 25, 25, 25, 25, 25]
Time = range(0,24)

#Obtaining list of buses connected to PVSystems
PVbuses = []
for pv in range(len(PVs)):
    PVbuses.append(PVs[pv]['bus'])
    
#Adding PVs to DSS Circuit
d_obj.dssText.command='redirect PVSystem.dss'
# d_obj.dssText.command = f"New LoadShape.MyIrrad0 npts=288 minterval=5 mult={Irradiance[0]} Action=Normalize"
d_obj.dssText.command = f"New Tshape.MyTemp npts=24 interval=1 temp={temp}"
for pv in range(len(PVs)):
        d_obj.dssText.command=f"New PVSystem.PV{PVs[pv]['no']} bus1={PVs[pv]['bus'] + PVs[pv]['phaseconn']} phases={str(PVs[pv]['numphase'])} kv={str(PVs[pv]['kV'])} kVA={str(PVs[pv]['KVA'])} pf=0.8 Pmpp={str(PVs[pv]['size'])} %cutin=0.1 %cutout=0.1 effcurve=Myeff P-TCurve=MyPvsT Daily=PVLoadShape2 TDaily=MyTemp"
 
        
##Generating Normal Scenarios##
buses=list(G_original.nodes())
edge_dict=list(G_original.edges(data=True)) 
Prelim_Dataset = []
Nsc_N = 1
count = 0
for i in range(Nsc_N):
    d_obj.dssText.command = f"New LoadShape.Lshape{i} npts=24 interval=1 mult={LoadShape[i]}"
    d_obj.dssText.command = f"Batchedit Load..* Daily=Lshape{i}"
    d_obj.dssText.command = "Set mode=Daily"
    d_obj.dssText.command = "Set stepsize=5m"
    d_obj.dssText.command = "Set number=1"
    #d_obj.dssText.command = f"New LoadShape.MyIrrad{i} npts=288 minterval=5 mult={Irradiance[i]} Action=Normalize"
    Normal_Data = []
    TimeSeries_Data = []
    nbua = randint(1,3)
    dsstime = 0
    while dsstime<24: 
        d_obj.dssText.Command = 'Solve'
        dsstime = d_obj.dssSolution.dblHour
        Branchflow_Data = []
        Bus_Data = []
        for pv in range(len(PVs)):
            d_obj.dssText.command=f"Edit PVSystem.PV{PVs[pv]['no']} bus1={PVs[pv]['bus'] + PVs[pv]['phaseconn']} phases={str(PVs[pv]['numphase'])} kv={str(PVs[pv]['kV'])} kVA={str(PVs[pv]['KVA'])} pf=0.8 Pmpp={str(PVs[pv]['size'])} %cutin=0.1 %cutout=0.1 effcurve=Myeff P-TCurve=MyPvsT Daily=PVLoadShape2 TDaily=MyTemp"
        for b in buses:
            d_obj.dssCircuit.SetActiveBus(b) #Setting Active Bus
            node_num=np.array(d_obj.dssBus.Nodes)
            bus_size = node_num.size
            V=np.array(d_obj.dssBus.puVmagAngle)
            ctidx = 2 * np.array(range(0, min(int(V.size/ 2), 3))) 
            if b in PVbuses:
                j= d_obj.dssCircuit.PVSystems.First
                while j>0:
                    elemName = d_obj.dssCircuit.ActiveCktElement.Name
                    busconnectn = d_obj.dssCircuit.ActiveCktElement.BusNames[0].split('.')[0]
                    if b == busconnectn:
                        S=np.array(d_obj.dssCircuit.ActiveCktElement.Powers)
                        P_Gen = S[ctidx]
                        break
                    j=d_obj.dssCircuit.PVSystems.Next
            else:
                P_Gen=np.empty(bus_size) * np.nan
            Bus_Data.append({'Bus Name':b, 'Voltage Magnitude':V[ctidx], 'Power Generated':P_Gen})
        for e in G_original.edges(data=True):
            name = e[2]['device'] + '.' +e[2]['label']
            d_obj.dssCircuit.SetActiveElement(name) #Setting Active Edge Element
            c=np.array(d_obj.dssCktElement.CurrentsMagAng)
            ctidx = 2 * np.array(range(0, min(int(c.size/ 4), 3)))
            I_mag = c[ctidx]
            I_avg=np.average(I_mag)
            Branchflow_Data.append({'Branch Name':name,'BranchFlow':I_avg}) 
        TimeSeries_Data.append({'Time Step':dsstime, 'Bus_Info': Bus_Data, 'Branchflow':Branchflow_Data})
    count = count + 1
    print(count)
    Scenario_Info={'Scenario_Number':count, 'Anomalous': 'No', 'Attack_Type':'Nil', 'Affected_PV':'Nil', 'TimeSeries_Data':TimeSeries_Data}
    Prelim_Dataset.append(Scenario_Info)
    
    
##Restructuring Dataset into multi-dimensional array##
Dataset = []
Ttot = 288
dim_x = Ttot
dim_Vy = len(buses)
dim_Cy = len(edges)
for d in range (len(Prelim_Dataset)):
    Varray_3d = np.zeros((dim_x,dim_Vy,3))
    Varray_2d = np.zeros((dim_Vy,3))
    Carray_3d = np.zeros((dim_x,dim_Cy,1))
    Carray_2d = np.zeros((dim_Cy,1))
    Parray_3d = np.zeros((dim_x,dim_Vy,3))
    Parray_2d = np.zeros((dim_Vy,3))
    for t in range(Ttot):
        for v in range(len(buses)):
            Varray_2d[v,:] = Prelim_Dataset[d]['TimeSeries_Data'][t]['Bus_Info'][v]['Voltage Magnitude']
            Parray_2d[v,:] = -Prelim_Dataset[d]['TimeSeries_Data'][t]['Bus_Info'][v]['Power Generated']
        Varray_3d[t,:,:]=Varray_2d
        Parray_3d[t,:,:]=Parray_2d
        for c in range(len(edges)):
            Carray_2d[c,:] = Prelim_Dataset[d]['TimeSeries_Data'][t]['Branchflow'][c]['BranchFlow']
        Carray_3d[t,:,:]=Carray_2d       
    Scenario_Data = {'Anomalous':Prelim_Dataset[d]['Anomalous'],'Attack_Type':'Nil','Targeted_PV':Prelim_Dataset[d]['Affected_PV'],'TimeSeries_PowerGen':Parray_3d, 'TimeSeries_Voltage':Varray_3d, 'BranchFlow':Carray_3d}
    Dataset.append(Scenario_Data)    
    
#Loading .pkl file
a = open('PVanomalydataset_norm(mult).pkl', 'wb')
pickle.dump(Dataset,a)
a.close()