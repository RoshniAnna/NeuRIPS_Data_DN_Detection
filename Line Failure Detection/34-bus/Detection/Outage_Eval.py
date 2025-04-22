# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:18:43 2021

@author: raj180002
"""

import win32com.client # DSS COM interace
import os
import numpy as np
import networkx as nx
from DSSCircuit_Interface import *

# def Outage_evaluate(DSSfile,Out_list):
#     d_obj= DSS(DSSfile) #create a circuit object
#     d_obj.compile_ckt_dss() #compiling the circuit
#     LoadNames=list(d_obj.dssLoads.AllNames)
#     load_dict=[]
#     d_obj.solve_snapshot_dss(1.0) #solving snapshot power flow
#     for e in Out_list:
#         d_obj.dssText.Command=('Edit '+ e['device'] + '.' +  e['label'] + ' enabled=False') #disable to component
#         d_obj.solve_snapshot_dss(1.0) #solving snapshot after removing elements

#     for ld in list(d_obj.dssLoads.AllNames):
#         d_obj.dssCircuit.SetActiveElement("Load." + ld) #set the load as the active element
#         bus=d_obj.dssCktElement.Busnames[0].split('.')[0]
#         S=np.array(d_obj.dssCktElement.Powers)
#         ctidx = 2 * np.array(range(0, min(int(S.size/ 2), 3)))
#         P = S[ctidx] #active power in KW
#         Q =S[ctidx + 1] #angle in KVA
#         Power_Supp=sum(P)
#         if np.isnan(Power_Supp):
#             Power_Supp=0.0
#         load_dict.append({'Load Name':ld, 'Bus':bus,  'Power':Power_Supp})
    
    
    # P,Q,P_loss,Q_loss=d_obj.get_results_dss()
    # P_supplied=P-(P_loss/1000)
    
    # return(load_dict,P_supplied)
    
#----------New Code----------------------

def Outage_evaluate(DSSfile,Outlist, Multfact, G_original):
    buses=list(G_original.nodes())
    edges=list(G_original.edges())
    loc_d_obj= DSS(DSSfile) #create a circuit object
    loc_d_obj.compile_ckt_dss() #compiling the circuit
    loc_d_obj.dssText.command = "Set Maxiterations=200" 
    loc_d_obj.dssText.command = "Set maxcontroliter=200"
    loc_d_obj.AllowForms=False 
    loc_d_obj.dssText.command = "Set controlmode=OFF"
    loc_d_obj.dssSolution.Solve()
    
    for e in Outlist:
        loc_d_obj.dssText.Command=('Edit '+ e['device'] + '.' +  e['label'] + ' enabled=False') #disable to component
        loc_d_obj.dssSolution.Solve()
    
    loc_d_obj.dssSolution.LoadMult=Multfact
    loc_d_obj.dssSolution.Solve()
    
    #####Getting load information ####
    LoadNames=list(loc_d_obj.dssLoads.AllNames) #list of all loads
    load_dict=[] # get the power supplied
    for ld in LoadNames:
        loc_d_obj.dssCircuit.SetActiveElement("Load." + ld) #set the load as the active element
        bus=loc_d_obj.dssCktElement.Busnames[0].split('.')[0] #get the bus connection of the load
        S=np.array(loc_d_obj.dssCktElement.Powers) #get the apparent power array of the load
        ctidx = 2 * np.array(range(0, min(int(S.size/ 2), 3)))
        P = S[ctidx] #active power in KW #get the active power array of the load
        #Q =S[ctidx + 1] #angle in KVAr
        Power_supplied=sum(P) #get the total active power served for the load
        loc_d_obj.dssCircuit.Loads.Name =ld 
        Power_demand = (loc_d_obj.dssCircuit.Loads.kW) * Multfact  # base case multiplication factor is maximum
        load_dict.append({'Load Name':ld, 'Bus':bus,  'PowerSupply':Power_supplied, 'PowerDemand':Power_demand}) #create a list of the dictionary for each load 
        
    out_flag=0
    NetOutage='No'
    Bus_Pserved={}
    Bus_PDemand={}
    Bus_Voltages={}
    for b in buses:
        loadnames=[] #name of loads connected to bus
        P_served = 0
        P_dem = 0
        for j in range(len(load_dict)):
            if load_dict[j]["Bus"] == b: #to get bus connection of loads
                loadnames.append(load_dict[j]['Load Name'])
                P_served=P_served + load_dict[j]['PowerSupply']
                P_dem=P_dem + load_dict[j]['PowerDemand']
        
        if (len(loadnames)!=0) and (P_served<0.1): #if load exists and power is 0
            out_flag=1
            
        bus_obj = Bus(loc_d_obj,b)
        Bus_Pserved[b] = P_served
        Bus_PDemand[b] = P_dem
        Bus_Voltages[b] = bus_obj.V
        
    BranchFlow={}    
    for e in G_original.edges(data=True):
        name = e[2]['device'] + '.' +e[2]['label']
        branch_obj = Branch(loc_d_obj,name)
        BranchFlow[name]=branch_obj.Cap
    if out_flag==1:
       NetOutage='Yes'    
       
    return (Bus_Pserved,Bus_PDemand,Bus_Voltages,BranchFlow,NetOutage)    