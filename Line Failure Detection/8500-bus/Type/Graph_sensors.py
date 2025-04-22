# -*- coding: utf-8 -*-
"""
To create the graph with and without sensors
Reduced graph

@author: raj180002
"""
import os
import networkx as nx
import numpy as np

FolderName = os.path.dirname(os.path.realpath("__file__"))
G_original = nx.read_gml(""+ FolderName+ r"\8500busEx.gml")


# Read the buses without sensors and lines without sensors to construct graph with sensors
missing_nodes = np.load('BuswoSensors.npy')
missing_edge_labels = np.load('LinewoSensors.npy') #This has the circuit branch name (label)

# Sensor Edges
missing_edges = [] # Converting the missing labels to graph end nodes 
sensor_edges = [] # Extracting sensor edges
for e in G_original.edges(data=True):
    ename =  e[2]['device']+ '.' + e[2]['label']
    if ename in missing_edge_labels:
       missing_edges.append((e[0],e[1]))
    else:
       sensor_edges.append((e[0],e[1])) #----this is the list of edges with sensors  
    
missing_edges = np.array(missing_edges) 


# Sensor Nodes
sensor_nodes = list(set(G_original.nodes()).difference(set(missing_nodes))) #---this is the list of nodes with sensors


# Construct a new graph where the sensor nodes and all its edges(incoming and outgoing) are present
# also the sensor edges are present

G_sensor = nx.Graph()

for (u,v) in sensor_edges: #for each sensor edges
    edge_dict = G_original.get_edge_data(u,v) 
    G_sensor.add_edge(u, v, label=edge_dict['label'], device=edge_dict['device'], resistance=edge_dict['resistance'], reactance=edge_dict['reactance'], phases=edge_dict['phases'], maxcap=edge_dict['maxcap'], cap=edge_dict['cap'], residue=edge_dict['residue'])


for n in sensor_nodes: #for each sensor node
    for (u,v) in nx.edges(G_original,n): # for each edge incident on node n
         if not(G_sensor.has_edge(u,v) or G_sensor.has_edge(v,u)):
             edge_dict = G_original.get_edge_data(u,v) 
             G_sensor.add_edge(u, v, label=edge_dict['label'], device=edge_dict['device'], resistance=edge_dict['resistance'], reactance=edge_dict['reactance'], phases=edge_dict['phases'], maxcap=edge_dict['maxcap'], cap=edge_dict['cap'], residue=edge_dict['residue'])


# Write the graph--- this is the sensor graph 
nx.readwrite.gml.write_gml(G_sensor, "SensorGraph_8500.gml")



