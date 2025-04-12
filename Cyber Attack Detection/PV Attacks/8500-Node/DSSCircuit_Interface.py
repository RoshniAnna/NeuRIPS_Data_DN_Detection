"""
Updated on  9/14/22
@author: Roshni Anna Jacob
Contains DSS circuit object definitions, bus object and branch object definitions
"""

import win32com.client # DSS COM interace
import os
import math
import cmath
import numpy as np


class DSS():  # to initialize the DSS circuit object and extract results
      
    def __init__(self,filename): 
        
        self.filename=filename
        self.dssObj=win32com.client.Dispatch("OpenDSSEngine.DSS") #deploying DSS Engine
        
        if self.dssObj.Start(0)==False:
           print ("Problem with OpenDSS Engine initialization")
           
        else: # redeclaring variables defined under DSS object
            self.dssText=self.dssObj.Text
            self.dssCircuit=self.dssObj.ActiveCircuit
            self.dssSolution=self.dssCircuit.Solution            
            self.dssTopology=self.dssCircuit.Topology
            self.dssBus = self.dssCircuit.ActiveBus
            self.dssCktElement=self.dssCircuit.ActiveCktElement
            self.dssLines=self.dssCircuit.Lines
            self.dssLoads=self.dssCircuit.Loads
            self.dssTransformers=self.dssCircuit.Transformers
    
    
    def version_dss(self):    # specifies the version of OpenDSS used
          return self.dssObj.Version            
            
    def compile_ckt_dss(self): # Compiling the OpenDSS circuit
        self.dssObj.ClearAll()
        self.dssText.Command="compile [" + self.filename +"]" 
        
    def get_cktname_dss(self): # The circuit name
        return self.dssCircuit.Name   
    
    def solve_snapshot_dss(self,loadmultFac): #solving snapshot powerflow for particular load multiplication factor
        self.dssText.Command="Set Mode=SnapShot"
        self.dssText.Command="Set ControlMode=OFF"
        self.dssSolution.LoadMult=loadmultFac
        self.dssSolution.Solve() 
        
    def get_results_dss(self): # total active and reactive power after power flow
        P= -1*(self.dssCircuit.Totalpower[0]) #active power in kW
        Q= -1*(self.dssCircuit.Totalpower[1]) #reactive power in kW
        losses=self.dssCircuit.Losses
        P_loss=(losses[0]) #active power loss in W
        Q_loss=(losses[1]) #reactive power loss in W
        return P,Q,P_loss,Q_loss   
    
    def get_AllBuses(self): #to get all the bus names
        return self.dssCircuit.AllBusNames
       
    def get_ckt_base(self):
      self.dssTransformers.First
      KVA_base=self.dssTransformers.kva
      KV_base=self.dssTransformers.kv
      Z_base=((KV_base**2)*1000)/KVA_base
      return(KVA_base,KV_base,Z_base)
  
    
    def get_AllLoads(self):
        Load_dict=[]
        i=self.dssCircuit.FirstPCElement() #set first power conversion element active
        while i>0:
              elname=self.dssCktElement.Name
              if elname.split('.')[0]=='Load':
                  name=elname.split('.')[1]
                  bus=self.dssCktElement.Busnames[0].split('.')[0]
                  self.dssLoads.Name=name
                  P_load=self.dssLoads.kW
                  Q_load=self.dssLoads.kvar
                  Load_dict.append({'Name':name, 'Bus':bus, 'Pload':P_load,'Qload':Q_load})
              i=self.dssCircuit.NextPCElement()
        return(Load_dict)
                                    
     
    # def get_AllLines(self): #to get all line names
    #     return self.dssLines.AllNames
    
    # def get_AllTransformers(self): #to get all line names
    #     return self.dssTransformers.AllNames
    
    def get_AllPDElements(self): #to get all Power delivery element names
        elem=[]
        i=self.dssCircuit.PDElements.First
        while i>0:
              elem.append(self.dssCircuit.PDElements.Name)
              i=self.dssCircuit.PDElements.Next
        return (elem)
    
    def get_Line_RX(self,name): #to get the line resistance and reactance matrix and length
        self.dssLines.Name=name #activating the line
         
        #https://sourceforge.net/p/electricdss/discussion/861976/thread/c438f681/
        Rmat_final=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        Xmat_final=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]        
        Rmat_l=self.dssLines.Rmatrix #extract Rmatrix of line (in ohms/length; tuple format)
        Xmat_l=self.dssLines.Xmatrix #extract Xmatrix of line (in ohms/length; tuple format)        
        len_line=self.dssLines.Length #extract length of line  #Rmatrix and Xmatrix will be by default converted to same unit of length by DSS
        Rmat=[len_line*j for j in Rmat_l] #list format; in ohms
        Xmat=[len_line*k for k in Xmat_l] #list format; in ohms
        Num_phases=self.dssLines.Phases
        self.dssCircuit.SetActiveElement("Line."+name)
        node_seq=self.dssCktElement.NodeOrder
        node_seq=node_seq[0:Num_phases]        
        
        # assigning values to corresponding phases
        if Num_phases==3:
            Rmat_final[0][:]=Rmat[0:3]
            Rmat_final[1][:]=Rmat[3:6]
            Rmat_final[2][:]=Rmat[6:9]
           
            Xmat_final[0][:]=Xmat[0:3]
            Xmat_final[1][:]=Xmat[3:6]
            Xmat_final[2][:]=Xmat[6:9]
           
        if Num_phases==2:
            (p,q)=node_seq
            Rmat_final[p-1][p-1]=Rmat[0]
            Rmat_final[p-1][q-1]=Rmat[1]
            Rmat_final[q-1][p-1]=Rmat[2]
            Rmat_final[q-1][q-1]=Rmat[3]
            
            Xmat_final[p-1][p-1]=Xmat[0]
            Xmat_final[p-1][q-1]=Xmat[1]
            Xmat_final[q-1][p-1]=Xmat[2]
            Xmat_final[q-1][q-1]=Xmat[3]
            
        if Num_phases==1:
            (p,)=node_seq
            Rmat_final[p-1][p-1]=Rmat[0]
            Xmat_final[p-1][p-1]=Xmat[0]
            
        Rmat_final=np.array(Rmat_final)
        Xmat_final=np.array(Xmat_final)
        
        Rs=Rmat_final[np.eye(Rmat_final.shape[0],dtype=bool)] # diagonal
        Xs=Xmat_final[np.eye(Xmat_final.shape[0],dtype=bool)]
        Rm=Rmat_final[~np.eye(Rmat_final.shape[0],dtype=bool)] #off diagonal
        Xm=Xmat_final[~np.eye(Xmat_final.shape[0],dtype=bool)]
        
        R_S=0
        for x in Rs:
            R_S=R_S+x
        R_S=R_S/3 #averaging
        
        R_M=0
        for x in Rm:
            R_M=R_M+x
        R_M=R_M/6
        R=R_S-R_M #averaging
        
        X_S=0
        for x in Xs:
            X_S=X_S+x
        X_S=X_S/3
        
        X_M=0
        for x in Xm:
            X_M=X_M+x
        X_M=X_M/6
        X=X_S-X_M                 
        # Rpos_l=self.dssLines.R1
        # Xpos_l=self.dssLines.X1
        # R=Rpos_l*len_line
        # X=Xpos_l*len_line        
        return(R,X)
 
    
    def get_Transformer_RX(self,name): #to get the transformer resistance  and reactance and length
        self.dssTransformers.Name=name #activating the transformer
        X_per=self.dssTransformers.Xhl #extract the leakage reactance (in percentage)
        R_per=self.dssTransformers.R #extract the winding resistance (in percentage)
        kva_trfr=self.dssTransformers.kva #extract base kva of transformer
        kv_trfr=self.dssTransformers.kv #extract base kv of transformer
        Z_trfr=((kv_trfr**2)*1000)/kva_trfr
        R=R_per*Z_trfr
        X=X_per*Z_trfr
        #In transformer with only two winding only Xhl is extracted then in the Rmatrix diagonal elements
        #are given value R and Xmatrix diagonal elements are given value X. Off diagonal elements are 0.
        #The self impedance(Zs=R+jX) and mutual impedance(Zm=0)
        #since Z1(positive sequence=avg(Zs)-avg(Zm)); positive sequence resistance and reactance are R and X !!!)
        return(R,X)


# Bus class contains the bus object details
class Bus:
    def __init__(self,DSSobj,bus_name):
        """
        Inputs:
            circuit object
            bus name
        Contains:
            V -  voltage at bus nodes
            distance - distance from the energymeter
            x  - bus x location(co-ordinate)
            y -  bus y location(co-ordinate)
        """ 
        V=np.zeros(3,)        
        # n is set to the number of buses in the circuit
        DSSobj.dssCircuit.SetActiveBus(bus_name)
        x=DSSobj.dssBus.x
        y=DSSobj.dssBus.y
        distance=DSSobj.dssBus.Distance
        v=np.array(DSSobj.dssBus.puVmagAngle)
        nodes=np.array(DSSobj.dssBus.Nodes)
        if nodes.size > 3: nodes = nodes[0:3] # we only require the 3 nodes of each bus
        if v.size !=1:
           cidx = 2 * np.array(range(0, min(int(v.size/ 2), 3)))
           V[nodes-1] = v[cidx] #Bus voltage magnitude in per unit at the nodes
            
        self.V = V     
        self.x = x
        self.y = y
        self.distance = distance
        
                              
    
# Branch class contains the branch object details
class Branch:  # to extract properties of branch
    def __init__(self, DSSobj, branch_fullname):
        """
        Inputs:
            circuit object
            branch name
        Contains:                
            bus_fr - from bus name
            bus_to - to bus name
            I -  branch current vector(complex)            
            nphases - number of phases
            
        """
        DSSobj.dssCircuit.SetActiveElement(branch_fullname)
        V_fr = np.zeros(3, dtype=complex)
        V_to = np.zeros(3, dtype=complex)
        
        bus_connections=DSSobj.dssCktElement.BusNames
        bus1= bus_connections[0]
        bus2= bus_connections[1]        
        i=np.array(DSSobj.dssCktElement.CurrentsMagAng)
        ctidx = 2 * np.array(range(0, min(int(i.size/ 4), 3)))
        I_mag = i[ctidx] #branch current in A
        I_ang=i[ctidx + 1] #angle in deg
        nphases=DSSobj.dssCktElement.NumPhases
        MaxCap=DSSobj.dssCktElement.EmergAmps
        #MaxCap=DSSobj.dssCktElement.NormalAmps
       # https://sourceforge.net/p/electricdss/discussion/861976/thread/8aa13830/
       # Problem is that Line.650632 already exceeds normal amps in Opendss=400 A and 
       # Normal Amps in Kerstings book =530 A. So I will consider EmergAmps=600 A
        I_avg=np.average(I_mag)
        self.bus_fr=bus1
        self.bus_to=bus2
        self.nphases=nphases
        self.Cap=I_avg
        self.MaxCap=MaxCap