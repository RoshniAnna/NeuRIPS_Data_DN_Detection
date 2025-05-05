"""
OpenDSS engine setup with Circuit, Bus, and Branch classes.
# circuit set up changes with modifications to base circuit - here it is PV
"""

import opendssdirect as dss
import numpy as np
import math
import networkx as nx

class DSSCircuit:
    """Initializes OpenDSS circuit and provides basic operations."""

    def __init__(self, filename: str):
        self.filename = filename
        self.dss = dss

    def compile_ckt(self):
        """Compiles the DSS circuit."""
        self.dss.Basic.ClearAll()
        self.dss.Text.Command(f"compile [{self.filename}]")

    def get_base_values(self):
        """Returns base values (KVA, KV, Zbase, Ibase) for circuit."""
        self.dss.Transformers.First()
        KVA_base = self.dss.Transformers.kVA()
        KV_base = self.dss.Transformers.kV()
        I_base = KVA_base / (math.sqrt(3) * KV_base)
        Z_base = (KV_base ** 2 * 1000) / KVA_base
        return KVA_base, KV_base, Z_base, I_base


class Bus:
    """Extracts voltage magnitude and node connection for a bus."""

    def __init__(self, DSSCktobj: DSSCircuit, bus_name: str):
        self.dss = DSSCktobj.dss
        self.bus_name = bus_name
        self._bus_info()

    def _bus_info(self):
        self.dss.Circuit.SetActiveBus(self.bus_name)
        voltages = self.dss.Bus.puVmagAngle()
        nodes = np.array(self.dss.Bus.Nodes())

        Vmag = np.zeros(3)  # Assuming 3-phase system
        if len(nodes) > 0 and len(voltages) > 0:
            for idx, node in enumerate(nodes):
                Vmag[node - 1] = voltages[idx * 2]

        self.Vmag = Vmag
        self.nodes = nodes


class Branch:
    """Extracts impedance and current capacity for a branch."""

    def __init__(self, DSSCktobj: DSSCircuit, branch_fullname: str):
        self.dss = DSSCktobj.dss
        self.branch_fullname = branch_fullname
        self._branch_info()

    def _branch_info(self):
        self.dss.Circuit.SetActiveElement(self.branch_fullname)
        bus_connections = self.dss.CktElement.BusNames()
        self.bus_fr, self.bus_to = bus_connections[0], bus_connections[1]
        self.nphases = self.dss.CktElement.NumPhases()
        self.MaxCap = self.dss.CktElement.EmergAmps()
        power_flow = np.array(self.dss.CktElement.Powers())
        currents = np.array(self.dss.CktElement.CurrentsMagAng())
        ctidx = 2 * np.array(range(0, min(int(currents.size/ 4), 3)))
        I_mag = currents[ctidx]  # pick magnitude part only for up to 3 phases
        self.flow = power_flow[ctidx]
        self.Cap = np.sum(I_mag)
        
        branch_type = self.branch_fullname.split('.')[0].lower()
        if branch_type == 'line':
            self.R, self.X = self._extract_line_impedance()
        elif branch_type == 'transformer':
            self.R, self.X = self._extract_transformer_impedance()
        # else:
        #     raise ValueError(f"Unsupported branch type: {branch_type}")

    def _extract_line_impedance(self):
        self.dss.Lines.Name(self.branch_fullname.split('.')[1])
        Rmat = np.array(self.dss.Lines.RMatrix()) * self.dss.Lines.Length()
        Xmat = np.array(self.dss.Lines.XMatrix()) * self.dss.Lines.Length()
        node_seq = self.dss.CktElement.NodeOrder()[:self.nphases]

        Rmat_final = np.zeros((3, 3))
        Xmat_final = np.zeros((3, 3))

        if self.nphases == 3:
            Rmat_final = Rmat.reshape((3, 3))
            Xmat_final = Xmat.reshape((3, 3))
        elif self.nphases == 2:
            (p, q) = node_seq
            indices = [(p-1, p-1), (p-1, q-1), (q-1, p-1), (q-1, q-1)]
            for idx, val in zip(indices, Rmat):
                Rmat_final[idx] = val
            for idx, val in zip(indices, Xmat):
                Xmat_final[idx] = val
        elif self.nphases == 1:
            (p,) = node_seq
            Rmat_final[p-1, p-1] = Rmat[0]
            Xmat_final[p-1, p-1] = Xmat[0]

        Rs = np.diag(Rmat_final)
        Rm = Rmat_final[~np.eye(3, dtype=bool)]
        Xs = np.diag(Xmat_final)
        Xm = Xmat_final[~np.eye(3, dtype=bool)]

        R_eq = np.mean(Rs) - np.mean(Rm)
        X_eq = np.mean(Xs) - np.mean(Xm)

        return R_eq, X_eq

    def _extract_transformer_impedance(self):
        self.dss.Transformers.Name(self.branch_fullname.split('.')[1])
        X_per = self.dss.Transformers.Xhl()
        R_per = self.dss.Transformers.R()
        kva_trfr = self.dss.Transformers.kVA()
        kv_trfr = self.dss.Transformers.kV()

        Z_trfr = (kv_trfr ** 2) * 1000 / kva_trfr
        R = (R_per/100) * Z_trfr 
        X = (X_per/100) * Z_trfr
        return R, X


def CircuitSetup(DSSfile: str, PV_generators) -> DSSCircuit:
    """Compiles and sets up a DSS circuit object."""
    dss_obj = DSSCircuit(DSSfile)
    dss_obj.compile_ckt()
    dss_obj.dss.Text.Command("Set Maxiterations=500")
    dss_obj.dss.Text.Command("Set Maxcontroliter=500")

    dss_obj.dss.Basic.AllowForms(0)
    dss_obj.dss.Solution.Solve()
    
    dss_obj.dss.Text.Command("redirect PVSystem.dss") # PV system properties
    for gen in PV_generators:
        dss_obj.dss.Text.Command(f"New PVSystem.PV{str(gen['no'])} bus1={gen['bus']}{gen['phaseconn']} phases={str(gen['numphase'])} kv={str(gen['kV'])} kVA={str(gen['size']*1.2)} pf=0.95 Pmpp={str(gen['size'])} irrad=1.0 temperature=25 %cutin=0.01 %cutout=0.01 effcurve=Myeff P-TCurve=MyPvsT")
                
    return dss_obj