
"""
The graph for the base DSS cicuit is built
"""

from DSSCircuit_Interface import *

def build_graph(DSSCktobj: DSSCircuit):
    """
    Builds a graph (G_original) from the DSS circuit object also using Branch class.

    Args:
        DSSCktobj (DSSCircuit): Compiled DSS circuit object.

    Returns:
        G_original (nx.Graph): NetworkX undirected graph.
        edges_dictlist (list of dict): List of edges with properties.
    """
    G_original = nx.Graph()

    PDElements = DSSCktobj.dss.PDElements.AllNames()

    for elem_fullname in PDElements:
        branch_obj = Branch(DSSCktobj, elem_fullname)
        
        # Check if it's line or transformer
        device_type = elem_fullname.split('.')[0].lower()
        if device_type not in ['line', 'transformer']:
            continue  # Skip non-line/non-transformer PDElements
        
        # Basic branch properties
        sr_node = branch_obj.bus_fr.split('.')[0]
        tar_node = branch_obj.bus_to.split('.')[0]
        label = elem_fullname.split('.')[1]
        numphases = branch_obj.nphases
        Inorm = branch_obj.Cap
        MaxCap = branch_obj.MaxCap
        R = branch_obj.R
        X = branch_obj.X

        # Edge dictionary
        edge_info = {
            "Label": label,
            "Device": device_type,
            "Resistance": R,
            "Reactance": X,
            "Phases": numphases,
            "MaxCap": MaxCap,
            "NominalCap": Inorm
        }

        # Add edge to graph
        G_original.add_edge(sr_node, tar_node, **edge_info)

    # Add Bus nodes (with load information)
    buses = DSSCktobj.dss.Circuit.AllBusNames()

    # Fetch all loads manually
    loads_info = []
    l_id = DSSCktobj.dss.Loads.First()
    while l_id>0:
            DSSCktobj.dss.Circuit.SetActiveElement(f"Load.{DSSCktobj.dss.Loads.Name()}")
            loads_info.append({
                "Bus": DSSCktobj.dss.CktElement.BusNames()[0].split('.')[0],
                "Pload": DSSCktobj.dss.Loads.kW(),
                "Qload": DSSCktobj.dss.Loads.kvar()
            })
            l_id = DSSCktobj.dss.Loads.Next()

    for b in buses:
        P_sum = sum(load["Pload"] for load in loads_info if load["Bus"] == b)
        Q_sum = sum(load["Qload"] for load in loads_info if load["Bus"] == b)

        if b not in G_original.nodes:
            G_original.add_node(b)

        G_original.nodes[b]["Component"] = "Bus"
        G_original.nodes[b]["ActiveLoad"] = P_sum
        G_original.nodes[b]["ReactiveLoad"] = Q_sum

    return G_original