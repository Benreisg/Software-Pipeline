import networkx as nx

from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel
from bef4llm.process_models.graph_representation.node_types import Gateway, Participant, Flows
import bef4llm.process_models.graph_representation.node_types as nt


def start_events(graph):
    """
    Returns the number of start events

    Parameters
    ----------
    graph: networkx.DiGraph

    Returns
    -------
    int
        number of start_events

    """
    nodelist = [node[1] for node in graph.nodes(data='type')]

    return nodelist.count('startEvent')


def end_events(graph):
    """
    Returns the number of end events

    Parameters
    ----------
    graph: networkx.DiGraph

    Returns
    -------
    int
        number of end_events

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction, and
    guidelines for correctness (Vol. 6). Springer Science & Business Media.

    """
    nodelist = [node[1] for node in graph.nodes(data='type')]

    return nodelist.count('endEvent')


def get_number_gateway_types(graph):
    """
    computes the number of gateways per gateway type, divided by split and join gateways
    Parameters
    ----------
    graph: networkx.DiGraph
        process model

    Returns
    -------
        gateway_types: dict
            dict with gateway type as key and number of gateways as value
    """
    gateway_types = {
        Gateway.exclusiveGateway.value: {"split": 0, "join": 0},
        Gateway.parallelGateway.value: {"split": 0, "join": 0},
        Gateway.inclusiveGateway.value: {"split": 0, "join": 0},
        Gateway.eventBasedGateway.value: {"split": 0, "join": 0},
    }

    for node in graph.nodes:
        # node withput attributes
        if "incoming" not in graph.nodes[node] or "outgoing" not in graph.nodes[node]:
            continue
        in_degree = len(graph.nodes[node]['incoming'])
        out_degree = len(graph.nodes[node]['outgoing'])
        node_type = graph.nodes[node]['type'] if "type" in graph.nodes[node] else None

        if node_type and any(node_type == item for item in gateway_types) and out_degree > 1:
            gateway_types[graph.nodes[node]["type"]]['split'] += 1

        elif node_type and any(node_type == item for item in gateway_types) and in_degree > 1:
            gateway_types[graph.nodes[node]["type"]]['join'] += 1

    return gateway_types


def get_gateways_by_types(graph):
    """
    returns gateways ordered by type and divided into split and join gateways
    Parameters
    ----------
    graph: networkx.DiGraph
        process model

    Returns
    -------
        gateway_types: dict
            dict with gateway type as key and list of gateways as value
    """
    gateway_types = {
        Gateway.exclusiveGateway.value: {"split": [], "join": []},
        Gateway.parallelGateway.value: {"split": [], "join": []},
        Gateway.inclusiveGateway.value: {"split": [], "join": []},
        Gateway.eventBasedGateway.value: {"split": [], "join": []},
    }

    for node in graph.nodes:
        # node withput attributes
        if "incoming" not in graph.nodes[node] or "outgoing" not in graph.nodes[node]:
            continue

        in_degree = len(graph.nodes[node]['incoming'])
        out_degree = len(graph.nodes[node]['outgoing'])
        node_type = graph.nodes[node]['type']

        if any(node_type == item for item in gateway_types) and out_degree > 1:
            gateway_types[node_type]['split'].append(node)

        elif any(node_type == item for item in gateway_types) and in_degree > 1:
            gateway_types[node_type]['join'].append(node)

    return gateway_types


def get_node_by_id(node_id, graph):
    """
    Returns the model including the node

    Parameters
    ----------
    node_id : string
        node id
    graph : networkx.DiGraph

    Returns
    -------
    node
        the node object, including the type, given the id of the node and the graph its taken from
    """
    for n in graph.nodes(data='type'):

        if n[0] == node_id:
            return n

    return None


def get_degree_node(node, graph):
    """
    Returns the degree of the node

    Parameters
    ----------
    node : string
        id in nx graph
    graph : networkx.DiGraph
        proces model

    Returns
    -------
    degree : int
    """
    return get_indegree_node(node, graph) + get_outdegree_node(node, graph)


def get_indegree_node(node, graph):
    """
    Returns the indegree of the node

    Parameters
    ----------
    node : string
        id in nx graph
    graph : networkx.DiGraph
        proces model

    Returns
    -------
    indegree : int
    """
    return len([(u, v) for u, v in graph.in_edges(node) if graph.get_edge_data(u, v)["type"] == "sequenceFlow"])


def get_outdegree_node(node, graph):
    """
    Returns the outdegree of the node

    Parameters
    ----------
    node : string
        id in nx graph
    graph : networkx.DiGraph
        proces model

    Returns
    -------
    outdegree : int
    """
    return len([(u, v) for u, v in graph.out_edges(node) if graph.get_edge_data(u, v)["type"] == "sequenceFlow"])


def is_reachable_with_sequence_flow(model, start, target):
    """
    Tests wheather a target node is reachable with a sequence flow from a given start node

    Parameters
    ----------
    start : str
        id in nx graph
    target : str
        id in nx graph

    Returns
    -------
    is_reachable : bool
    """

    if model.subgraph_sequence_flow == None:
        model.init_subgraph_sequence_flows()

    if start not in model.subgraph_sequence_flow.nodes or target not in model.subgraph_sequence_flow.nodes:
        return False

    return nx.has_path(model.subgraph_sequence_flow, start, target)

def get_predecessor_by_type(graph: nx.DiGraph, node, types=None, visited_nodes=None):
    """
    Returns the next predecessors of a node with the given type

    Parameters
    ----------
    graph : networkx.DiGraph
    node : string
    types : list of strings
        type values from Enums

    Returns:
    -------
    List of all predecessors
    """

    if not visited_nodes:
        visited_nodes = []

    if types == [] or types == None:
        types = [t.name for t in nt.Task] + [g.name for g in nt.Gateway] + [e.name for e in nt.Event]

    if node in visited_nodes:
        return []

    visited_nodes.append(node)

    pre = []
    for pre_node in graph.predecessors(node):

        if ("type" not in graph.nodes[pre_node] or
                not graph.nodes[pre_node]['type'] in types or graph[pre_node][node]['type'] == Flows.sequenceFlow):
            pre = pre + get_predecessor_by_type(graph, pre_node, types, visited_nodes)
        else:
            pre.append(pre_node)
    return pre


def get_successor_by_type(graph, node, types=None, visited_nodes=None):
    """
    Returns the next successor of a node with the given type

    Parameters
    ----------
    graph : networkx.DiGraph
    node : string
    types : list of strings
        type values from Enums

    Returns:
    -------
    List of all successors
    """
    if not visited_nodes:
        visited_nodes = []

    if types == [] or types == None:
        types = [t.name for t in nt.Task] + [g.name for g in nt.Gateway] + [e.name for e in nt.Event]

    if node in visited_nodes:
        return []

    visited_nodes.append(node)

    suc = []
    for s in graph.successors(node):
        if ("type" not in graph.nodes[s] or
                graph.nodes[s]['type'] not in types or graph[node][s]['type'] == Flows.sequenceFlow):
            suc = suc + get_successor_by_type(graph, s, types, visited_nodes)
        else:
            suc.append(s)
    return suc

def get_number_of_nodes_by_type(graph: nx.DiGraph, types:list):
    """
    Returns the number of nodes of a graph sorted by the given types

    Parameters
    ----------
    graph : networkx.DiGraph
        process model
    types : list
        list of types that for which the number of nodes needs to be returned

    Returns
    -------
    number_of_nodes : dict
    """
    number_of_nodes = 0
    process_nodes = [node for node in graph.nodes if "type" in graph.nodes[node]]
    for type in types:
        number_of_nodes += len([n for n in process_nodes if graph.nodes[n]["type"] == type])

    return number_of_nodes

def get_subprocess_node(graph, node):
    """
    Returns the subprocess of a node, if node is a subprocess element
    Parameters
    ----------
    graph : networkx.DiGraph
        process model
    node : string
        id in nx graph

    Returns
    --------
    subprocess_node : dict
        nx node
    """
    for subprocess_id in graph.subprocesses:
        subprocess = graph.subprocesses[subprocess_id]
        if node in subprocess.process_graph.nodes:
            return subprocess.process_graph.nodes[node]