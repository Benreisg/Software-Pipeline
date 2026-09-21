from bef4llm.process_models.graph_representation.node_types import Event, Gateway, Task, Flows
from bef4llm.semantic_quality.similarity.similarity_utils import create_euquivalence_mapping_for_nodes

import copy


def graph_edit_distance(matching=None, sim=None, threshold=1.00, events=True, tasks=True, gateways=False):
    """
    Function returns calculate_graph_edit_distance function which just requires two graphs as arguments, with the rest
    of the parameters set.

    Parameters
    ----------
    matching : dict, optional
        Any previously performed matching can be added here.
        Default is None.
        If None, matching is created.
    sim : similarity method, optional
        Default is None.
        If None, the equal label similarity is used.
    threshold : float, optional
        Value between 0 and 1.
        Default is 1.0.
    events : bool, optional
        Default is True.
    tasks : bool, optional
        Default is True.
    gateways : bool, optional
        Default is False.
        The boolean values events, tasks and gateways allow the user to include or exclude these from the further
        measuring of similarities

    Returns
    -------
    function

    References
    ----------
    [1] R. Dijkman et al., "Similarity of business process models: Metrics and evaluation", Information Systems,
    vol. 36, pp. 498-516, 2011.
    """

    matching_new = matching
    sim_new = sim

    def calculate_graph_edit_distance(graph1, graph2):
        """
        Function computes the graph edit distance, normalized to the number of nodes in the two graphs in total.

        Parameters
        ----------
        graph1 : networkx.DiGraph
        graph2 : networkx.DiGraph

        Returns
        -------
        float
            Normalized value for the distance between 0 and 1.

        References
        ----------
        [1] R. Dijkman et al., "Similarity of business process models: Metrics and evaluation", Information Systems,
        vol. 36, pp. 498-516, 2011.


        """

        alt_graph1, graph1_nodes, graph1_edges = __compute_nodes_edges(graph1, events, tasks, gateways)
        alt_graph2, graph2_nodes, graph2_edges = __compute_nodes_edges(graph2, events, tasks, gateways)

        nonlocal matching_new
        nonlocal sim_new
        if matching_new is None:
            matching_new = create_euquivalence_mapping_for_nodes(alt_graph1, alt_graph2, sim_new, threshold)

        sub_n, sn = __compute_sn(graph1_nodes, graph2_nodes, matching_new)
        sub_e, se = __compute_se(alt_graph1, alt_graph2, graph1_edges, graph2_edges, matching_new)

        snv = len(sn) / (len(list(graph1_nodes)) + len(list(graph2_nodes)))
        if not graph1_edges and not graph2_edges:
            sev = 0
        else:
            sev = len(se) / (len(list(graph1_edges)) + len(list(graph2_edges)))

        sbv = 0

        for matched_node in matching_new:
            sim = matching_new[matched_node][1]
            sbv = sbv + (1 - sim)

        if sbv != 0:
            sbv = sbv / (graph1.number_of_nodes() + graph2.number_of_nodes() - len(sn))

        elif sbv == 0 and (graph1.number_of_nodes() + graph2.number_of_nodes() - len(sn)) == 0:
            sbv = 1

        distance = (snv + sev + sbv) / 3
        return distance

    return calculate_graph_edit_distance


def __compute_sn(graph1_nodes, graph2_nodes, matching):
    """
    Function splits a list of all nodes from two graphs into a list containing the matched nodes and a list containing
    the nodes that were not matched.

    Parameters
    ----------
    graph1_nodes : list of networkx.DiGraph.nodes or networkx.DiGraph
    graph2_nodes : list of networkx.DiGraph.nodes or networkx.DiGraph
    matching : dict

    Returns
    -------
    list, list
        list of matched nodes, list of unmatched nodes

    """
    all_nodes = list(copy.deepcopy(graph1_nodes))
    all_nodes.extend(graph2_nodes)

    node_ids = [x[0] for x in all_nodes]
    sub = [node_id for node_id in node_ids if node_id in matching]

    sn = [node_id for node_id in node_ids if node_id not in sub]

    return sub, sn


def __compute_se(graph1, graph2, graph1_edges, graph2_edges, matching):
    """
    Function splits a list of all edges from two graphs into a list containing the matched edges and a list containing
    the edges that were not matched.

    Parameters
    ----------
    graph1_edges : list of networkx.DiGraph.edges
    graph2_edges : list of networkx.DiGraph.edges
    matching : dict

    Returns
    -------
    list, list
        list of matched edges, list of unmatched edges
    """

    sub = []

    for edge1 in graph1_edges:

        for edge2 in graph2_edges:

            if edge2 in sub:
                continue

            edge1_1 = edge1[0]
            edge1_2 = edge1[1]
            edge2_1 = edge2[0]
            edge2_2 = edge2[1]

            if edge1_1 in matching and edge2_1 == matching[edge1_1][0] \
                    and edge1_2 in matching and edge2_2 == matching[edge1_2][0]\
                    and graph1.get_edge_data(edge1_1, edge1_2)["type"] == graph2.get_edge_data(edge2_1, edge2_2)["type"]:
                sub.append(edge1)
                sub.append(edge2)
                break

    all_edges = list(copy.deepcopy(graph1_edges))
    all_edges.extend(graph2_edges)

    se = [x for x in all_edges if x not in sub]

    return sub, se


def __compute_nodes_edges(graph, events, tasks, gateways):
    """
    Function that calculates a new graph as well as the nodes and edges, including and excluding specified node types.

    Parameters
    ----------
    graph : networkx.DiGraph
    events : bool
    tasks : bool
    gateways : bool

    Notes
    -----
    Node types with the value True are included in the graph and the list, new edges are drawn between the successors
    and predecessors of removed nodes.

    Returns
    -------
    graph : networkx.DiGraph
    list : list of networkx.DiGraph.nodes
    list : list of networkx.DiGraph.edges

    """
    # if every node type is included, the graph is returned unchanged.
    if events and tasks and gateways:
        nodes = list(graph.nodes(data=True))
        edges = list(graph.edges)
        return graph, nodes, edges

    else:

        alternative_graph = copy.deepcopy(graph)

        all_nodes = list(alternative_graph.nodes(data=True))

        # for all nodes of the deepcopy it is checked whether they belong to a type that should be removed.
        # if so, the node is removed and new edges are drawn between the predecessors and successors.
        for node in all_nodes:

            typ = node[1]['type'] if "type" in node[1] else None

            if typ and typ in Event.__members__ and not events \
                    or typ in Task.__members__ and not tasks \
                    or typ in Gateway.__members__ and not gateways:

                predecessors = list(alternative_graph.predecessors(node[0]))
                successors = list(alternative_graph.successors(node[0]))

                alternative_graph.remove_node(node[0])

                for predecessor in predecessors:
                    for successor in successors:
                        alternative_graph.add_edge(predecessor, successor, type=Flows.sequenceFlow)

        return alternative_graph, list(alternative_graph.nodes(data=True)), list(alternative_graph.edges)


def common_percentage_similarity(matching=None, sim=None, threshold=1.00, nodes=True, edges=False, events=True, tasks=True,
                                 gateways=False):
    """
    Function returns the function calculate_common_percentage which just requires two graphs as inputs,
    with other parameters set.

    Parameters
    ----------
    matching : dict, optional
        If None, a matching is created by create_matching
    sim : similarity method, optional
         If None, equal label similarity will be used.
    threshold : float, optional
        Default is 1.00
    nodes : bool
        Default True
    edges : bool
        Default False
        These booleans let the user define whether just nodes, just edges or both should be compared.
    events : bool
    tasks : bool
    gateways : bool
        These booleans define which node types will be taken into account.

    Returns
    -------
    function
    """

    def calculate_common_percentage_similarity(graph1, graph2):
        """
        Function calculates the percentage of matched nodes of defined types and/or edges of two graphs and returns
        a similarity value between 0 and 1.

        Parameters
        ----------
        graph1 : networkx.DiGraph
        graph2 : networkx.DiGraph

        Returns
        -------
        float
            Value between 0 and 1

        See Also
        --------
        create_matching
        compute_sn
        compute_se

        """

        alt_graph1, graph1_nodes, graph1_edges = __compute_nodes_edges(graph1, events, tasks,
                                                                       gateways)
        alt_graph2, graph2_nodes, graph2_edges = __compute_nodes_edges(graph2, events, tasks,
                                                                       gateways)

        nonlocal matching
        if matching is None:
            matching = create_euquivalence_mapping_for_nodes(alt_graph1, alt_graph2, sim, threshold)

        if nodes:
            sub_n, sn = __compute_sn(graph1_nodes, graph2_nodes, matching)
            common_nodes = len(sub_n) / (alt_graph1.number_of_nodes() + alt_graph2.number_of_nodes())

            if edges:
                sub_e, se = __compute_se(alt_graph1, alt_graph2, graph1_edges, graph2_edges, matching)

                common_edges = len(sub_e) / (alt_graph1.number_of_edges() + alt_graph2.number_of_edges())
                return (common_nodes + common_edges) / 2
            else:
                return common_nodes


        elif edges:
            sub_e, se = __compute_se(alt_graph1, alt_graph2, graph1_edges, graph2_edges, matching)
            common_edges = len(sub_e) / (alt_graph1.number_of_edges() + alt_graph2.number_of_edges())
            return common_edges
        return 0

    return calculate_common_percentage_similarity






