from bef4llm.semantic_quality.similarity.language_similarity import natural_language_similarity as ns
from bef4llm.process_models.graph_representation.node_types import Task, Event
from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel

def create_euquivalence_mapping_for_nodes(graph1, graph2, sim, threshold=0.0):
    """
    Function creates a matching between the lists of nodes of two graphs with a given similarity method above a certain
    threshold. Returns matching as a dictionary including the nodes as keys and values and the value of the similarity.

    Parameters
    ----------
    graph1 : networkx.DiGraph
    graph2 : networkx.DiGraph
    sim : similarity method without parameters
        If None, equal label similarity will be used.
    threshold : float
                Value between 0 and 1.

    Returns
    -------
    dict
        Dict with node 1 as keys and a tuple of node 2 and the similarity value as the value.
        The mapping is only saved in the dict, if the similarity value is above a threshold

    Notes
    -----
    Matched nodes from both graphs are returned as keys and values.
    This mapping is injective, since this is a

    References
    ----------
    [1] R. Dijkman et al., "Similarity of business process models: Metrics and evaluation", Information Systems,
    vol. 36, pp. 498-516, 2011.

    """

    if sim is None:
        sim = ns.syntactic_similarity()

    matching = {}



    for node1 in graph1.nodes:
        type1 = graph1.nodes[node1]['type'] if "type" in graph1.nodes[node1] else None

        if not type1 or type1 not in Task.__members__ and type1 not in Event.__members__:
            continue

        for node2 in graph2.nodes:
            type2 = graph2.nodes[node2]['type'] if "type" in graph2.nodes[node2] else None

            if not type2 or type2 not in Task.__members__ and type2 not in Event.__members__:
                continue

            # node type are ignored since. "ts be a set of types of nodes that should be ignored." for node similarity
            node1_data = graph1.nodes[node1]
            node2_data = graph2.nodes[node2]
            sim_n1_n2 = sim(node1_data, node2_data)

            if sim_n1_n2 >= threshold:
                if node1 in matching:
                    if matching[node1][1] < sim_n1_n2:
                        matching[node1] = (node2, sim_n1_n2)
                        #matching[node2] = (node1, sim_n1_n2)

                if node2 in matching:
                    if matching[node2][1] < sim_n1_n2:
                        #matching[node1] = (node2, sim_n1_n2)
                        matching[node2] = (node1, sim_n1_n2)
                # ensure that every node is in the matching
                if node1 not in matching:
                    matching[node1] = (node2, sim_n1_n2)

                if node2 not in matching:
                    matching[node2] = (node1, sim_n1_n2)
    return matching

def create_euquivalence_mapping_for_pools(graph1, graph2, sim, threshold=0.0):
    """
    Function creates a matching between the lists of nodes of two graphs with a given similarity method above a certain
    threshold. Returns matching as a dictionary including the nodes as keys and values and the value of the similarity.

    Parameters
    ----------
    graph1 : CollaborationModel
    graph2 : CollaborationModel
    sim : similarity method without parameters
        If None, equal label similarity will be used.
    threshold : float
                Value between 0 and 1.

    Returns
    -------
    dict
        Dict with node 1 as keys and a tuple of node 2 and the similarity value as the value.
        The mapping is only saved in the dict, if the similarity value is above a threshold

    Notes
    -----
    Matched nodes from both graphs are returned as keys and values.
    This mapping is injective, since this is a

    References
    ----------
    [1] R. Dijkman et al., "Similarity of business process models: Metrics and evaluation", Information Systems,
    vol. 36, pp. 498-516, 2011.

    """
    def create_label_dicts_for_pool_lanes(graph):
        label_dict = dict()
        for pool in graph.pools:
            if "name" in graph.pools[pool]:
                label_dict[pool] = {"name": graph.pools[pool]["name"]}

        for lane in graph.lanes:
            if "name" in graph.lanes[lane]:
                label_dict[lane] = {"name": graph.lanes[lane]["name"]}

        return label_dict

    if sim is None:
        sim = ns.syntactic_similarity()

    matching = {}
    participant1 = create_label_dicts_for_pool_lanes(graph1)
    participant2 = create_label_dicts_for_pool_lanes(graph2)


    for p1 in participant1:
        for p2 in participant2:
            node1_data = participant1[p1]
            node2_data = participant2[p2]
            sim_n1_n2 = sim(node1_data, node2_data)

            if sim_n1_n2 >= threshold:

                if p1 in matching:
                    if matching[p1][1] < sim_n1_n2 :
                        matching[p1] = (p2, sim_n1_n2)
                        matching[p2] = (p1, sim_n1_n2)

                elif p2 in matching:
                    if matching[p2][1] < sim_n1_n2:
                        matching[p1] = (p2, sim_n1_n2)
                        matching[p2] = (p1, sim_n1_n2)

                else:
                    matching[p1] = (p2, sim_n1_n2)
                    matching[p2] = (p1, sim_n1_n2)
    return matching
