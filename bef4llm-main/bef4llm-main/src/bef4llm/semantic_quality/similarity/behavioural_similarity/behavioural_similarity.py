import math

from bef4llm.process_models.graph_representation.node_types import Event, Task, Flows
from bef4llm.semantic_quality.similarity.similarity_utils import create_euquivalence_mapping_for_nodes
from bef4llm.process_models.collaboration_model_utils import get_predecessor_by_type, get_successor_by_type
from bef4llm.semantic_quality.similarity.behavioural_similarity.trace_simulation import TraceExtractor
from bef4llm.semantic_quality.similarity.behavioural_similarity.behavioural_profiles import behavioural_profiles

import copy
import networkx as nx
import numpy as np


def causal_footprint_similarity(matching=None, sim=None, threshold=0.5):
    """
    Function returns the function calculate_causal_profile_similarity which just requires two graphs as inputs,
    with other parameters set.

    Parameters
    ----------
    matching : dict, optional
        If None, a matching is created by create_matching
    sim : similarity method, optional
         If None, equal label similarity will be used.
    threshold : float, optional
        Default is 0.5

    Returns
    -------
    function
    """

    def causal_footprint(graph):
        """
        Function that computes the causal footprints for a graph
        The look ahead and look back links are saved by type -> message flow is a different behaviour than sequence flow

        Parameters
        ----------
        graph : networkx graph

        Result
        ------
        look_ahead_links: dict
        look_back_links: dict
        """
        look_ahead_links = dict()
        look_back_links = dict()

        for node in graph.nodes:
            node_type = graph.nodes[node]['type'] if "type" in graph.nodes[node] else None
            if node_type and node_type in Task.__members__ or node_type in Event.__members__:
                pre = get_predecessor_by_type(graph, node, list(Task.__members__) + list(Event.__members__))
                suc = get_successor_by_type(graph, node, list(Task.__members__) + list(Event.__members__))
                #type = graph[node]['type'] if "type" in graph[node] else None

                if pre != []:
                    look_back_links[node] = {Flows.sequenceFlow.value: [], Flows.messageFlow.value: []}
                    for p in pre:
                        type = graph.edges[p, node]['type'] if [p, node] in graph.edges and "type" in graph.edges[p, node] else None
                        if type:
                            look_back_links[node][type].append(p)
                if suc != []:
                    look_ahead_links[node] = {Flows.sequenceFlow.value: [], Flows.messageFlow.value: []}
                    for s in suc:
                        type = graph.edges[node, s]['type'] if [node, s] in graph.edges and "type" in graph.edges[node, s] else None
                        if type:
                            look_ahead_links[node][type].append(s)
                    #look_ahead_links[node] = suc
        return look_ahead_links, look_back_links

    def index_vectors(not_matched_nodes,
                      look_ahead_links1, look_ahead_links2, look_back_links1, look_back_links2,
                      matching, graph1, graph2):
        """
        Function that computes the index vectors for both graphs

        Parameters
        ----------
        matching : dict, optional
            If None, a matching is created by create_matching
        look_ahead_links1 : dict, optional
            Look ahead links for graph 1
        look_ahead_links2 : dict, optional
            Look ahead links for graph 1
        look_back_links1: list
            Look back links for graph 1
        look_back_links2: list
           Look back links for graph 2
        graph1 : networkx graph
        graph2 : networkx graph

        Returns
        -------
        vector1: list
            Index vector for graph 1
        vector2: list
            Index vector for graph 2
        """
        vector1 = list()
        vector2 = list()

        for node in matching:
            vector1.append(matching[node][1])
            vector2.append(matching[node][1])

        for not_matched_node in not_matched_nodes:
            vector1.append(0)
            vector2.append(0)

        vector1, vector2 = add_indexes_to_vector(look_ahead_links1, look_ahead_links2, vector1, vector2)
        vector1, vector2 = add_indexes_to_vector(look_back_links1, look_back_links2, vector1, vector2)
        vector2, vector1 = add_indexes_to_vector(look_ahead_links2, look_ahead_links1, vector2, vector1)
        vector2, vector1 = add_indexes_to_vector(look_back_links2, look_back_links1, vector2, vector1)

        return np.array(vector1), np.array(vector2)

    def add_indexes_to_vector(look_links1, look_links2, vector1, vector2):
        for node in look_links1:
            for flow_type in look_links1[node]:
                if node in matching:
                    matched_node = matching[node][0]
                    vector1.append((matching[node][1]) / (2 ** len(look_links1[node][flow_type])))
                    if matched_node in look_links2 and flow_type in look_links2[matched_node]:
                        vector2.append((matching[matched_node][1]) / (2 ** len(look_links2[matched_node][flow_type])))
                    else:
                        vector2.append(0)
                else:
                    vector1.append(0)
                    vector2.append(0)

        return vector1, vector2



    def calculate_causal_footprint_similarity(graph1: nx.DiGraph, graph2: nx.DiGraph):
        """
        Function that calculates the similarity based on the causal profile of the two graphs according to [1]

        Parameters
        ----------
        graph1 : networkx.DiGraph
        graph2 : networkx.DiGraph

        Returns
        -------
        similarity_value : float
            cosine of the angle between the index vectors of the two graphs based on the causal footprints

        References
        ----------
        [1] R. Dijkman et al., "Similarity of business process models: Metrics and evaluation", Information Systems,
        vol. 36, pp. 498-516, 2011.
        """
        look_ahead1, look_back1 = causal_footprint(graph1)
        look_ahead2, look_back2 = causal_footprint(graph2)

        nonlocal matching
        if matching is None:
            matching = create_euquivalence_mapping_for_nodes(graph1, graph2, sim, threshold)

        not_matched = not_matched_nodes(graph1, graph2, matching)
        index_vector1, index_vector2 = index_vectors(not_matched,
                                                     look_ahead1, look_ahead2,
                                                     look_back1, look_back2,
                                                     matching, graph1, graph2)

        cross_product = 0
        dp1 = 0
        dp2 = 0
        for i in range(len(index_vector1)):
            cross_product += index_vector1[i] * index_vector2[i]
            dp1 += (index_vector1[i] ** 2)
            dp2 += (index_vector2[i] ** 2)

        dot_product = math.sqrt(dp1) * math.sqrt(dp2)

        if dot_product == 0:
            simcf = 0
        else:
            simcf = cross_product / dot_product

        if simcf < 0:
            return 0

        return simcf


    def not_matched_nodes(graph1, graph2, matching):
        """
        Function that find all nodes from graph 1 that do not have a match in graph 2 and reversed

        Parameters
        ----------
        graph1 : networkx.DiGraph
        graph2 : networkx.DiGraph
        matching: dict
            matches nodes from graph 1 and graph 2

        Returns
        -------
        not_matched_nodes : list
            list of nodes from graph 1 that do not have a match in graph 2 and reversed
        """
        not_matched = set()

        for node in graph1.nodes:
            node_type = graph1.nodes[node]['type'] if "type" in graph1.nodes[node] else None
            if node_type and node_type in Task.__members__ or node_type in Event.__members__:
                if node not in matching:
                    not_matched.add(node)

        for node in graph2.nodes:
            node_type = graph2.nodes[node]['type'] if "type" in graph2.nodes[node] else None
            if node_type and node_type in Task.__members__ or node_type in Event.__members__:
                if node not in matching:
                    not_matched.add(node)

        return not_matched




    return calculate_causal_footprint_similarity

def dependency_graph_similarity(matching=None, sim=None, threshold=0.5):
    """
    Function returns the function calculate_dependency_graph_similarity which just requires two graphs as inputs,
    with other parameters set.

    Parameters
    ----------
    matching : dict, optional
        If None, a matching is created by create_matching
    sim : similarity method, optional
         If None, equal label similarity will be used.
    threshold : float, optional
        Default is 0.5

    Returns
    -------
    function
    """

    def calculate_dependency_graph_similarity(graph1, graph2):
        """
        Function that calculates the dissimilarity based on the causal profile of the two graphs according to [1]

        Parameters
        ----------
        graph1 : networkx.DiGraph
        graph2 : networkx.DiGraph

        Returns
        -------
        similarity_value : float
            percentage of edges that exist in one graph but not in the other

        References
        ----------
        [1] M. Becker, R. Laue, "A Comparative Survey of Business Process Similarity Measures", Computers in Industry,
        vol. 63, pp. 148-167, 2012.
        """
        dependency_graph1 = build_dependency_graph(graph1)
        dependency_graph2 = build_dependency_graph(graph2)

        if len(dependency_graph1.edges) == 0 and len(dependency_graph2.edges) == 0:
            return 0

        nonlocal matching
        if matching is None:
            matching = create_euquivalence_mapping_for_nodes(graph1, graph2, sim, threshold)

        uncommon_edges = (number_of_uncommon_edges_in_graph(dependency_graph1, dependency_graph2)
                          + number_of_uncommon_edges_in_graph(dependency_graph2, dependency_graph1))

        return uncommon_edges / (len(dependency_graph1.edges) + len(dependency_graph2.edges))

    def number_of_uncommon_edges_in_graph(reference_graph, checked_graph):
        """
        Function that calculates the number of edges that exist in one graph but not in the other
        Edges are also uncommon if they have the same target and source, but are of a different type

        Parameters
        ----------
        reference_graph : networkx.DiGraph
        checked_graph : networkx.DiGraph

        Result
        ------
        uncommon_edges: int
            number of uncommon edges
        """
        uncommon_edges = 0

        for source, target in reference_graph.edges:
            if source in matching:
                source2 = matching[source][0]
                if target in matching:
                    target2 = matching[target][0]
                    if [source2, target2] not in checked_graph.edges:
                        uncommon_edges += 1
                    else:
                        if (reference_graph.get_edge_data(source, target)["type"]
                                != checked_graph.get_edge_data(source2, target2)["type"]):
                            uncommon_edges += 1
                else:
                    uncommon_edges += 1
            else:
                uncommon_edges += 1

        return uncommon_edges


    def build_dependency_graph(graph):
        """
        Function that builds dependency graph from a networkx DiGraph, with sequence and message flows

        Parameters
        ----------
        graph : networkx.DiGraph

        Returns
        -------
        networkx.DiGraph
            dependency graph
        """
        dependency_graph = nx.DiGraph()
        for node in graph.nodes:
            node_type = graph.nodes[node]['type'] if "type" in graph.nodes[node] else None

            if node_type and node_type in Task.__members__:
                if node not in dependency_graph:
                    dependency_graph.add_node(node, type=node_type)
                successors = get_successor_by_type(graph, node, [t for t in Task.__members__])
                for successor in successors:
                    if successor not in dependency_graph:
                        dependency_graph.add_node(node, type=node_type)
                    if [node, successor] not in dependency_graph:
                        edge_type = Flows.sequenceFlow.value
                        if [node, successor] in graph.edges:
                            edge_type = graph.get_edge_data(node, successor)["type"]
                        dependency_graph.add_edge(node, successor, type=edge_type)

        return dependency_graph

    return calculate_dependency_graph_similarity


def causal_behavioural_profile_similarity(matching=None, sim=None, threshold=0.5):
    # doesn't work yet
    """
    Function returns the function calculate_causal_profile_similarity which just requires two graphs as inputs,
    with other parameters set.

    Parameters
    ----------
    matching : dict, optional
        If None, a matching is created by create_matching
    sim : similarity method, optional
         If None, equal label similarity will be used.
    threshold : float, optional
        Default is 0.5

    Returns
    -------
    function
    """

    def calculate_causal_behavioural_profile_similarity(model1, start_nodes1, model2, start_nodes2):
        """
        Function that calculates the similarity based on the causal profile of the two graphs according to [1]

        Parameters
        ----------
        graph1 : networkx.DiGraph
        graph2 : networkx.DiGraph

        Returns
        -------
        similarity_value : float
            percentage of edges that exist in one graph but not in the other

        References
        ----------
        [1] M. Becker, R. Laue, "A Comparative Survey of Business Process Similarity Measures", Computers in Industry,
        vol. 63, pp. 148-167, 2012.
        """
        graph1 = model1.process_graph
        graph2 = model2.process_graph
        nonlocal matching
        if matching is None:
            matching = create_euquivalence_mapping_for_nodes(graph1, graph2, sim, threshold)

        trace_extractor1 = TraceExtractor(model1, connector="Xor")
        traces_g1 = trace_extractor1.get_traces_as_pandas(trace_extractor1.main(start_nodes=start_nodes1))


        trace_extractor2 = TraceExtractor(model2, connector="Xor")
        traces_g2 = trace_extractor2.get_traces_as_pandas(trace_extractor2.main(start_nodes=start_nodes2))

        strict_order_1, reverse_strict_order_1, exclusive_1, interleaving_1 = behavioural_profiles(traces_g1, as_plot=False, as_df=False)
        strict_order_2, reverse_strict_order_2, exclusive_2, interleaving_2 = behavioural_profiles(traces_g2, as_plot=False, as_df=False)

        common_behaviour = (get_common_behaviour(strict_order_1, strict_order_2, matching) +
                            get_common_behaviour(reverse_strict_order_1, reverse_strict_order_2, matching) +
                            get_common_behaviour(exclusive_1, exclusive_2, matching) +
                            get_common_behaviour(interleaving_1, interleaving_2, matching))

        behaviour_num_1 = len(strict_order_1) + len(reverse_strict_order_1) + len(exclusive_1) + len(interleaving_1)
        behaviour_num_2 = len(strict_order_2) + len(reverse_strict_order_2) + len(exclusive_2) + len(interleaving_2)
        return common_behaviour / (behaviour_num_1 + behaviour_num_2)


    def get_common_behaviour(behav1, behav2, mapping):
        common_behav = 0
        for (node1, node2) in behav1:
            if node1 in mapping and node2 in mapping:
                node1_1 = mapping[node1][0]
                node1_2 = mapping[node2][0]
                if (node1_1, node1_2) in behav2:
                    common_behav += 1

        for (node1, node2) in behav2:
            if node1 in mapping and node2 in mapping:
                node1_1 = mapping[node1][0]
                node1_2 = mapping[node2][0]
                if (node1_1, node1_2) in behav1:
                    common_behav += 1

        return common_behav





    return calculate_causal_behavioural_profile_similarity

