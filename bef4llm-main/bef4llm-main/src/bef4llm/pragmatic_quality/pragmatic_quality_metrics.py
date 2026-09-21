import math
import networkx as nx
from bef4llm.process_models import collaboration_model_utils as utils
from bef4llm.process_models.graph_representation.node_types import Gateway, Event, Flows, Task, Others


def diameter(graph):
    """
    Returns the diameter

    Parameters
    ----------
    graph : networkx.DiGraph

    Returns
    -------
    int
        the length of the longest path

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction,
    and guidelines for correctness (Vol. 6). Springer Science & Business Media.

    """

    # List with endEvents to start the loop with
    endlist = [node for node in graph.nodes if ("type" in graph.nodes[node] and graph.nodes[node]["type"] == Event.endEvent.value)]

    # List with the connectors to exclude them later
    conlist = [node[0] for node in graph.nodes(data='type') if str(node[1]).endswith('Gateway')]

    # Dict denoting the longest path to act
    longest_path_dict = dict()

    # Recursiv function to find the longest path

    def countpaths(nodeslist, visited_pre, level):
        # count activity occurences -> return empty list
        """
        counts all paths, only with sequenceflows

        Parameters
        ----------
        nodeslist : list

        Returns
        -------
        int
            the max counter
        """
        maxcounter = 1

        for n in nodeslist:
            if n in longest_path_dict:
                return longest_path_dict[n]

            # end loop
            if n in visited_pre and visited_pre[n] > 2:
                return 0
            if "type" in graph.nodes[n] and graph.nodes[n]["type"] == Event.endEvent.value:
                visited_pre = dict()
            if n in conlist:
                counter = 0
            else:
                counter = 1

            predecessors = [p for p in list(graph.predecessors(n))
                            if graph.edges[p, n]["type"] == Flows.sequenceFlow.value]

            if predecessors == []:
                return counter
            else:
                for pre in predecessors:
                    if pre in visited_pre:
                        visited_pre[pre] = visited_pre[pre] + 1
                    else:
                        visited_pre[pre] = 1

                #visited_pre += predecessors
                lp = countpaths(predecessors, visited_pre, level + 1)
                longest_path_dict[n] = lp
                counter = counter + lp

                if maxcounter < counter:
                    maxcounter = counter
        return maxcounter

    return countpaths(endlist, visited_pre=dict(), level=0)

def gateway_heterogeneity(graph):
    """
    Returns the heterogeneity according to [1]_

    Parameters
    ----------
    graph: networkx.DiGraph

    Returns
    -------
    float
        entropy over the diﬀerent connector types

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction,
    and guidelines for correctness (Vol. 6). Springer Science & Business Media.
    """
    gateway_nums = utils.get_number_gateway_types(graph)
    con_num = sum([sum(gateway_nums[g_dict].values()) for g_dict in gateway_nums])
    if con_num == 0:
        return 0

    parallel_val = sum(gateway_nums[Gateway.parallelGateway.value].values())/con_num
    exlusive_val = sum(gateway_nums[Gateway.exclusiveGateway.value].values())/con_num
    event_val = sum(gateway_nums[Gateway.eventBasedGateway.value].values())/con_num
    inclusive_val = sum(gateway_nums[Gateway.inclusiveGateway.value].values()) / con_num

    try:
        parallel_res = parallel_val * math.log(parallel_val, 3)
    except ValueError:
        parallel_res = 0

    try:
        exlusive_res = exlusive_val * math.log(exlusive_val, 3)
    except ValueError:
        exlusive_res = 0

    try:
        event_res = event_val * math.log(event_val, 3)
    except ValueError:
        event_res = 0

    try:
        inclusive_res = inclusive_val * math.log(inclusive_val, 3)
    except ValueError:
        inclusive_res = 0
    result = parallel_res + exlusive_res + event_res + inclusive_res

    return -1 * result


def control_flow_complexity(graph):
    """
    Returns the control flow complexity according to [1]_

    Parameters
    ----------
    graph: networkx.DiGraph

    Returns
    -------
    int
        the control flow complexity as an integer

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction,
    and guidelines for correctness (Vol. 6). Springer Science & Business Media.
    """
    gateways = utils.get_gateways_by_types(graph)

    sum_and = len(gateways[Gateway.parallelGateway.value]["split"])
    sum_xor = 0
    sum_or = 0

    for conn in gateways[Gateway.exclusiveGateway.value]["split"]:
        sum_xor = sum_xor + len(graph.nodes[conn]['outgoing'])

    for conn in gateways[Gateway.eventBasedGateway.value]["split"]:
        sum_xor = sum_xor + len(graph.nodes[conn]['outgoing'])

    for conn in gateways[Gateway.inclusiveGateway.value]["split"]:
        sum_or = sum_or + (2 ** len(graph.nodes[conn]['outgoing'])) - 1


    return sum_and + sum_xor + sum_or

def cross_connectivity(graph, number_nodes):
    """
    Returns the cross connectivity according to [1]_

    Parameters
    ----------
    graph: networkx.DiGraph

    Returns
    -------
    float
        the maximal weights for any path between every
        two nodes and divided by the number of paths between
        every two nodes

    References
    ----------
    [1] Vanderfeesten, I., Reijers, H. A., Mendling, J., van der Aalst,
    W. M., & Cardoso, J. (2008, June).  c. In International Conference on Advanced
     Information Systems Engineering (pp. 480-494). Springer,
     Berlin, Heidelberg.
    """

    def edge_list(pathlist):
        """
        Returns a list of edges given a certain path

        Parameters
        ----------
        pathlist: node list list
            list of paths


        Returns
        -------
        list
            a list of edge_paths given a list of node_paths


        """
        list_ = []
        for i in range(0, len(pathlist) - 1):
            edge = graph[pathlist[i]][pathlist[i + 1]]['id']
            list_.append(edge)
        return list_

    def weight_of_node(node):
        """
        Returns the weight of the node

        Parameters
        ----------
        node: node
            node object

        Returns
        -------
        float
            the weight of a node

        """
        # when gateway and degree
        weights = {
            Gateway.inclusiveGateway.value:
                1 / (2 ** utils.get_degree_node(node, graph) - 1) +
                (((2 ** utils.get_degree_node(node, graph) - 2) /
                (2 ** utils.get_degree_node(node, graph) - 1) * 1) * 1/utils.get_degree_node(node, graph)) ,
            Gateway.exclusiveGateway.value: 1 / utils.get_degree_node(node, graph),
            Gateway.eventBasedGateway.value: 1 / utils.get_degree_node(node, graph),
            Gateway.complexGateway.value: 1 / utils.get_degree_node(node, graph),
            Gateway.parallelGateway.value: 1,
            Task.task.value: 1,
            Task.function.value: 1,
            Task.userTask.value: 1,
            Task.serviceTask.value: 1,
            Task.manualTask.value: 1,
            Task.sendTask.value: 1,
            Task.receiveTask.value: 1,
            Task.callActivity.value: 1,
            Task.businessRuleTask.value: 1,
            Others.subProcess.value: 1,
            Event.event.value: 1,
            Event.startEvent.value: 1,
            Event.intermediateCatchEvent.value: 1,
            Event.endEvent.value: 1,
            Event.intermediateThrowEvent.value: 1,
            Event.boundaryEvent.value: 1,

        }
        return weights.get(node[1], "no matching type")

    def max_weight_paths(pathlist):
        """
        Returns the max weight found in the list of paths

        Parameters
        ----------
        pathlist: node list list
            list with paths


        Returns
        -------
        float
            the max. weight found in list of paths

        """
        max_weight = 0
        current_weight = 1
        for path in pathlist:
            edge_path = edge_list(path)

            for i, item in enumerate(edge_path):
                if edge_dict.get(item):
                    current_weight = current_weight * edge_dict.get(item)
            if current_weight > max_weight:
                max_weight = current_weight
        return max_weight

    nodeslist = graph.nodes()

    edges = graph.edges(data='id')
    edge_dict = {}
    # create empty dictionary
    for tuple in edges:
        if len(tuple) == 2:
            # fills the dict with edges and their value
            firstnode = utils.get_node_by_id(tuple[0], graph)
            secondnode = utils.get_node_by_id(tuple[1], graph)
            #print(firstnode, secondnode)
            edge_id = tuple[2]
            if graph.get_edge_data(tuple[0], tuple[1])['type'] == Flows.sequenceFlow.name\
                    and graph.nodes[firstnode] != {} and graph.nodes[secondnode] != {}:
                weight_edge = weight_of_node(firstnode) * weight_of_node(secondnode)
                edge_dict[edge_id] = weight_edge

    resultcounter = 0
    # sum up matrix

    for n1 in nodeslist:
        for n2 in nodeslist:
            all_path_list = list(nx.all_simple_paths(graph, n1, n2))

            max_weight = max_weight_paths(all_path_list)
            resultcounter = resultcounter + max_weight

    return resultcounter / (number_nodes * (number_nodes - 1))

def token_splits(graph):
    """
    Returns the number of token splits according to [1]_

    Parameters
    ----------
    graph: networkx.DiGraph

    Returns
    -------
    int
        the sum of the output-degree of AND-joins and OR-joins minus one

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction,
    and guidelines for correctness (Vol. 6). Springer Science & Business Media.
    """
    gateways = utils.get_gateways_by_types(graph)

    result = 0

    for conn in gateways[Gateway.parallelGateway.value]["split"]:
        result = result + (len(graph.nodes[conn]["outgoing"]) - 1)


    for conn in gateways[Gateway.parallelGateway.value]["split"]:
        result = result + (len(graph.nodes[conn]["outgoing"]) - 1)


    return result

def depth(graph):
    """
    Returns the depth of the model according to [1]_

    Parameters
    ----------
    graph : networkx.DiGraph

    Returns
    -------
    int
        the depth of the model

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction,
    and guidelines for correctness (Vol. 6). Springer Science & Business Media.
    """
    nodes = graph.nodes(data='type')
    startlist = []
    for node in nodes:
        if node[1] == Event.startEvent.value:
            startlist.append(node[0])

    id_splitlist = []
    id_joinlist = []
    gateways = utils.get_gateways_by_types(graph)
    for gateway in gateways:
        if "split" in gateways[gateway]:
            id_splitlist.extend(gateways[gateway]["split"])
        if "join" in gateways[gateway]:
            id_joinlist.extend(gateways[gateway]["join"])

    def in_depth(no, visited:set):
        """
        recursive help function
        Parameters
        ----------
        no: node
        visited: set
            denotes nodes that are already visited, when they are reached again they function as the end piont of the search

        Returns
        -------
        """
        #print(graph.nodes[no]["name"])
        if no in visited:
            return 0

        visited.add(no)
       
        counter = 0
        maxcounter = 0

        if no in id_splitlist:
            counter = counter + 1
        if no in id_joinlist:
            counter = counter - 1
        if counter > maxcounter:
            maxcounter = counter
        for succ in graph.successors(no):
            new_max = maxcounter + in_depth(succ, visited)

            if new_max > maxcounter:
                maxcounter = new_max
        return maxcounter

    def out_depth(no, visited:set):
        """
        recursive help function
        Parameters
        ----------
        no: node
        visited: set
            denotes nodes that are already visited, when they are reached again they function as the end piont of the search

        Returns
        -------
        """
        if no in visited:
            return 0

        visited.add(no)

        counter = 0
        maxcounter = 0

        if no in id_splitlist:
            counter = counter - 1
        if no in id_joinlist:
            counter = counter + 1
        if counter > maxcounter:
            maxcounter = counter
        for succ in graph.successors(no):
            new_max = maxcounter + out_depth(succ, visited)

            if new_max > maxcounter:
                maxcounter = new_max
        return maxcounter

    max_in_depth = 0
    for s in startlist:
        new_max = in_depth(s, visited=set())
        if new_max > max_in_depth:
            max_in_depth = new_max

    max_out_depth = 0
    for s in startlist:
        new_max = out_depth(s, visited=set())
        if new_max > max_out_depth:
            max_out_depth = new_max

    return min(max_out_depth, max_in_depth)


def sequentiality(model):
    """
    Returns the depth of the model according to [1]_

    Parameters
    ----------
    model : CollaborationModel

    Returns
    -------
    int
        sequentiality of the model

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction,
    and guidelines for correctness (Vol. 6). Springer Science & Business Media.
    """
    if len(model.sequence_flow_id_edge_mapping) == 0:
        return 0


    # message flows are not taken into account, only sequence flows!
    arcs_between_non_gateway_nodes = 0
    for edge in model.sequence_flow_id_edge_mapping:
        source, target = model.sequence_flow_id_edge_mapping[edge]
        if "type" in model.process_graph.nodes[source] and "type" in model.process_graph.nodes[target] :
            if not model.process_graph.nodes[source]["type"] in [item.value for item in Gateway] \
                    and not model.process_graph.nodes[target]["type"] in [item.value for item in Gateway]:
                arcs_between_non_gateway_nodes += 1

    return arcs_between_non_gateway_nodes / len(model.sequence_flow_id_edge_mapping)


def seperatibility(graph, number_nodes):
    """
    Returns the seperatibility of the model according to [1]

    Parameters
    ----------
    graph: networkx.DiGraph
    number_nodes: int (number of nodes in the process model)

    Returns
    -------
    int
        seperatibility of the model

    References
    ==========
    [1] Mendling, J. (2008). Metrics for process models: empirical foundations of verification, error prediction,
    and guidelines for correctness (Vol. 6). Springer Science & Business Media.
    """
    def cut_vertices(graph):
        """
        Returns the number of cut vertices

        Parameters
        ----------
        graph: networkx.DiGraph

        Returns
        -------
        int
            the number of cut-vertices

        """
        return len(list(nx.articulation_points(graph.to_undirected())))

    return cut_vertices(graph) / (number_nodes - 2)


