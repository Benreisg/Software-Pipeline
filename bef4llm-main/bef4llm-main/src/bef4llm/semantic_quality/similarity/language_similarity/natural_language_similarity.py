from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import tokenize, remove_special_chars
from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import stem_tokens, remove_stopwords, \
    Language, check_synonym
from bef4llm.process_models.graph_representation.node_types import Gateway
import numpy as np
from copy import deepcopy


def syntactic_similarity(remove_spec_chars=True):
    """
        Functions returns the function calculate_syntactic_similarity which just requires two nodes as inputs
        and specification of the remove special characters flag.

        Parameters
        ----------
        remove_spec_chars : bool
            If true, special characters such as ".,'{[' will be removed.

        Returns
        -------
        Function - calculate_syntactic_similarity().
        """

    def calculate_syntactic_similarity(node1, node2):
        """
        Function calculates Syntactic Similarity of two nodes.

        Parameters
        ----------
        node1: string
            Data dict of the node (from nx DiGraph)
        node2: string
            Data dict of the node (from nx DiGraph)

        Returns
        -------
            similarity: float
                Degree of similarity as measured by the string-edit distance.
        """

        if not isinstance(node1, dict) or not isinstance(node2, dict):
            raise TypeError("The node is represented by the its attributes as a dictionary.")

        if ("name" not in node1 or "name" not in node2 or # no name attribute
                (node1["name"] == "" and node2["name"] == "")): # name is "" in both cases
            return 0

        label_node1 = node1["name"]
        label_node2 = node2["name"]

        token1 = tokenize(label_node1)
        token2 = tokenize(label_node2)

        if remove_spec_chars:
            token1 = remove_special_chars(token1)
            token2 = remove_special_chars(token2)

        string1 = ' '.join(token1).lower()
        string2 = ' '.join(token2).lower()

        dist_fun = levenshtein_dist()
        string_distance = dist_fun(string1, string2)
        max_length = max(len(string1), len(string2))

        if max_length == 0:
            print("ZeroDivisionError - at least one of the strings (labels) must be not empty")
            raise ZeroDivisionError
        else:
            return 1 - (string_distance / max_length)

    return calculate_syntactic_similarity


def semantic_similarity(lang=Language.ENGLISH,
                        remove_spec_chars=True,
                        remove_all_stopwords=True,
                        stemmed=True):
    """
    Functions returns the function calculate_semantic_similarity which just requires two nodes as inputs
    and specification of relevant flags.

    Parameters
    ----------
    lang: language
        Language of the string to be analysed.
    remove_spec_chars: bool
        If true, special characters such as ".,'{[' will be removed.
    remove_all_stopwords: bool
        If true, stopwords will be removed.
    stemmed: bool
        If true, function will be stemmed.

    Returns
    -------
    Function - calculate_syntactic_similarity().
    """

    def calculate_semantic_similarity(node1, node2):
        """
        Function calculates Semantic Similarity of two nodes.

        Parameters
        ----------
        node1: dict
            Data dict of the node (from nx DiGraph)
        node2: dict
            Data dict of the node (from nx DiGraph)

        Returns
        -------
            similarity: float
                Degree of similarity, based on equivalence between the words they consist of.
                An exact match is assumed to be preferred over a match on synonyms.
        """
        if not isinstance(node1, dict) or not isinstance(node2, dict):
            raise TypeError("The node is represented by the its attributes as a dictionary.")

        if "name" not in node1 or "name" not in node2 or (node1["name"] == "" and node2["name"] == ""):
            return 0

        label_node1 = node1["name"]
        label_node2 = node2["name"]

        token1 = sorted(tokenize(label_node1.lower()))
        token2 = sorted(tokenize(label_node2.lower()))

        max_length = max(len(token1), len(token2))

        if max_length == 0:
            print("ZeroDivisionError - no of items of the max label cannot be 0")
            raise ZeroDivisionError

        if remove_spec_chars:
            token1 = remove_special_chars(token1)
            token2 = remove_special_chars(token2)

        if remove_all_stopwords:
            token1 = remove_stopwords(token1, lang)
            token2 = remove_stopwords(token2, lang)

        max_length = max(len(token1), len(token2))

        if stemmed:
            stemmed1 = stem_tokens(token1, lang)
            stemmed2 = stem_tokens(token2, lang)
        else:
            # deepcopy to assure reference to a separate list
            stemmed1 = deepcopy(token1)
            stemmed2 = deepcopy(token2)

        same_strings = 0
        factor = 0
        for i, this_word in enumerate(stemmed1, start=0):
            for j, other_word in enumerate(stemmed2, start=0):
                if this_word == other_word and i >= factor and j >= factor:
                    del token1[i - factor]
                    del token2[j - factor]
                    same_strings += 1
                    factor += 1
        synonyms = 0
        for this_word in token1:
            for other_word in token2:
                if check_synonym(this_word, other_word):
                    synonyms += 1

        return ((1.0 * same_strings) + (0.75 * synonyms)) / max_length

    return calculate_semantic_similarity


def context_similarity(set_of_graphs, mapping):
    """
    Functions returns the function calculate_bow_similarity which just requires two nodes as inputs,
    with parameters graph1, graph2, mapping set.

    Parameters
    ----------
    set_of_graphs : list of networkx graph
        node1 and node2 need to be contained in the graphs from the list.
    mapping : dict
        mapping of graph1 and graph2

    Returns
    -------
    function
    """

    def get_predecessors(node, graph, visited):
        """
        Computes all non-gateway predecessors of a node

        Parameters
        ----------
        node : str
            node to get predecessors from
        graph : networkx.DiGraph
            graph the node is contained
        visited : list of networkx.Graph.NodeView
            all currently visited nodes

        Returns
        -------
        predecessors : list of networkx.Graph.NodeView
            all predecessors of the given node
        """

        result = []

        for predecessor in list(graph.predecessors(node)):

            if predecessor in visited:
                continue

            visited.append(predecessor)
            node_type = graph.nodes[predecessor]['type'] if "type" in graph.nodes[predecessor] else None

            if node_type and node_type in Gateway.__members__ or 'name' not in graph.nodes[predecessor]:
                recursive = get_predecessors(predecessor, graph, visited)

                for n in recursive:
                    if n not in result:
                        result.append(n)

            else:
                result.append(predecessor)

        return result

    def get_successors(node, graph, visited):
        """
        Computes all non-gateway successors of a node

        Parameters
        ----------
        node : str
            node to get successors from
        graph : networkx.DiGraph
            graph the node is contained
        visited : list of networkx.Graph.NodeDataView
            all currently visited nodes

        Returns
        -------
        successors : list of networkx.Graph.NodeDataView
            all successors of the given node
        """
        result = []

        for successor in list(graph.successors(node)):

            if successor in visited:
                continue

            visited.append(successor)
            node_type = graph.nodes[successor]['type'] if "type" in graph.nodes[successor] else None

            if node_type and node_type in Gateway.__members__ or 'name' not in graph.nodes[successor]:
                recursive = get_successors(successor, graph, visited)

                for n in recursive:
                    if n not in result:
                        result.append(n)

            else:
                result.append(successor)

        return result

    def calculate_context_similarity(node1, node2):
        """
        Calculates the context_sim similarity of two nodes

        Parameters
        ----------
        node1 : dict
            Data dict of the node (from nx DiGraph)
        node2 : dict
            Data dict of the node (from nx DiGraph)

        Returns
        -------
        similarity : float
            resulting context_sim similarity
        Notes
        _____
        Based on [1] La Rosa, M., Dumas, M., Uba, R., & Dijkman, R. (2010, October).
        Merging business process models. In OTM Confederated International Conferences"
        On the Move to Meaningful Internet Systems" (pp. 96-113). Springer, Berlin, Heidelberg.
        """
        if not isinstance(node1, dict) or not isinstance(node2, dict):
            raise TypeError("The node is represented by the its attributes as a dictionary.")
        graph1 = get_graph_for_node(node1["id"], set_of_graphs)
        graph2 = get_graph_for_node(node2["id"], set_of_graphs)

        node1_predecessors = get_predecessors(node1["id"], graph1, [])
        node2_predecessors = get_predecessors(node2["id"], graph2, [])

        node1_successors = get_successors(node1["id"], graph1, [])
        node2_successors = get_successors(node2["id"], graph2, [])
        map1 = []
        map2 = []

        pre1_matches = []
        succ1_matches = []

        for pre1 in node1_predecessors:
            if pre1 in mapping.keys():
                pre1_matches.append(mapping[pre1][0])

        for succ1 in node1_successors:
            if succ1 in mapping.keys():
                succ1_matches.append(mapping[succ1][0])

        for pre2 in node2_predecessors:
            if pre2 in pre1_matches:
                map1.append(pre2)

        for succ2 in node2_successors:
            if succ2 in succ1_matches:
                map2.append(succ2)

        dividor = (max(node1_predecessors.__len__(), node2_predecessors.__len__())
                   + max(node1_successors.__len__(), node2_successors.__len__()))

        if dividor == 0:
            return 0
        context = (map1.__len__() + map2.__len__()) / dividor

        return context

    return calculate_context_similarity


def levenshtein_dist(case_sensitive=False):
    """
    Functions returns the function __calculate_levenshtein_distance which just requires two words as inputs,
    with parameter case_sensitive set.

    Parameters
    ----------
    case_sensitive : bool, optional
        default False, if True, levenshtein_sim is case sensitive.

    Returns
    -------
    function

    """

    def calculate_levenshtein_distance(word1, word2):
        """
        Calculates the Levenshtein Distance between two words

        Parameters
        ----------
        word1 : str
            first string
        word2 : str
            second string

        Returns
        -------
        distance: float
            resulting Levenshtein Distance

        """
        size_x = len(word1) + 1
        size_y = len(word2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if (case_sensitive and word1[x - 1].__eq__(word2[y - 1])) or \
                        (not case_sensitive and word1[x - 1].casefold().__eq__(word2[y - 1].casefold())):
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1],
                        matrix[x, y - 1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )

        return matrix[size_x - 1, size_y - 1]

    return calculate_levenshtein_distance


def get_graph_for_node(node, set_of_graphs):
    """
    Finds the graph that contains a node.

    Parameters
    ----------
    node : dict
    set_of_graphs : list of networkx.DiGraph

    Returns
    -------
    networkx.DiGraph
    """
    graphs = [graph for graph in set_of_graphs if node in graph.nodes]
    graph = graphs[0]

    return graph
