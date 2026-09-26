from bef4llm.semantic_quality.similarity.language_similarity.natural_language_similarity import *
from bef4llm.semantic_quality.similarity.similarity_utils import create_euquivalence_mapping_for_nodes
from bef4llm.process_models.graph_representation.node_types import Task, Event

import unittest
import networkx as nx


class TestNodeSimilarities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph1 = nx.DiGraph()
        cls.graph1.add_node('1_1', type='task', name="Activity A")
        cls.graph2 = nx.DiGraph()
        cls.graph2.add_node('2_1', type="startEvent", name="den Kuchen verputzen")
        cls.graph3 = nx.DiGraph()
        cls.graph3.add_node('3_1', type="startEvent", name="der Kuchen verputzt")
        cls.graph4 = nx.DiGraph()
        cls.graph4.add_node('4_1', type="startEvent", name="stemming a stopword")
        cls.graph5 = nx.DiGraph()
        cls.graph5.add_node('5_1', type="startEvent", name="stemming the stopwords")
        cls.graph6 = nx.DiGraph()
        cls.graph6.add_nodes_from([('6_1', {'type': 'startEvent', 'name': 'A', "id": '6_1'}),
                                   ('6_2', {'type': 'task', 'name': 'B', "id": '6_2'}),
                                   ('6_3', {'type': 'exclusiveGateway', 'name': 'C', "id": '6_3'}),
                                   ('6_4', {'type': 'intermediateCatchEvent', 'name': 'D', "id": '6_4'}),
                                   ('6_5', {'type': 'intermediateCatchEvent', 'name': 'E', "id": '6_5'}),
                                   ('6_6', {'type': 'sendTask', 'name': 'F', "id": '6_6'}),
                                   ('6_7', {'type': 'task', 'name': 'G', "id": '6_7'}),
                                   ('6_8', {'type': 'parallelGateway', 'name': 'H', "id": '6_8'}),
                                   ('6_9', {'type': 'sendTask', 'name': 'I', "id": '6_9'}),
                                   ('6_10', {'type': 'manualTask', 'name': 'J', "id": '6_10'}),
                                   ('6_11', {'type': 'parallelGateway', 'name': 'K', "id": '6_11'}),
                                   ('6_12', {'type': 'task', 'name': 'L', "id": '6_12'}),
                                   ('6_13', {'type': 'endEvent', 'name': 'M', "id": '6_13'})])
        cls.graph6.add_edges_from(
            [('6_1', '6_2'),
             ('6_2', '6_3'),
             ('6_2', '6_3'),
             ('6_3', '6_4'),
             ('6_3', '6_5'),
             ('6_4', '6_6'),
             ('6_5', '6_7'),
             ('6_6', '6_2'),
             ('6_7', '6_8'),
             ('6_8', '6_9'),
             ('6_8', '6_10'),
             ('6_9', '6_11'),
             ('6_10', '6_11'),
             ('6_11', '6_12'),
             ('6_12', '6_13')])

        cls.graph66 = nx.DiGraph()
        cls.graph66.add_nodes_from([('66_1', {'type': 'startEvent', 'name': 'A', "id": '66_1'}),
                                   ('66_2', {'type': 'task', 'name': 'B', "id": '66_2'}),
                                   ('66_3', {'type': 'exclusiveGateway', 'name': 'C', "id": '66_3'}),
                                   ('66_4', {'type': 'intermediateCatchEvent', 'name': 'D', "id": '66_4'}),
                                   ('66_5', {'type': 'intermediateCatchEvent', 'name': 'E', "id": '66_5'}),
                                   ('66_6', {'type': 'sendTask', 'name': 'F', "id": '66_6'}),
                                   ('66_7', {'type': 'task', 'name': 'G', "id": '66_7'}),
                                   ('66_8', {'type': 'parallelGateway', 'name': 'H', "id": '66_8'}),
                                   ('66_9', {'type': 'sendTask', 'name': 'I', "id": '66_9'}),
                                   ('66_10', {'type': 'manualTask', 'name': 'J', "id": '66_10'}),
                                   ('66_11', {'type': 'parallelGateway', 'name': 'K', "id": '66_11'}),
                                   ('66_12', {'type': 'task', 'name': 'L', "id": '66_12'}),
                                   ('66_13', {'type': 'endEvent', 'name': 'M', "id": '66_13'})])
        cls.graph66.add_edges_from(
            [('66_1', '66_2'),
             ('66_2', '66_3'),
             ('66_2', '66_3'),
             ('66_3', '66_4'),
             ('66_3', '66_5'),
             ('66_4', '66_6'),
             ('66_5', '66_7'),
             ('66_6', '66_2'),
             ('66_7', '66_8'),
             ('66_8', '66_9'),
             ('66_8', '66_10'),
             ('66_9', '66_11'),
             ('66_10', '66_11'),
             ('66_11', '66_12'),
             ('66_12', '66_13')])

        cls.graph6_2 = nx.DiGraph()
        cls.graph6_2.add_nodes_from([('6_2_1', {'type': 'startEvent', 'name': 'A', "id": "6_2_1"}),
                                     ('6_2_2', {'type': 'task', 'name': 'B', "id": "6_2_2"}),
                                     ('6_2_3', {'type': 'exclusiveGateway', 'name': 'C', "id": "6_2_3"}),
                                     ('6_2_3_1', {'type': 'task', 'name': 'C', "id": "6_2_3_1"}),
                                     ('6_2_4', {'type': 'intermediateCatchEvent', 'name': 'C2', "id": "6_2_4"}),
                                     ('6_2_5', {'type': 'intermediateCatchEvent', 'name': 'E', "id": "6_2_5"}),
                                     ('6_2_6', {'type': 'sendTask', 'name': 'F', "id": "6_2_6"}),
                                     ('6_2_7', {'type': 'task', 'name': 'G', "id": "6_2_7"}),
                                     ('6_2_8', {'type': 'parallelGateway', 'name': 'H', "id": "6_2_8"}),
                                     ('6_2_9', {'type': 'sendTask', 'name': 'I', "id": "6_2_9"}),
                                     ('6_2_10', {'type': 'manualTask', 'name': 'J', "id": "6_2_10"}),
                                     ('6_2_11', {'type': 'parallelGateway', 'name': 'K', "id": "6_2_11"}),
                                     ('6_2_12', {'type': 'task', 'name': 'L', "id": "6_2_12"}),
                                     ('6_2_13', {'type': 'endEvent', 'name': 'M', "id": "6_2_13"})])
        cls.graph6_2.add_edges_from(
            [('6_2_1', '6_2_2'),
             ('6_2_2', '6_2_3'),
             ('6_2_3_1', '6_2_3'),
             ('6_2_3', '6_2_4'),
             ('6_2_3', '6_2_5'),
             ('6_2_4', '6_2_6'),
             ('6_2_5', '6_2_7'),
             ('6_2_6', '6_2_2'),
             ('6_2_7', '6_2_8'),
             ('6_2_8', '6_2_9'),
             ('6_2_8', '6_2_10'),
             ('6_2_9', '6_2_11'),
             ('6_2_10', '6_2_11'),
             ('6_2_11', '6_2_12'),
             ('6_2_12', '6_2_13')])

        cls.graph7 = nx.DiGraph()
        cls.graph7.add_node('7_1', type='Gateway')
        cls.graph7.add_node('7_2', type='task')
        cls.graph7.add_node('7_3', type='Gateway')

        cls.graph9 = nx.DiGraph()
        cls.graph9.add_nodes_from([('9_1', {'type': 'task', 'name': 'A'}),
                                   ('9_2', {'type': 'task', 'name': 'B'}),
                                   ('9_3', {'type': 'task', 'name': 'suc1'})])
        cls.graph9.add_edges_from([('9_1', '9_3'), ('9_3', '9_2')])
        cls.graph10 = nx.DiGraph()
        cls.graph10.add_nodes_from([('10_1', {'type': 'task', 'name': 'A'}),
                                    ('10_2', {'type': 'task', 'name': 'B'}),
                                    ('10_3', {'type': 'task', 'name': 'suc1'}),
                                    ('10_4', {'type': 'task', 'name': 'suc2'}),
                                    ('10_5', {'type': 'task', 'name': 'suc3'}),
                                    ('10_6', {'type': 'task', 'name': 'suc4'})])
        cls.graph10.add_edges_from(
            [('10_1', '10_3'),
             ('10_1', '10_4'),
             ('10_1', '10_5'),
             ('10_1', '10_6'),
             ('10_3', '10_2'),
             ('10_4', '10_2'),
             ('10_5', '10_2'),
             ('10_6', '10_2')])
        cls.graph11 = nx.DiGraph()
        cls.graph11.add_node('11_1', name="Activity B", type="task")
        cls.graph_sims = nx.Graph()
        cls.graph_sims.add_node(0, name="Customer inquiries about product")
        cls.graph_sims.add_node(1, name="Customer inquiries - about product")
        cls.graph_sims.add_node(2, name="Customer inquiry about product")
        cls.graph_sims.add_node(3, name=" ")
        cls.graph_sims.add_node(4, name="Customer inquiry processing")
        cls.graph_sims.add_node(5, name="Client inquiry query processing")

    @classmethod
    def tearDownClass(cls):
        del cls.graph1
        del cls.graph2
        del cls.graph3
        del cls.graph4
        del cls.graph5
        del cls.graph6
        del cls.graph6_2
        del cls.graph7
        del cls.graph9
        del cls.graph10
        del cls.graph11
        del cls.graph_sims

    def test_get_graph_for_node(self):
        list_of_graphs = [self.graph1, self.graph2, self.graph3, self.graph4, self.graph5, self.graph6]
        self.assertEqual(len(list_of_graphs), 6)
        n1 = list(self.graph1.nodes(data=True))[0][0]
        n4 = list(self.graph4.nodes(data=True))[0][0]
        self.assertEqual(get_graph_for_node(n1, list_of_graphs), self.graph1)
        self.assertEqual(get_graph_for_node(n4, list_of_graphs), self.graph4)

    def test_syntactic_simimlarity(self):
        self.nodes = list(self.graph_sims.nodes(data=True))
        self.node1 = self.nodes[0][1]
        self.label1_chars = self.nodes[1][1]
        self.node2 = self.nodes[2][1]
        self.empty_label = self.nodes[3][1]
        syn_sim = syntactic_similarity()
        self.assertEqual(syn_sim(self.node1, self.node2), 0.90625)
        self.assertEqual(syn_sim(self.label1_chars, self.node2), 0.90625)
        self.assertEqual(syn_sim(self.empty_label, self.node2), 0)
        kwargs = {"node1": self.empty_label, "node2": self.empty_label}
        self.assertRaises(ZeroDivisionError, syn_sim, **kwargs)
        nodes = {"node1": "blub", "node2": 123}
        self.assertRaises(TypeError, syntactic_similarity, **nodes)

    def test_semantic_similarity(self):
        self.nodes = list(self.graph_sims.nodes(data=True))
        self.node1 = self.nodes[4][1]
        self.node2 = self.nodes[5][1]
        self.empty_label = self.nodes[3][1]
        sim = semantic_similarity()
        self.assertEqual(sim(self.node1, self.node2), 0.6875)
        self.assertEqual(sim(self.empty_label, self.node2), 0)
        kwargs = {"node1": self.empty_label, "node2": self.empty_label}
        self.assertRaises(ZeroDivisionError, sim, **kwargs)
        nodes = {"node1": "blub", "node2": 123}
        self.assertRaises(TypeError, sim, **nodes)
        sim = semantic_similarity(stemmed=False)
        self.assertEqual(sim(self.node1, self.node2), 0.6875)
        sim = semantic_similarity(remove_spec_chars=False, remove_all_stopwords=False)
        self.assertEqual(sim(self.node1, self.node2), 0.6875)
        node = {"name": "Time for a test", id: "123"}
        self.assertEqual(sim(node, node), 1.0)

    def test_context_similarity(self):
        syntactic_sim = syntactic_similarity()

        node_g6 = self.graph6.nodes["6_3"]
        node_g66 = self.graph66.nodes["66_3"]
        #mapping = self.calculate_mapping(self.graph6, self.graph6)
        mapping = create_euquivalence_mapping_for_nodes(self.graph6, self.graph66, sim=syntactic_sim)
        con_sim = context_similarity(set_of_graphs=[self.graph6, self.graph66], mapping=mapping)
        self.assertEqual(con_sim(node_g6, node_g66), 1.0)
        node_g6_2 = self.graph6_2.nodes["6_2_3"]
        #mapping = self.calculate_mapping(self.graph6, self.graph6_2)
        mapping = create_euquivalence_mapping_for_nodes(self.graph6, self.graph6_2, sim=syntactic_sim)
        con_sim = context_similarity(set_of_graphs=[self.graph6, self.graph6_2], mapping=mapping)
        self.assertEqual(con_sim(node_g6, node_g6_2), 1 / 2)

    def calculate_mapping(self, graph, reference):
        def get_max_similarity(current_sim, max_sim, node, reference_node):
            if current_sim > max_sim[0]:
                return current_sim, node, reference_node
            else:
                return max_sim

        syntactic_sim = syntactic_similarity()
        node_sim = []
        for node in graph.nodes:
            max_sim = (0, None, None)
            for reference_node in reference.nodes:
                print(graph[node])
                if (graph[node]["type"] in Task.__members__ or reference[reference_node]["type"] in Task.__members__
                        or graph[node]["type"] in Event.__members__ or reference[reference_node][
                            "type"] in Event.__members__):
                    node = graph.nodes[node]
                    reference = reference.nodes[reference_node]
                    max_sim = get_max_similarity(syntactic_sim(node, reference), max_sim, node, reference_node)

            node_sim.append(max_sim)

        print(create_euquivalence_mapping_for_nodes(graph, reference, node_sim))

        return create_euquivalence_mapping_for_nodes(graph, reference, node_sim)

    def test_levenshtein_distance(self):
        ld = levenshtein_dist(case_sensitive=False)
        self.assertEqual(ld("test levenshtein distance", "test levenshtein distance"), 0.0)
        ld = levenshtein_dist(case_sensitive=True)
        self.assertEqual(ld("test levenshtein distance", "test Levenshtein distance"), 1)
        self.assertEqual(ld("test distance levenshtein", "test Levenshtein distance"), 19)


if __name__ == '__main__':
    unittest.main()
