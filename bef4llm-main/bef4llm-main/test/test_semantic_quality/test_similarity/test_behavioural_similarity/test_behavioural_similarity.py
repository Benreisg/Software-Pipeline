import unittest
import networkx as nx
import sys
sys.path.append("src")
from bef4llm.semantic_quality.similarity.behavioural_similarity.behavioural_similarity import *

class TestModelSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph1 = nx.DiGraph()
        cls.graph1.add_node(1_1, name="Event A", type="startEvent")
        cls.graph1.add_node(1_2, name="Task 1", type="task")
        cls.graph1.add_node(1_3, name="Gateway", type="exclusiveGateway")
        cls.graph1.add_node(1_4, name="Task 2", type="task")
        cls.graph1.add_node(1_5, name="Gateway", type="inclusiveGateway")
        cls.graph1.add_node(1_6, name="Event B", type="endEvent")
        cls.graph1.add_edges_from([(1_1, 1_2), (1_2, 1_3), (1_3, 1_4), (1_4, 1_5), (1_5, 1_6)])
        cls.graph2 = nx.DiGraph()
        cls.graph2.add_node('2_1', type="startEvent", name="den Kuchen verputzen")
        cls.graph3 = nx.DiGraph()
        cls.graph3.add_node('3_1', type="startEvent", name="der Kuchen verputzt")
        cls.graph4 = nx.DiGraph()
        cls.graph4.add_node('4_1', type="startEvent", name="stemming a stopword")
        cls.graph5 = nx.DiGraph()
        cls.graph5.add_node('5_1', type="startEvent", name="stemming the stopwords")
        cls.graph6 = nx.DiGraph()
        cls.graph6.add_nodes_from([('6_1', {'type': 'task', 'name': 'A'}),
                                   ('6_2', {'type': 'task', 'name': 'B'}),
                                   ('6_3', {'type': 'task', 'name': 'suc1'})])
        cls.graph6.add_edges_from([('6_1', '6_3'), ('6_3', '6_2')])
        cls.graph7 = nx.DiGraph()
        cls.graph7.add_nodes_from([('7_1', {'type': 'task', 'name': 'A'}),
                                   ('7_2', {'type': 'task', 'name': 'B'}),
                                   ('7_3', {'type': 'task', 'name': 'suc1'}),
                                   ('7_4', {'type': 'task', 'name': 'suc2'}),
                                   ('7_5', {'type': 'task', 'name': 'suc3'}),
                                   ('7_6', {'type': 'task', 'name': 'suc4'})])
        cls.graph7.add_edges_from(
            [('7_1', '7_3'),
             ('7_1', '7_4'),
             ('7_1', '7_5'),
             ('7_1', '7_6'),
             ('7_3', '7_2'),
             ('7_4', '7_2'),
             ('7_5', '7_2'),
             ('7_6', '7_2')], type=Flows.sequenceFlow.value)
        cls.graph7a = nx.DiGraph()
        cls.graph7a.add_nodes_from([('7a_1', {'type': 'task', 'name': 'A'}),
                                    ('7a_2', {'type': 'task', 'name': 'B'}),
                                    ('7a_3', {'type': 'task', 'name': 'suc1'}),
                                    ('7a_4', {'type': 'task', 'name': 'suc2'}),
                                    ('7a_5', {'type': 'task', 'name': 'suc3'}),
                                    ('7a_6', {'type': 'task', 'name': 'suc4'})])
        cls.graph7a.add_edges_from(
            [('7a_1', '7a_3'),
             ('7a_1', '7a_4'),
             ('7a_1', '7a_5'),
             ('7a_1', '7a_6'),
             ('7a_3', '7a_2'),
             ('7a_4', '7a_2'),
             ('7a_5', '7a_2'),
             ('7a_6', '7a_2')], type=Flows.sequenceFlow.value)
        cls.graph8 = nx.DiGraph()
        cls.graph8.add_node(8_1, type="startEvent", name="Complaint received")
        cls.graph8.add_node(8_2, type="task", name="document complaint")
        cls.graph8.add_node(8_4, type="task", name="send gift card")
        cls.graph8.add_node(8_5, type="task", name="document gift")
        cls.graph8.add_node(8_6, type="intermediateCatchEvent", name="wait for feedback")
        cls.graph8.add_node(8_7, type="task", name="Document feedback")
        cls.graph8.add_node(8_8, type="endEvent", name="send docs to data analysis")
        cls.graph8.add_edges_from([(8_1, 8_2), (8_2, 8_4), (8_4, 8_5), (8_5, 8_6), (8_6, 8_7), (8_7, 8_8)],
                                  type=Flows.sequenceFlow.value)
        cls.graph9 = nx.DiGraph()
        cls.graph9.add_node(9_1, type="startEvent", name="Complaint received")
        cls.graph9.add_node(9_2, type="task", name="document complaint")
        cls.graph9.add_node(9_3, type="task", name="restore customer satisfaction")
        cls.graph9.add_node(9_4, type="task", name="send gift card")
        cls.graph9.add_node(9_5, type="task", name="document gift")
        cls.graph9.add_node(9_6, type="intermediateCatchEvent", name="wait for feedback")
        cls.graph9.add_node(9_7, type="task", name="Document feedback")
        cls.graph9.add_node(9_8, type="endEvent", name="send docs to data analysis")
        cls.graph9.add_edges_from([(9_1, 9_2), (9_2, 9_3), (9_3, 9_4), (9_4, 9_5), (9_5, 9_6), (9_6, 9_7), (9_7, 9_8)],
                                  type=Flows.sequenceFlow.value)
        cls.graph10 = nx.DiGraph()
        cls.graph10.add_node(10_1, type="startEvent", name="Complaint received")
        cls.graph10.add_node(10_2, type="task", name="document complaint")
        cls.graph10.add_node(10_3, type="task", name="restore customer satisfaction")
        cls.graph10.add_node(10_4, type="task", name="send gift card")
        cls.graph10.add_node(10_5, type="task", name="document gift")
        cls.graph10.add_node(10_6, type="intermediateCatchEvent", name="wait for feedback")
        cls.graph10.add_node(10_7, type="task", name="Document feedback")
        cls.graph10.add_node(10_8, type="endEvent", name="send docs to data analysis")
        cls.graph10.add_edges_from([(10_1, 10_2), (10_2, 10_3), (10_3, 10_2), (10_3, 10_4), (10_4, 10_5), (10_5, 10_6), (10_6, 10_7), (10_7, 10_8)],
                                  type=Flows.sequenceFlow.value)



    @classmethod
    def tearDownClass(cls):
        del cls.graph1
        del cls.graph2
        del cls.graph3
        del cls.graph4
        del cls.graph5
        del cls.graph6
        del cls.graph7
        del cls.graph8
        del cls.graph9

    def test_dependency_graph_similarity(self):
        sim = dependency_graph_similarity()
        self.assertEqual(0, sim(self.graph7, self.graph7a))
        sim = dependency_graph_similarity()
        self.assertEqual(3/7, sim(self.graph8, self.graph9))

    def test_causal_footprint_similarity(self):
        sim = causal_footprint_similarity()
        self.assertAlmostEqual(1.0, sim(self.graph7, self.graph7a),  5, "both graphs are the same")
        sim2 = causal_footprint_similarity()
        self.assertAlmostEqual(1.0, sim2(self.graph8, self.graph9), 5, "added nodes are not considered")
        sim2 = causal_footprint_similarity()
        self.assertAlmostEqual(0.997558, sim2(self.graph9, self.graph10), 5, "the similarity is 0.997558")
