import unittest
import networkx as nx
import sys
sys.path.append("src")
from bef4llm.semantic_quality.similarity.graph_structure.structural_similarity import *
from bef4llm.semantic_quality.similarity.language_similarity.natural_language_similarity import semantic_similarity

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
        cls.graph10.add_node(10_1, name="Mama", type="task")
        cls.graph10.add_node(10_2, name="Mama", type="task")
        cls.graph11 = nx.DiGraph()
        cls.graph11.add_node(11_1, name="Mama", type="task")
        cls.graph11.add_node(11_2, name="Mama", type="task")
        cls.graph11.add_node(11_3, name="Uncle", type="task")
        cls.graph12 = nx.DiGraph()
        cls.graph12.add_node(12_1, name="Bama", type="task")
        cls.graph12.add_node(12_2, name="Rama", type="task")



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

    def test_common_percentage_sim(self):
        sim = common_percentage_similarity()
        percent = sim(self.graph6, self.graph7)
        self.assertEqual(percent, 0.6666666666666666)
        percent = common_percentage_similarity(nodes=True, edges=False, events=True, tasks=False, gateways=False)
        self.assertEqual(percent(self.graph8, self.graph8), 1.0)
        percent = common_percentage_similarity(nodes=True, edges=False, events=True, tasks=True, gateways=False)
        self.assertEqual(percent(self.graph7, self.graph7a), 1.0)
        percent = common_percentage_similarity(nodes=True, edges=False, events=True, tasks=False, gateways=False)
        self.assertEqual(percent(self.graph8, self.graph9), 1.0)
        percent = common_percentage_similarity(nodes=True, edges=False, events=True, tasks=True, gateways=False)
        self.assertEqual(percent(self.graph8, self.graph9), 0.9333333333333333)
        percent = common_percentage_similarity(nodes=True, edges=False, events=True, tasks=True, gateways=True)
        self.assertEqual(percent(self.graph8, self.graph9), 0.9333333333333333)
        percent = common_percentage_similarity(nodes=True, edges=True, events=True, tasks=True, gateways=True)
        self.assertEqual(percent(self.graph8, self.graph9), 0.8512820512820514)
        percent = common_percentage_similarity(nodes=True, edges=False, matching=None, threshold=0.4)
        self.assertEqual(percent(self.graph10, self.graph11), 0.8)
        matching = {111: [(101, 1.0), (102, 1.0)], 101: [(112, 1.0)], 102: [(111, 1.0), (112, 1.0)], 112: [(101, 1.0),
                                                                                                           (102, 1.0)]}
        percent = common_percentage_similarity(matching=matching)
        self.assertEqual(percent(self.graph10, self.graph11), 0.8)
        percent = common_percentage_similarity(nodes=False, edges=True)
        self.assertEqual(percent(self.graph9, self.graph9), 1)
        percent = common_percentage_similarity(nodes=False, edges=False)
        self.assertEqual(percent(self.graph9, self.graph9), 0)

    def test_graph_edit_dist(self):
        sim = graph_edit_distance()
        ged = sim(self.graph7, self.graph7a)
        self.assertEqual(ged, 0)
        # graph_edit_dist needs to be initialized again, otherwise the matching
        # from the previous call is transferred.
        sim2 = graph_edit_distance()
        ged2 = sim2(self.graph8, self.graph8)
        self.assertEqual(ged2, 0)
        ged3 = graph_edit_distance(events=True, tasks=True, gateways=True)
        self.assertEqual(ged3(self.graph8, self.graph9), 0.09914529914529914)
        ged4 = graph_edit_distance(events=True, tasks=False, gateways=True)
        self.assertEqual(ged4(self.graph8, self.graph9), 0)
        ged5 = graph_edit_distance(events=False, tasks=True, gateways=False)
        self.assertEqual(ged5(self.graph8, self.graph9), ((1/9) + (3/7) + 0) / 3)
        ged6 = graph_edit_distance(events=True, tasks=False, gateways=False)
        self.assertEqual(ged6(self.graph8, self.graph9), 0)
        match_sim = semantic_similarity()
        sim3 = graph_edit_distance(sim=match_sim)
        ged7 = sim3(self.graph8, self.graph2)
        self.assertEqual(ged7, 1)





if __name__ == '__main__':
    unittest.main()
