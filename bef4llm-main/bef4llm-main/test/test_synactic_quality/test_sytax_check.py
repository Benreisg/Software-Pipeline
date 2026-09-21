import os.path
import unittest
import sys

import networkx as nx

from bef4llm.resource_controller.path_helper import look_for_file, look_for_directory
from bef4llm.process_models.importer.bpmn_importer import load_bpmn_from_directory, load_diagram_from_et_tree
from bef4llm.process_models.graph_representation.node_types import Task, Event, Gateway, Participant

from bef4llm.synactic_quality.synactic_quality_check import SyntacticQualityCheckBPMN
from bef4llm.synactic_quality.synactic_quality_check import Sytax_Mistakes

class TestSytaxCheck(unittest.TestCase):
    def init_syntax_check_camuda(self, camuda_dir):
        path = look_for_directory(camuda_dir)
        path = os.path.join(path, "03-Solution")
        model = load_bpmn_from_directory(path, recursive=False, randomize_ids=False)
        syntax_checker = SyntacticQualityCheckBPMN(model[0])

        syntax_checker.sources = [start_event for start_event, attribute in
                                  syntax_checker.model.process_graph.nodes(data=True) if
                                  attribute.get('type') == Event.startEvent.name]
        syntax_checker.sinks = [end_event for end_event, attribute in
                                syntax_checker.model.process_graph.nodes(data=True) if
                                attribute.get('type') == Event.endEvent.name]
        syntax_checker.model.map_edges_to_id()
        return syntax_checker

    """
    def is_error_in_list(self, error, error_list):
        for e in error_list:
            if e.name == error.name:
                return True
        return False
    """

    def test_check_syntax(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="03-Credit-scoring")
        score = syntax_checker.syntax_check()
        print(syntax_checker.syntax_errors)
        self.assertEqual(1, score, f"The model does not have any syntax errors, but the score is {score}")

    def test_check_syntax_start_end_event(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")

        # check removed sources and sinks
        syntax_checker.model.process_graph.remove_nodes_from(syntax_checker.sources)
        syntax_checker.model.process_graph.remove_nodes_from(syntax_checker.sinks)
        syntax_checker.check_syntax_start_end_event()
        self.assertEqual(1, syntax_checker.syntax_errors[Sytax_Mistakes.existence_start_event.value]["mistakes"],
                         "There is no start event")
        self.assertEqual(1, syntax_checker.syntax_errors[Sytax_Mistakes.existence_end_event.value]["mistakes"],
                         "There is no end event")

        # check in and outdegree of sources and sinks
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        syntax_checker.model.process_graph.add_edge(syntax_checker.sources[0], syntax_checker.sinks[0], type="sequenceFlow")
        syntax_checker.model.process_graph.add_edge(syntax_checker.sinks[0], syntax_checker.sources[0], type="sequenceFlow")
        syntax_checker.check_syntax_start_end_event()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.start_event_in_out_degree.value]["mistakes"],
            "The indegree of the start event is greater 0")
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.end_event_in_out_degree.value]["mistakes"],
                           "The outdegree of the end event is greater 0")

        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        source = syntax_checker.model.process_graph.nodes[syntax_checker.sources[0]]
        sink = syntax_checker.model.process_graph.nodes[syntax_checker.sinks[0]]
        syntax_checker.model.process_graph.remove_edge(syntax_checker.sources[0],
                                                       syntax_checker.model.sequence_flow_id_edge_mapping[source["outgoing"][0]][1])
        syntax_checker.model.process_graph.remove_edge(syntax_checker.model.sequence_flow_id_edge_mapping[sink["incoming"][0]][0],
                                                       syntax_checker.sinks[0])
        syntax_checker.check_syntax_start_end_event()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.start_event_in_out_degree.value]["mistakes"],
                           "The outdegree of the start event is 0")
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.end_event_in_out_degree.value]["mistakes"],
                           "The indegree of the end event is 0")

        # check number sources and sinks
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="03-Credit-scoring")
        syntax_checker.check_syntax_start_end_event()
        self.assertEqual(0, syntax_checker.syntax_errors[Sytax_Mistakes.one_start_event.value]["mistakes"],
                         "There are multiple start event in different process")
        self.assertEqual(0, syntax_checker.syntax_errors[Sytax_Mistakes.one_end_event.value]["mistakes"],
                         "There are multiple end event in different process")

        for source in syntax_checker.sources:
            syntax_checker.model.process_graph.nodes[source]["process"] = "process_1"

        for sink in syntax_checker.sinks:
            syntax_checker.model.process_graph.nodes[sink]["process"] = "process_1"

        syntax_checker.check_syntax_start_end_event()

        self.assertEqual(1, syntax_checker.syntax_errors[Sytax_Mistakes.one_start_event.value]["mistakes"],
                         "There are multiple start event in the same process")
        self.assertEqual(1, syntax_checker.syntax_errors[Sytax_Mistakes.one_end_event.value]["mistakes"],
                         "There are multiple end event in the same process")
    """
    def test_get_in_outdegree_sequence_flow_of_node(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        sources = [start_event for start_event, attribute in syntax_checker.model.process_graph.nodes(data=True)
                   if attribute.get('type') == Event.startEvent.name]
        indegree, outdegree = syntax_checker.get_in_outdegree_sequence_flow_of_node(sources[0])
        self.assertTrue(indegree == 0, "The indegree of the start event is 0")
        self.assertTrue(outdegree == 1, "The outdegree of the start event is 1")
    """

    def test_check_task_event_properties(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        nx.set_node_attributes(syntax_checker.model.process_graph, values={"Task_0a2rm9v": {"name": ""}})

        syntax_checker.model.process_graph.remove_edge("Task_0a2rm9v", "Task_12h2fs9")
        syntax_checker.model.process_graph.add_edge("IntermediateCatchEvent_1nu2fvu", "Task_0a2rm9v", type="sequenceFlow", id="new_edge_1")
        syntax_checker.model.process_graph.remove_edge("Task_1udyby3", "IntermediateCatchEvent_1nu2fvu")
        syntax_checker.model.process_graph.nodes["IntermediateCatchEvent_1nu2fvu"]["outgoing"].append("new_edge_1")
        syntax_checker.model.process_graph.nodes["IntermediateCatchEvent_1nu2fvu"]["incoming"] = []
        syntax_checker.check_task_event_properties()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.labeled_tasks.value]["mistakes"],
                           "The task Task_0a2rm9v has no label")
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.tasks_in_outdegree.value]["mistakes"],
                           "The task Task_0a2rm9v has an outdegree greater than 1 and an indegree of 0")
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.intermediate_event_in_out_degree.value]["mistakes"],
                           "The event IntermediateCatchEvent_1nu2fvu has an outdegree of 2 and an indegree greater than 1")
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.connected_nodes.value]["mistakes"],
                           "The end event is not reachable from event IntermediateCatchEvent_1nu2fvu")

        syntax_checker.model.process_graph.remove_edge("IntermediateCatchEvent_1nu2fvu", "Task_0a2rm9v")
        syntax_checker.check_task_event_properties()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.connected_nodes.value]["mistakes"],
                           "The task Task_0a2rm9v is unconnected")


    def test_check_gateway_properties_normal_gateways(self):
        # checks for XOR, OR and AND Gateways
        syntax_checker = self.init_syntax_check_camuda("01-Dispatch-of-goods")
        syntax_checker.check_gateway_properties()
        self.assertEqual(2, syntax_checker.syntax_errors[Sytax_Mistakes.pair_gateways.value]["mistakes"],
                           "Gateway ParallelGateway_02fgrfq has no matching gateway")


        nx.set_node_attributes(syntax_checker.model.process_graph,
                               values={"ParallelGateway_02fgrfq": {"type": Gateway.exclusiveGateway.name}})
        syntax_checker.check_gateway_properties()
        self.assertEqual(0, syntax_checker.syntax_errors[Sytax_Mistakes.pair_gateways.value]["mistakes"],
                          "All gateways have a corresponding gateway")


        syntax_checker.model.process_graph.add_node("new_start_node", type=Event.startEvent)
        syntax_checker.model.process_graph.add_edge("new_start_node", "ParallelGateway_02fgrfq",
                                                    type="sequenceFlow", id="new_edge_1")
        syntax_checker.check_gateway_properties()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.gateway_in_out_degree.value]["mistakes"],
                           "The gateway ParallelGateway_02fgrfq has multiple indegrees and outdegrees")


    """
    ToDo: Exceptions for gateways
    def test_check_gateway_properties_event_based_gateways(self):
        # checks for Eventbased gateways
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="03-Credit-scoring")
        syntax_checker.check_gateway_properties()
        print(syntax_checker.syntax_errors)
    """

    def test_find_corresponding_join_gateway_to_split_gateway(self):
        syntax_checker = self.init_syntax_check_camuda("01-Dispatch-of-goods")
        not_matched_gateways, gateway_pairs = (syntax_checker.find_corresponding_join_gateway_to_split_gateway
                                               (gateway="InclusiveGateway_0p2e5vq",
                                                gateway_pairs=dict(),
                                                not_matched_gateways=list()))

        self.assertIn("InclusiveGateway_0p2e5vq", gateway_pairs)
        self.assertEqual(gateway_pairs["InclusiveGateway_0p2e5vq"], "InclusiveGateway_1dgb4sg")

        not_matched_gateways, gateway_pairs = (syntax_checker.find_corresponding_join_gateway_to_split_gateway
                                               (gateway="ParallelGateway_02fgrfq",
                                                gateway_pairs=dict(),
                                                not_matched_gateways=list()))
        print(gateway_pairs, not_matched_gateways)
        self.assertIn("ParallelGateway_02fgrfq", not_matched_gateways)

    """
    def test_check_soundness(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="01-Dispatch-of-goods")
        syntax_checker.check_soundness()

        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.sound_process_model.value]["mistakes"],
                           "The gateway ParallelGateway_02fgrfq has multiple indegrees and outdegrees")

        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        syntax_checker.check_soundness()

        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.sound_process_model.value]["mistakes"],
                           "The gateway ParallelGateway_02fgrfq has multiple indegrees and outdegrees")
    """

    def test_check_message_flow_properties(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="03-Credit-scoring")
        syntax_checker.check_message_flow_properties()
        print(syntax_checker.syntax_errors)
        self.assertEqual(0, syntax_checker.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"])

        syntax_checker.model.process_graph.nodes["IntermediateCatchEvent_0a8iz14"]["type"] = Event.intermediateThrowEvent
        syntax_checker.check_message_flow_properties()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"])

        syntax_checker = self.init_syntax_check_camuda(camuda_dir="03-Credit-scoring")
        process = syntax_checker.model.process_graph.nodes["IntermediateCatchEvent_0a8iz14"]["process"]
        syntax_checker.model.processes[process]["pool"] = ["Participant_1x9zkso"]
        syntax_checker.check_message_flow_properties()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"])

    def test_check_sequence_flow_properties(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        syntax_checker.check_sequence_flow_properties()
        self.assertEqual(0, syntax_checker.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"])

        syntax_checker.model.process_graph.add_edge("Task_1kt8dzo", "Task_1y7mm27", id="new_edge_1", type="sequenceFlow")
        syntax_checker.model.map_edges_to_id()
        syntax_checker.check_sequence_flow_properties()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"])

        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        syntax_checker.model.process_graph.add_edge("Participant_10vg9sk", "Task_1y7mm27", id="new_edge_2", type="sequenceFlow")
        process = syntax_checker.model.process_graph.nodes["Task_1y7mm27"]["process"]
        syntax_checker.model.process_graph.nodes["Participant_10vg9sk"]["type"] = Participant.participant
        syntax_checker.model.process_graph.nodes["Participant_10vg9sk"]["process"] = process
        syntax_checker.model.map_edges_to_id()
        syntax_checker.check_sequence_flow_properties()
        self.assertLess(0, syntax_checker.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"])

    """
    def test_nullification(self):
        syntax_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant",
                                                       nullification=True,
                                                       null_elements={Event: 1000, Task: 1000, Gateway: 1000})
        score = syntax_checker.syntax_check()
        se = syntax_checker.syntax_errors
        self.assertEqual(0, (se[Sytax_Mistakes.tasks_in_outdegree.value]["mistakes"] /
                             se[Sytax_Mistakes.tasks_in_outdegree.value]["total"]) - 1)
        self.assertEqual(0, (se[Sytax_Mistakes.intermediate_event_in_out_degree.value]["mistakes"] /
                             se[Sytax_Mistakes.intermediate_event_in_out_degree.value]["total"]) - 1)
        self.assertEqual(0, (se[Sytax_Mistakes.labeled_tasks.value]["mistakes"] /
                             se[Sytax_Mistakes.labeled_tasks.value]["total"]) - 1)
        self.assertEqual(0, (se[Sytax_Mistakes.pair_gateways.value]["mistakes"] /
                             se[Sytax_Mistakes.pair_gateways.value]["total"]) - 1)
        self.assertEqual(0, (se[Sytax_Mistakes.event_gateway_predecessor_successor_wrong.value]["mistakes"] /
                             se[Sytax_Mistakes.event_gateway_predecessor_successor_wrong.value]["total"]) - 1)
        self.assertEqual(0, (se[Sytax_Mistakes.gateway_in_out_degree.value]["mistakes"] /
                             se[Sytax_Mistakes.gateway_in_out_degree.value]["total"]) - 1)
    """















