import os.path
import unittest

from bef4llm.resource_controller.path_helper import look_for_directory
from bef4llm.process_models.importer.bpmn_importer import load_bpmn_from_directory
import bef4llm.process_models.collaboration_model_utils as utils
from bef4llm.process_models.graph_representation.node_types import Task

class TestCollaborationUtils(unittest.TestCase):
    def init_syntax_check_camunda(self, camunda_dir):
        path = look_for_directory(camunda_dir)
        path = os.path.join(path, "03-Solution")
        model = load_bpmn_from_directory(path)
        return model[0]

    def test_get_successors_by_type(self):
        model = self.init_syntax_check_camunda("01-Dispatch-of-goods")
        succ = utils.get_successor_by_type(graph=model.process_graph, node="Task_0vaxgaa")
        self.assertEqual(["ExclusiveGateway_1mpgzhg"], succ)
        succ = utils.get_successor_by_type(graph=model.process_graph, node="Task_0vaxgaa",
                                           types=[t for t in Task.__members__])
        self.assertCountEqual(["Task_0e6hvnj", "Task_0jsoxba", "Task_12j0pib"], succ)
        succ = utils.get_successor_by_type(graph=model.process_graph, node="Task_0sl26uo",
                                           types=[t for t in Task.__members__])
        self.assertCountEqual([], succ)

    def test_get_predecessors_by_type(self):
        model = self.init_syntax_check_camunda("01-Dispatch-of-goods")
        pre = utils.get_predecessor_by_type(graph=model.process_graph, node="Task_0vaxgaa")
        self.assertEqual(["ParallelGateway_02fgrfq"], pre)
        pre = utils.get_predecessor_by_type(node="Task_0sl26uo", graph=model.process_graph,
                                             types=[t for t in Task.__members__])
        self.assertCountEqual(["Task_0s79ile", "Task_0jsoxba", "Task_12j0pib", "Task_05ftug5"], pre)
        pre = utils.get_predecessor_by_type(graph=model.process_graph, node="Task_0vaxgaa",
                                           types=[t for t in Task.__members__])
        self.assertCountEqual([], pre)

