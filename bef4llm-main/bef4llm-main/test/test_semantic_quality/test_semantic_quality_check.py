import os.path
import unittest

from bef4llm.resource_controller.path_helper import look_for_directory
from bef4llm.process_models.importer.bpmn_importer import load_bpmn_from_directory
from bef4llm.semantic_quality.semantic_quality_check import SemanticQualityCheckBPMN
from bef4llm.definitions import Similarity_Groups, Similarity_Metrics
from bef4llm.semantic_quality.similarity.language_similarity.natural_language_similarity import semantic_similarity
from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language
import copy


class TestSemanticQualityCheckBPMN(unittest.TestCase):

    def init_semantic_check_camuda(self, camuda_dir):
        path = look_for_directory(camuda_dir)
        ref_path = os.path.join(path, "03-Solution")
        reference_model = load_bpmn_from_directory(ref_path, recursive=False, randomize_ids=False)[0]
        model_path = os.path.join(path, "02-Results")
        model = load_bpmn_from_directory(model_path, recursive=False, randomize_ids=False)[26]
        semantic_quality_checker = SemanticQualityCheckBPMN(model, reference_model, lang=Language.ENGLISH)
        return semantic_quality_checker

    def init_semantic_check_camuda_same_graph(self, camuda_dir):
        path = look_for_directory(camuda_dir)
        ref_path = os.path.join(path, "03-Solution")
        reference_model = load_bpmn_from_directory(ref_path, recursive=False, randomize_ids=False)[0]
        model = copy.deepcopy(reference_model)

        for node in reference_model.process_graph.nodes(data=True):
            model.process_graph.remove_node(node[0])
            model.process_graph.add_node(f"{node[0]}_2")
            for attr in node[1]:
                model.process_graph.nodes[f"{node[0]}_2"][attr] = node[1][attr]
            model.process_graph.nodes[f"{node[0]}_2"]["id"] = f"{node[0]}_2"

        for u, v in reference_model.process_graph.edges:
            model.process_graph.add_edge(f"{u}_2", f"{v}_2")
            for attr in reference_model.process_graph[u][v]:
                model.process_graph[f"{u}_2"][f"{v}_2"][attr] = reference_model.process_graph[u][v][attr]

        model.pools = {}
        for pool in reference_model.pools:
            model.pools[f"{pool}_1"] = reference_model.pools[pool]

        model.lanes = {}
        for lane in reference_model.lanes:
            model.lanes[f"{lane}_1"] = reference_model.lanes[lane]

        semantic_quality_checker = SemanticQualityCheckBPMN(model, reference_model, lang=Language.ENGLISH)
        return semantic_quality_checker

    def init_semantic_check_camuda_different_dir(self, camuda_dir1, camuda_dir2):
        path1 = look_for_directory(camuda_dir1)
        model1_path = os.path.join(path1, "03-Solution")
        model1 = load_bpmn_from_directory(model1_path, recursive=False, randomize_ids=False)[0]
        path2 = look_for_directory(camuda_dir2)
        model2_path = os.path.join(path2, "03-Solution")
        model2 = load_bpmn_from_directory(model2_path, recursive=False, randomize_ids=False)[0]
        semantic_quality_checker = SemanticQualityCheckBPMN(model1, model2, lang=Language.ENGLISH)
        return semantic_quality_checker

    def init_multiple_instances_camunda(self, camuda_dir, num_instances=10):
        path = look_for_directory(camuda_dir)
        ref_path = os.path.join(path, "03-Solution")
        reference_model = load_bpmn_from_directory(ref_path, recursive=False, randomize_ids=False)[0]
        model_path = os.path.join(path, "02-Results")
        semantic_checker = []
        for i in range(num_instances):
            model = load_bpmn_from_directory(model_path, recursive=False, randomize_ids=False)[i]
            semantic_quality_checker = SemanticQualityCheckBPMN(model, reference_model, lang=Language.ENGLISH)
            semantic_checker.append(semantic_quality_checker)

        return semantic_checker

    def get_model(self, camuda_dir):
        path = look_for_directory(camuda_dir)
        path = os.path.join(path, "03-Solution")
        model = load_bpmn_from_directory(path, recursive=False, randomize_ids=False)[0]
        return model


    def test_natrual_language_similarity(self):
        semantic_checker = self.init_semantic_check_camuda("01-Dispatch-of-goods")
        semantic_checker.natural_language_similarity()
        self.assertIsNotNone(semantic_checker.similarity_metrics)

        semantic_checker = self.init_semantic_check_camuda_same_graph("01-Dispatch-of-goods")
        semantic_checker.natural_language_similarity()
        self.assertEqual({Similarity_Groups.natural_language.value:
                              {Similarity_Metrics.label_sim_syntactic.value: 1.0,
                              Similarity_Metrics.label_sim_semantic.value: 1.0,
                              Similarity_Metrics.label_sim_context.value: 1.0},
                          Similarity_Groups.behaviour.value: {},
                          Similarity_Groups.graph_structure.value: {}}, semantic_checker.similarity_metrics)
        """
        semantic_checker = self.init_multiple_instances_camunda("02-Recourse", num_instances=25)
        for sm in semantic_checker:
            sm.natural_language_similarity()
            for metric in sm.similarity_metrics[Similarity_Groups.natural_language.value]:
                value = sm.similarity_metrics[Similarity_Groups.natural_language.value][metric]
                print(metric, value)
                self.assertGreaterEqual(1, value, "The similarity value is between 0 and 1")
                self.assertLessEqual(0, value, "The similarity value is between 0 and 1")
        """


    def test_structural_similarity(self):
        semantic_checker = self.init_semantic_check_camuda("01-Dispatch-of-goods")
        semantic_checker.strucural_similarity()
        self.assertIsNotNone(semantic_checker.similarity_metrics)

        semantic_checker = self.init_semantic_check_camuda_same_graph("01-Dispatch-of-goods")
        semantic_checker.strucural_similarity()
        self.assertEqual({Similarity_Groups.natural_language.value: {},
                          Similarity_Groups.graph_structure.value: {
                              Similarity_Metrics.graph_edit_distance.value: 1.0,
                              Similarity_Metrics.common_percentage.value: 1.0,
                          },
                          Similarity_Groups.behaviour.value: {}}, semantic_checker.similarity_metrics)

        semantic_checker = self.init_multiple_instances_camunda("02-Recourse", num_instances=25)
        for sm in semantic_checker:
            sm.strucural_similarity()
            for metric in sm.similarity_metrics[Similarity_Groups.graph_structure.value]:
                value = sm.similarity_metrics[Similarity_Groups.graph_structure.value][metric]
                self.assertGreaterEqual(1, value, "The similarity value is between 0 and 1")
                self.assertLessEqual(0, value, "The similarity value is between 0 and 1")


    def test_behavioural_similarity(self):
        semantic_checker = self.init_semantic_check_camuda("01-Dispatch-of-goods")
        semantic_checker.behavioural_similarity()
        self.assertIsNotNone(semantic_checker.similarity_metrics)

        semantic_checker = self.init_semantic_check_camuda_same_graph("01-Dispatch-of-goods")
        semantic_checker.behavioural_similarity()
        print({Similarity_Groups.natural_language.value: {},
                          Similarity_Groups.graph_structure.value: {},
                          Similarity_Groups.behaviour.value: {
                              Similarity_Metrics.dependency_graph.value: 1.0,
                              Similarity_Metrics.causal_footprint.value: 1.0,
                          }})
        print(semantic_checker.similarity_metrics)
        self.assertEqual({Similarity_Groups.natural_language.value: {},
                          Similarity_Groups.graph_structure.value: {},
                          Similarity_Groups.behaviour.value: {
                              Similarity_Metrics.dependency_graph.value: 1.0,
                              Similarity_Metrics.causal_footprint.value: 0.9999999999999998,
                          }}, semantic_checker.similarity_metrics)

        semantic_checker = self.init_multiple_instances_camunda("02-Recourse", num_instances=25)
        for sm in semantic_checker:
            sm.behavioural_similarity()
            for metric in sm.similarity_metrics[Similarity_Groups.behaviour.value]:
                value = sm.similarity_metrics[Similarity_Groups.behaviour.value][metric]
                self.assertGreaterEqual(1, value, "The similarity value is between 0 and 1")
                self.assertLessEqual(0, value, "The similarity value is between 0 and 1")

    def test_compute_similarity(self):
        semantic_checker = self.init_semantic_check_camuda_same_graph("01-Dispatch-of-goods")
        sim_score = semantic_checker.semantic_quality_check()
        print(semantic_checker.similarity_metrics)
        self.assertIsNotNone(semantic_checker.similarity_metrics)
        self.assertEqual(1.0, sim_score)


