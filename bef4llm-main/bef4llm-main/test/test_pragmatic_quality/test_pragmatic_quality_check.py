import math
import os.path
import unittest
import sys

import networkx as nx

from bef4llm.resource_controller.path_helper import look_for_directory
from bef4llm.process_models.importer.bpmn_importer import load_bpmn_from_directory
from bef4llm.definitions import Pragmatic_Subgroups as prag_mis_group
from bef4llm.definitions import Pragmatic_Metrics as prag_mis_metr
from bef4llm.pragmatic_quality.pragmatic_quality_check import PragmaticQualityCheckBPMN

class TestPragmaticQualityCheckBPMN(unittest.TestCase):

    def init_syntax_check_camuda(self, camuda_dir):
        path = look_for_directory(camuda_dir)
        path = os.path.join(path, "03-Solution")
        model = load_bpmn_from_directory(path, recursive=False, randomize_ids=False)
        pragmatic_quality_checker = PragmaticQualityCheckBPMN(model[0])
        return pragmatic_quality_checker

    def test_compute_size_metrics(self):
        pragmatic_quality_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        pragmatic_quality_checker.compute_size_metrics()

        self.assertEqual(1, pragmatic_quality_checker.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_gateways.value], "The model has one gateway")
        self.assertEqual(30, pragmatic_quality_checker.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_nodes.value], "The model has 30 nodes")
        self.assertEqual(28, pragmatic_quality_checker.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_sequence_flows.value], "The model has 28 sequence flows")
        self.assertEqual(10, pragmatic_quality_checker.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_message_flows.value], "The model has 10 message flows")
        self.assertEqual(12, pragmatic_quality_checker.metric_scores[prag_mis_group.size.value][prag_mis_metr.diameter.value], "The diameter is 11")

    def test_compute_density_metrics(self):
        pragmatic_quality_checker = self.init_syntax_check_camuda(camuda_dir="04-Self-service-restaurant")
        pragmatic_quality_checker.compute_density_metrics()

        self.assertEqual(28 / (30 * (30 - 1)), pragmatic_quality_checker.metric_scores[prag_mis_group.density.value][prag_mis_metr.density.value],
                         "Density: 28 / (30 * (30 - 1)")
        self.assertEqual((1 / 1) * 4, pragmatic_quality_checker.metric_scores[prag_mis_group.density.value][prag_mis_metr.average_gateway_degree.value],
                         "Average Gateway Degree: (1 / 1) * 4")
        self.assertEqual(28 / 30, pragmatic_quality_checker.metric_scores[prag_mis_group.density.value][prag_mis_metr.connectivity_coefficient.value],
                         "Connectivity Coefficient: 28 / 30")

        pragmatic_quality_checker = self.init_syntax_check_camuda(camuda_dir="03-Credit-scoring")
        pragmatic_quality_checker.compute_density_metrics()
        self.assertEqual((1 / 4) * 12, pragmatic_quality_checker.metric_scores[prag_mis_group.density.value][prag_mis_metr.average_gateway_degree.value],
                         "Average Gateway Degree: (1 / 4) * 12")

    def test_compute_connector_interplay_metrics(self):
        pragmatic_quality_checker = self.init_syntax_check_camuda(camuda_dir="01-Dispatch-of-goods")#01-Dispatch-of-goods"
        pragmatic_quality_checker.compute_connector_interplay_metrics()

        self.assertEqual(-1 * (((2/6) * math.log(2/6, 3))
                              + ((3/6) * math.log(3/6, 3))
                              + (((1/6) * math.log(1/6, 3)))),
                         pragmatic_quality_checker.metric_scores[prag_mis_group.connector_interplay.value][prag_mis_metr.gateway_heterogeneity.value],
                         "wrong gateway heterogeneity")
        self.assertEqual(1 + 2 + ((2 ** 2) - 1), pragmatic_quality_checker.metric_scores[prag_mis_group.connector_interplay.value][prag_mis_metr.control_flow_complexity.value],
                         "wrong control-flow complexity")
        self.assertLess(0, pragmatic_quality_checker.metric_scores[prag_mis_group.connector_interplay.value][prag_mis_metr.cross_connectivity.value],
                         "wrong cross-connectivity")
        self.assertGreater(1, pragmatic_quality_checker.metric_scores[prag_mis_group.connector_interplay.value][prag_mis_metr.cross_connectivity.value],
                        "wrong cross-connectivity")

    def test_compute_partitionability_metrics(self):
        pragmatic_quality_checker = self.init_syntax_check_camuda(
            camuda_dir="01-Dispatch-of-goods")
        pragmatic_quality_checker.compute_partitionability_metrics()
        self.assertEqual(2/17, pragmatic_quality_checker.metric_scores[prag_mis_group.partionability.value][
            prag_mis_metr.sequentiality.value],
                         "Wrong sequentiality")
        self.assertEqual(3, pragmatic_quality_checker.metric_scores[prag_mis_group.partionability.value][prag_mis_metr.depth.value],
                         "wrong depth")

    def test_compute_concurrency_metrics(self):
        pragmatic_quality_checker = self.init_syntax_check_camuda(
            camuda_dir="01-Dispatch-of-goods")
        pragmatic_quality_checker.compute_concurrency_metrics()
        self.assertEqual(2, pragmatic_quality_checker.metric_scores[prag_mis_group.concurrency.value][prag_mis_metr.token_split.value],
                         "Wrong sequentiality")

    def test_pragmatic_quality_check(self):
        pragmatic_quality_checker = self.init_syntax_check_camuda(
            camuda_dir="01-Dispatch-of-goods")
        score = pragmatic_quality_checker.pragmatic_quality_check()
        self.assertLess(0, score)
        self.assertGreater(1, score)

    def test_compute_total_pragmatic_quality_score(self):
        pragmatic_quality_checker = self.init_syntax_check_camuda(camuda_dir="01-Dispatch-of-goods")
        scores = {
            prag_mis_group.size.value:
                {prag_mis_metr.total_numbers_of_nodes.value: 43,
                 prag_mis_metr.total_numbers_of_gateways.value: 6,
                 prag_mis_metr.total_numbers_of_sequence_flows.value: 19,
                 prag_mis_metr.total_numbers_of_message_flows.value: 1,
                 prag_mis_metr.diameter: 24},
            prag_mis_group.density.value:
                {prag_mis_metr.density.value: 0.35,
                 prag_mis_metr.average_gateway_degree.value: 3.8,
                 prag_mis_metr.connectivity_coefficient.value: 2.3},
            prag_mis_group.connector_interplay.value:
                {prag_mis_metr.gateway_heterogeneity.value: 0.9,
                 # "gm": [], -> take since in syntax?
                 prag_mis_metr.control_flow_complexity.value: 52,
                 prag_mis_metr.cross_connectivity.value: 0.007},
            prag_mis_group.partionability.value:
                {prag_mis_metr.seperatibility.value: 1,
                 prag_mis_metr.sequentiality.value: 0.2,
                 prag_mis_metr.depth.value: 0.4},
            prag_mis_group.cyclicity.value: {},
            prag_mis_group.concurrency.value:
                {prag_mis_metr.token_split.value: 0.1}
        }
        pragmatic_quality_checker.metric_scores = scores
        self.assertEqual(9.75 / 15, pragmatic_quality_checker.compute_total_pragmatic_quality_score())
