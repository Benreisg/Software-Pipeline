import unittest

from bef4llm.llm_comparison import prepare_datasets, quality_check
from bef4llm.llm_comparison.quality_check import *
from bef4llm.definitions import thresholds_pragmatic_quality_metrics as prag_group_metric_mapping
from bef4llm.definitions import Pragmatic_Metrics, Sytax_Mistakes, Similarity_Metrics, Similarity_Groups


class TestCheckQuality(unittest.TestCase):
    def test_get_invalid_models(self):
        pass

    def test_quality_check_llms(self):
        datasets = dict()
        datasets["camunda"] = prepare_datasets.prepare_camunda()
        datasets["bpmn_and_text"] = prepare_datasets.prepare_text_and_bpmn()

        df_metric = quality_check.compute_overall_quality_llms(datasets=datasets,
                                                        test_llm_dir=f"{get_folder_path(Folder.TEST)}/test_benchmark/test_bpmn/testdata_llm_quality_check",

                                                        target_file=f"{get_folder_path(Folder.TEST)}/test_benchmark/test_bpmn/res_metruic.csv",
                                                        analyse_method="metric_score")

        df_detail = quality_check.compute_overall_quality_llms(datasets=datasets,
                                                        test_llm_dir=f"{get_folder_path(Folder.TEST)}/test_benchmark/test_bpmn/testdata_llm_quality_check",

                                                        target_file=f"{get_folder_path(Folder.TEST)}/test_benchmark/test_bpmn/res_metruic.csv",
                                                        analyse_method="detail")

        df_quality_group = quality_check.compute_overall_quality_llms(datasets=datasets,
                                                               test_llm_dir=f"{get_folder_path(Folder.TEST)}/test_benchmark/test_bpmn/testdata_llm_quality_check",
                                                               target_file=f"{get_folder_path(Folder.TEST)}/test_benchmark/test_bpmn/res_metruic.csv",
                                                               analyse_method="quality_group_score")

        # test pragmatic quality
        scores_group = []
        for subgroup in prag_group_metric_mapping:
            scores_subgroup = []
            for metric in prag_group_metric_mapping[subgroup]:
                metric_score = df_metric[metric].loc[df_metric.index[0]]
                scores_subgroup.append(metric_score)
                scores_group.append(metric_score)

            if scores_subgroup == []:
                continue
            self.assertAlmostEqual(mean(scores_subgroup), df_detail[subgroup].loc[df_detail.index[0]], places=6)
        self.assertAlmostEqual(mean(scores_group), df_quality_group["pragmatic quality"].loc[df_quality_group.index[0]], places=6)

        # test syntactic quality
        self.assertEqual(df_detail["syntactic quality"].loc[df_detail.index[0]], df_quality_group["syntactic quality"].loc[df_quality_group.index[0]])
        metric_scores = [df_metric[metric.value].loc[df_metric.index[0]] for metric in Sytax_Mistakes]
        print(metric_scores)
        self.assertAlmostEqual(mean(metric_scores), df_quality_group["syntactic quality"].loc[df_quality_group.index[0]])

        # test semantic quality
        sem_metric_group_mapping = {
            Similarity_Groups.natural_language.value: [Similarity_Metrics.label_sim_context.value,
                                                       Similarity_Metrics.label_sim_semantic.value,
                                                       Similarity_Metrics.label_sim_syntactic.value,],
            Similarity_Groups.graph_structure.value:[Similarity_Metrics.graph_edit_distance.value,
                                                     Similarity_Metrics.common_percentage.value,],
            Similarity_Groups.behaviour.value:[Similarity_Metrics.causal_footprint.value,
                                               Similarity_Metrics.dependency_graph.value,]
        }

        print(df_quality_group)
        scores_group = []
        for subgroup in sem_metric_group_mapping:
            scores_subgroup = []
            for metric in sem_metric_group_mapping[subgroup]:
                metric_score = df_metric[metric].loc[df_metric.index[0]]
                scores_subgroup.append(metric_score)
                scores_group.append(metric_score)

            if scores_subgroup == []:
                continue

            self.assertAlmostEqual(mean(scores_subgroup), df_detail[subgroup].loc[df_detail.index[0]], places=6)
        self.assertAlmostEqual(mean(scores_group), df_quality_group["semantic quality"].loc[df_detail.index[0]],
                                   places=6)
