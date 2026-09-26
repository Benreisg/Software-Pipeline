from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel
from bef4llm.process_models.collaboration_model_utils import get_number_of_nodes_by_type
from bef4llm.process_models.graph_representation.node_types import Task, Event, Participant

from bef4llm.semantic_quality.similarity.language_similarity.natural_language_similarity import syntactic_similarity, semantic_similarity, context_similarity
from bef4llm.semantic_quality.similarity.similarity_utils import create_euquivalence_mapping_for_nodes, create_euquivalence_mapping_for_pools
from bef4llm.semantic_quality.similarity.graph_structure.structural_similarity import graph_edit_distance, common_percentage_similarity
from bef4llm.semantic_quality.similarity.behavioural_similarity.behavioural_similarity import causal_footprint_similarity, dependency_graph_similarity

from bef4llm.definitions import Similarity_Groups as sim_group
from bef4llm.definitions import Similarity_Metrics as sim_metric

from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language
from pandas import DataFrame


class SemanticQualityCheck():
    def __init__(self, model:CollaborationModel, reference_model:CollaborationModel):
        self.model = model
        self.reference_model = reference_model

    def semantic_quality_check(self):
        pass


class SemanticQualityCheckBPMN(SemanticQualityCheck):
    def __init__(self, model:CollaborationModel, reference_model:CollaborationModel, lang:Language):
        super().__init__(model, reference_model)
        self.similarity_metrics = {group.value: dict() for group in sim_group}
        self.lang = lang
        if self.lang == None:
            raise Exception("Language must be specified")
        self.model.add_subproccess_nodes_to_graph()


    def semantic_quality_check(self):
        """
        Computes the semantic quality score based on different metrics

        Returns
        -------
        int
            overall score for the semantic quality
        """
        self.natural_language_similarity()
        self.strucural_similarity()
        self.behavioural_similarity()

        score = 0
        num_metrics = 0

        for group in self.similarity_metrics:
            for metric in self.similarity_metrics[group]:
                score += self.similarity_metrics[group][metric]
                num_metrics += 1

        return score/num_metrics

    def semantic_quality_check_detailed(self):
        """
        Computes the semantic quality scores for each subgroup based on different metrics

        Returns
        -------
        dict
            dict with subgroup name as key and score as value
        """
        self.natural_language_similarity()
        self.strucural_similarity()
        self.behavioural_similarity()

        score_dict = dict()

        for group in self.similarity_metrics:
            score = 0
            num_metrics = 0
            for metric in self.similarity_metrics[group]:
                score += self.similarity_metrics[group][metric]
                num_metrics += 1
            score_dict[group] = score / num_metrics

        return score_dict

    def semantic_quality_check_metric_results(self):
        """
        Computes the semantic quality score based on different metrics

        Returns
        -------
        dict
            dict with metric name as key and score as value
        """

        self.natural_language_similarity()
        self.strucural_similarity()
        self.behavioural_similarity()

        score_metric_dict = dict()

        for group in self.similarity_metrics:
            for metric in self.similarity_metrics[group]:
                score_metric_dict[metric] = self.similarity_metrics[group][metric]

        return score_metric_dict

    def natural_language_similarity(self):
        """
        Computes the score for all natrual language similarity metrics
            - node matching with syntactic smilarity
            - node matching with semantic similarity
            - node matching with context similarity
        """

        # for events, tasks and Lanes
        type_node = list(Task.__members__) + list(Event.__members__)
        num_elements_model_1 = (get_number_of_nodes_by_type(self.model.process_graph, type_node) +
                             len(self.model.pools) + len([l for l in self.model.lanes if "name" in self.model.lanes[l]]))
        num_elements_model_2 = (get_number_of_nodes_by_type(self.reference_model.process_graph, type_node) +
                             len(self.reference_model.pools) + len([l for l in self.reference_model.lanes if "name" in self.reference_model.lanes[l]]))

        # syntactic similarity
        syntactic_sim = syntactic_similarity()
        syn_mapping_nodes = create_euquivalence_mapping_for_nodes(self.model.process_graph, self.reference_model.process_graph,
                                                                  sim=syntactic_sim, threshold=0.0)
        syn_mapping_participants = create_euquivalence_mapping_for_pools(self.model, self.reference_model,
                                                                 sim=syntactic_sim, threshold=0.0)

        sum_sim_syn = (sum([syn_mapping_nodes[node1][1] for node1 in syn_mapping_nodes]) +
                       sum([syn_mapping_participants[node2][1] for node2 in syn_mapping_participants]))


        self.similarity_metrics[sim_group.natural_language.value][sim_metric.label_sim_syntactic.value] = (
                (sum_sim_syn) / (num_elements_model_1 + num_elements_model_2))

        # semantic similarity
        semantic_sim = semantic_similarity(lang=self.lang)
        sem_mapping_nodes = create_euquivalence_mapping_for_nodes(self.model.process_graph, self.reference_model.process_graph,
                                                            sim=semantic_sim, threshold=0.0)
        sem_mapping_participants = create_euquivalence_mapping_for_pools(self.model, self.reference_model,
                                                            sim=semantic_sim, threshold=0.0)
        sum_sim_sem = (sum([sem_mapping_nodes[node1][1] for node1 in sem_mapping_nodes]) +
                       sum([sem_mapping_participants[node2][1] for node2 in sem_mapping_participants]))
        self.similarity_metrics[sim_group.natural_language.value][sim_metric.label_sim_semantic.value] = (
                (sum_sim_sem) / (num_elements_model_1 + num_elements_model_2))

        # compute context similarity based on previous matchings
        context_sim = context_similarity(set_of_graphs=[self.model.process_graph, self.reference_model.process_graph],
                                         mapping=sem_mapping_nodes)
        con_mapping = create_euquivalence_mapping_for_nodes(self.model.process_graph, self.reference_model.process_graph,
                                                            sim=context_sim, threshold=0.0)
        sum_sim_con = sum([con_mapping[node][1] for node in con_mapping])
        self.similarity_metrics[sim_group.natural_language.value][sim_metric.label_sim_context.value] = (
                (sum_sim_con) / (get_number_of_nodes_by_type(self.reference_model.process_graph, type_node)
                                 + get_number_of_nodes_by_type(self.model.process_graph, type_node)))


    def strucural_similarity(self, map_pool=False):
        """
        Computes the score for all behavioural similarity metrics
            - graph edit distance
            - number of equal nodes and edges
        """
        sem_sim = semantic_similarity(lang=self.lang)
        matching = create_euquivalence_mapping_for_nodes(self.model.process_graph, self.reference_model.process_graph, sem_sim, threshold=0.0)

        # graph edit distance
        ged = graph_edit_distance(matching=matching)
        self.similarity_metrics[sim_group.graph_structure.value][sim_metric.graph_edit_distance.value] = \
           1 - ged(self.model.process_graph, self.reference_model.process_graph)

        # number of equal nodes and edges
        sim_common_perc = common_percentage_similarity(matching=matching)
        self.similarity_metrics[sim_group.graph_structure.value][sim_metric.common_percentage.value] = \
            sim_common_perc(self.model.process_graph, self.reference_model.process_graph)


    def behavioural_similarity(self):
        """
        Computes the score for all behavioural similarity metrics
            - behavioural similarity with causal footprints
            - similarity based on a dependency graph
        """

        sem_sim = semantic_similarity(lang=self.lang)
        matching = create_euquivalence_mapping_for_nodes(self.model.process_graph, self.reference_model.process_graph, sem_sim, threshold=0.0)

        # behavioural similarity with causal footprints
        simdgs = dependency_graph_similarity(matching=matching)
        self.similarity_metrics[sim_group.behaviour.value][sim_metric.dependency_graph.value] = \
            1 - simdgs(self.model.process_graph, self.reference_model.process_graph)

        # similarity based on a dependency graph
        simcf = causal_footprint_similarity(matching=matching)
        self.similarity_metrics[sim_group.behaviour.value][sim_metric.causal_footprint.value] = \
            simcf(self.model.process_graph, self.reference_model.process_graph)
