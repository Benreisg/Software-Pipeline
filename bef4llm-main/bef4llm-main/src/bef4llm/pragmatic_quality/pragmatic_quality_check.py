from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel
from bef4llm.process_models.graph_representation.node_types import Task, Event, Gateway, Participant, Flows
from bef4llm.pragmatic_quality import pragmatic_quality_metrics as metrics
from bef4llm.definitions import Pragmatic_Subgroups as prag_mis_group
from bef4llm.definitions import Pragmatic_Metrics as prag_mis_metr
from bef4llm.definitions import thresholds_pragmatic_quality_metrics


class PragmaticQualityCheck():
    """
    Class that checks the pragmatic quality of a process model
    """
    def __init__(self, model: CollaborationModel):
        self.model = model

    def pragmatic_quality_check(self):
        pass


class PragmaticQualityCheckBPMN(PragmaticQualityCheck):
    """
    Class that checks the pragmatic quality of a BPMN
    """
    def __init__(self, model: CollaborationModel):
        super().__init__(model)
        self.metric_scores = {
            prag_mis_group.size.value: {},
            prag_mis_group.density.value: {},
            prag_mis_group.connector_interplay.value: {},
            prag_mis_group.partionability.value: {},
            prag_mis_group.cyclicity.value: {},
            prag_mis_group.concurrency.value: {},
            prag_mis_group.other_metrics.value: {}
        }
        self.model.add_subproccess_nodes_to_graph()
        self.model.map_edges_to_id()
        self.threshold = thresholds_pragmatic_quality_metrics

    def pragmatic_quality_check(self):
        """
        Computes the pragmatic quality scores for a BPMN. The metrics are categorized in the seven groups:
            - Size
            - Density
            - Connector interplay
            - Partionability
            - Cyclicity
            - Concurrency
            - other metrics

        Parameters
        ----------

        Returns
        -------
        int
            overall score for the pragmatic quality of a BPMN
        """
        self.compute_size_metrics()
        self.compute_density_metrics()
        self.compute_connector_interplay_metrics()
        self.compute_partitionability_metrics()
        self.compute_cyclicity_metrics()
        self.compute_concurrency_metrics()
        self.compute_other_metrics()

        return self.compute_total_pragmatic_quality_score()

    def pragmatic_quality_check_detailed(self):
        """
        Computes the pragmatic quality scores for a BPMN. The metrics are categorized in the seven groups:
            - Size
            - Density
            - Connector interplay
            - Partionability
            - Cyclicity
            - Concurrency
            - other metrics

        Parameters
        ----------

        Returns
        -------
        dict
            dictionary of pragmatic quality scores for a BPMN
        """
        self.compute_size_metrics()
        self.compute_density_metrics()
        self.compute_connector_interplay_metrics()
        self.compute_partitionability_metrics()
        self.compute_cyclicity_metrics()
        self.compute_concurrency_metrics()
        self.compute_other_metrics()

        return self.compute_score_per_group()

    def pragmatic_quality_check_metric_results(self):
        """
        Computes the pragmatic quality scores for a BPMN. The metrics are categorized in the seven groups:
            - Size
            - Density
            - Connector interplay
            - Partionability
            - Cyclicity
            - Concurrency
            - other metrics

        Parameters
        ----------

        Returns
        -------
        dict
            dictionary of pragmatic quality scores for a BPMN
        """
        self.compute_size_metrics()
        self.compute_density_metrics()
        self.compute_connector_interplay_metrics()
        self.compute_partitionability_metrics()
        self.compute_cyclicity_metrics()
        self.compute_concurrency_metrics()
        self.compute_other_metrics()
        return self.compute_score_per_metric()

    def compute_size_metrics(self):
        """
        Computes the metrics of categorized in the group prag_mis_group.size.value":
            - total numbers of gateways
            - total numbers of nodes
            - total numbers of sequence flows
            - total numbers of message flows
            - diameter
        """
        # graph nodes that have a type
        process_nodes = [node for node in self.model.process_graph.nodes if "type" in self.model.process_graph.nodes[node]]

        # Total number of Gateways (TNG)
        self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_gateways.value] = len(
            [node for node in process_nodes
             if any(self.model.process_graph.nodes[node]["type"] == item.value
                    for item in Gateway)])

        # Total Number of nodes (TNN)
        self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_nodes.value] = (
                len(process_nodes) + len(self.model.subprocesses))

        # Total Number of Sequence Flows (TNSF)
        self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_sequence_flows.value] = len(
            [e for e in self.model.process_graph.edges
             if self.model.process_graph.edges[e]["type"] == Flows.sequenceFlow.value])

        # Total Number of message Flows (TNMF)
        self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_message_flows.value] = len(self.model.message_flows)

        # Diameter
        self.metric_scores[prag_mis_group.size.value][prag_mis_metr.diameter.value] = metrics.diameter(
            self.model.process_graph)

    def compute_density_metrics(self):
        """
        Computes the metrics of categorized in the group prag_mis_group.density.value:
            - density
            - average gateway degree
            - connectivity coefficient
        """
        if not self.metric_scores[prag_mis_group.size.value]:
            self.compute_size_metrics()

        process_nodes = [node for node in self.model.process_graph.nodes if "type" in self.model.process_graph.nodes[node]]

        # density
        number_nodes = self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_nodes.value]
        self.metric_scores[prag_mis_group.density.value][prag_mis_metr.density.value] \
            = (self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_sequence_flows.value] /
               (number_nodes * (number_nodes - 1)))

        # average gateway degree (AGD)
        sum_in_out_degree = 0
        gateways = [node for node in process_nodes
                    if any(self.model.process_graph.nodes[node]["type"] == item.value for item in Gateway)]

        for gateway in gateways:
            sum_in_out_degree += len(self.model.process_graph.nodes[gateway]["incoming"])
            sum_in_out_degree += len(self.model.process_graph.nodes[gateway]["outgoing"])

        number_gateways = self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_gateways.value]
        if number_gateways > 0:
            self.metric_scores[prag_mis_group.density.value][prag_mis_metr.average_gateway_degree.value] \
                = (1 / number_gateways) * sum_in_out_degree
        else:
            self.metric_scores[prag_mis_group.density.value][prag_mis_metr.average_gateway_degree.value] \
                = 0

        # Connectivity coefficient (CNC)
        self.metric_scores[prag_mis_group.density.value][prag_mis_metr.connectivity_coefficient.value] = \
            self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_sequence_flows.value] / \
            self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_nodes.value]

    def compute_connector_interplay_metrics(self):
        """
        Computes the metrics of categorized in the group "connector interplay":
            - gateway heterogeneity
            - control-flow complexity
            - cross-connectivity
        """

        if not self.metric_scores[prag_mis_group.size.value]:
            self.compute_size_metrics()

        # gateway heterogeneity (GH)
        self.metric_scores[prag_mis_group.connector_interplay.value][
            prag_mis_metr.gateway_heterogeneity.value] = metrics.gateway_heterogeneity(
            self.model.process_graph)

        # gateway mismatch (MM)
        # self.metric_scores[prag_mis_group.connector_interplay.value]["mm"] = metrics.gateway_mismatch(self.model.process_graph)

        # Control-flow Complexity (CFC)
        self.metric_scores[prag_mis_group.connector_interplay.value][
            prag_mis_metr.control_flow_complexity.value] = metrics.control_flow_complexity(self.model.process_graph)

        # Cross_Connectivity (CC)
        self.model.init_subgraph_sequence_flows()
        self.metric_scores[prag_mis_group.connector_interplay.value][
            prag_mis_metr.cross_connectivity.value] = metrics.cross_connectivity(
            self.model.subgraph_sequence_flow,
            self.metric_scores[prag_mis_group.size.value][prag_mis_metr.total_numbers_of_nodes.value])

    def compute_partitionability_metrics(self):
        """
        Computes the metrics of categorized in the group "partitionability":
            - sequentiality
            - seperatibility
            - depth
        """
        if not self.metric_scores[prag_mis_group.size.value]:
            self.compute_size_metrics()
        # sequentiality (only take a look at the sequence flows???)
        self.metric_scores[prag_mis_group.partionability.value][
            prag_mis_metr.sequentiality.value] = metrics.sequentiality(self.model)

        # seperatibility
        self.metric_scores[prag_mis_group.partionability.value][
            prag_mis_metr.seperatibility.value] = metrics.seperatibility(self.model.process_graph,
                                                                         self.metric_scores[
                                                                             prag_mis_group.size.value][
                                                                             prag_mis_metr.total_numbers_of_nodes.value])
        # depth
        self.metric_scores[prag_mis_group.partionability.value][prag_mis_metr.depth.value] = metrics.depth(
            self.model.process_graph)

    def compute_cyclicity_metrics(self):
        pass

    def compute_concurrency_metrics(self):
        """
        Computes the metrics of categorized in the group "partitionability":
            - token split
        """
        # token split (ts)
        self.metric_scores[prag_mis_group.concurrency.value][prag_mis_metr.token_split.value] = metrics.token_splits(
            self.model.process_graph)

    def compute_other_metrics(self):
        pass

    def compute_total_pragmatic_quality_score(self):
        """
        Computes the overall score for the pragmatic quality

        Parameters
        ----------

        Returns
        -------
        int
            the quality score, as the average of the ranking of the scores,
            between 0 (worst result) and 1 (best result)
        """

        if not self.metric_scores:
            return None

        # check thrshold and match score
        score = 0
        num_metric = 0
        for group in self.metric_scores:
            for metric in self.metric_scores[group]:
                try:
                    score += self.get_rank(self.metric_scores[group][metric], self.threshold[group][metric])
                    num_metric += 1
                except KeyError:
                    continue
                    #print(f"No thresholds for metric {metric} found -> metric is not included in computation")
        return score / num_metric

    def compute_score_per_group(self):
        """
        Computes the score for each group in the pragmatic quality

        Parameters
        ----------

        Returns
        -------
        int
            the quality score, as the average of the ranking of the scores,
            between 0 (worst result) and 1 (best result)
        """
        group_scores = dict()

        if not self.metric_scores:
            return None

        # check threshold and match score
        for group in self.metric_scores:
            num_metric = 0
            score = 0
            for metric in self.metric_scores[group]:
                try:
                    score += self.get_rank(self.metric_scores[group][metric], self.threshold[group][metric])
                    num_metric += 1
                except KeyError:
                    continue
                    # print(f"No thresholds for metric {metric} found -> metric is not included in computation")
            if num_metric > 0:
                group_scores[group] = score / num_metric
            else:
                group_scores[group] = 0
        return group_scores

    def compute_score_per_metric(self):
        """
        Computes the score for each group in the pragmatic quality

        Parameters
        ----------

        Returns
        -------
        int
            the quality score, as the average of the ranking of the scores,
            between 0 (worst result) and 1 (best result)
        """
        metric_scores = dict()

        if not self.metric_scores:
            return None

        # check threshold and match score
        for group in self.metric_scores:
            for metric in self.metric_scores[group]:
                try:
                    metric_scores[metric] = self.get_rank(self.metric_scores[group][metric], self.threshold[group][metric])
                except KeyError:
                    continue
                    # print(f"No thresholds for metric {metric} found -> metric is not included in computation")

        return metric_scores

    def get_rank(self, score, list_thresholds: list):
        """
        helper function to get the rank for each computed score of the metric

        Parameters
        ----------
        score: score for one metric
        list_thresholds: list of thresholds for one metric, needed to rank the score

        Returns
        -------
        int
            ranked score between 0 (worst result) and 1 (best result)
            0.25 * ranking, whereby the ranking is between 0 (worst result) and 4 (best result)
            """
        min_to_max = True
        if list_thresholds[0] > list_thresholds[1]:
            min_to_max = False

        if min_to_max:
            rank = 4
        else:
            rank = 0
        for ts in list_thresholds:
            if min_to_max:
                if score < ts:
                    rank = list_thresholds.index(ts)
                    break
            else:
                if score > ts:
                    rank = list_thresholds.index(ts)
                    break
        try:
            rank_score = ((4 - rank) * 0.25)
        except ZeroDivisionError:
            rank_score = 0
        return rank_score