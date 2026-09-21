import networkx as nx

from bef4llm.process_models.graph_representation.collaboration_model import ProcessModel, CollaborationModel
from bef4llm.process_models.graph_representation.node_types import Task, Event, Gateway, Participant, Flows
from bef4llm.definitions import Sytax_Mistakes
from bef4llm.process_models.collaboration_model_utils import get_indegree_node, get_outdegree_node, \
    is_reachable_with_sequence_flow, get_subprocess_node


class SyntaxCheck():
    """
   Class to check the syntactic quality of a process model
   """
    def __init__(self, model: CollaborationModel):
        self.model = model

    def syntax_check(self):
        pass

    def compute_syntax_score(self):
        pass


class SyntacticQualityCheckBPMN(SyntaxCheck):
    """
    Class to check the syntactic quality of a BPMN
    """
    def __init__(self, model: CollaborationModel):
        super().__init__(model)
        self.model = model
        self.syntax_errors = dict()  #set() # key: enum, value: (total number nodes, right number nodes)
        self.synatx_score = 0

        # subprocesses are treated as a part of the process graph, not checked indiviually
        model.add_subproccess_nodes_to_graph()

    def compute_syntax_score(self):
        """
        Computes the sytactic quality scores based on the previous computed number of syntax errors
            - some syntactical mistakes are treated as point measures -> only 0 or 1 as a measure
            - other syntactical mistakes are treated have a discrete score
                computed by number of the syntax error / the number of occurences of the element

        Parameters
        ----------

        Returns
        -------
        int
            overall score for the syntactic quality
        """

        syntax_score = 0
        for syntax_rule in Sytax_Mistakes.__members__:
            if syntax_rule in self.syntax_errors and self.syntax_errors[syntax_rule]["total"] > 0:
                syntax_score += self.syntax_errors[syntax_rule]["mistakes"] / self.syntax_errors[syntax_rule]["total"]

        return 1.0 - (syntax_score / len(Sytax_Mistakes.__members__))

    def syntax_check(self):
        """
        Computes the syntactic quality scores for a BPMN by checking all syntax rules.
        The sytax rules are checked for the following categories:
            - start and end event
            - task and event
            - gateways
            - message flow
            - sequence flow
        Parameters
        ----------

        Returns
        -------
        dict
            dictioinary of metrics scores
        """

        self.syntax_score = 0
        self.check_syntax_start_end_event()
        self.check_task_event_properties()
        self.check_number_processes_in_pool()
        self.check_gateway_properties()
        self.check_message_flow_properties()
        self.check_sequence_flow_properties()
        return self.compute_syntax_score()

    def syntax_check_metric_results(self):
        """
        Computes the sytactic quality scores for a BPMN by checking all syntax rules.
        The sytax rules are checked for the following categories:
            - start and end event
            - task and event
            - gateways
            - message flow
            - sequence flow
        Parameters
        ----------

        Returns
        -------
        dict
            dictionary of metrics scores
        """

        self.syntax_score = 0
        self.check_syntax_start_end_event()
        self.check_task_event_properties()
        self.check_number_processes_in_pool()
        self.check_gateway_properties()
        self.check_message_flow_properties()
        self.check_sequence_flow_properties()

        syntax_metrics_scores = dict()

        for syntax_rule in Sytax_Mistakes.__members__:
            if syntax_rule in self.syntax_errors and self.syntax_errors[syntax_rule]["total"] > 0:
                 syntax_metrics_scores[syntax_rule] = 1.0 - (self.syntax_errors[syntax_rule]["mistakes"] / self.syntax_errors[syntax_rule]["total"])
            else:
                syntax_metrics_scores[syntax_rule] = 1.0
        return syntax_metrics_scores

    def check_syntax_start_end_event(self):
        """
        Checks correct syntax for start and end events
        - Existence of start and end events: point measure
        - Number of start and end events: 1/number of events
        - in and outdegree for start and end events: 1/correct number of events

        """
        self.sources = [start_event for start_event, attribute in
                        self.model.process_graph.nodes(data=True) if attribute.get('type') == Event.startEvent.name]
        self.sinks = [end_event for end_event, attribute in
                      self.model.process_graph.nodes(data=True) if attribute.get('type') == Event.endEvent.name]
        self.syntax_errors[Sytax_Mistakes.one_start_event.value] = {"mistakes": 0, "total": 0}
        self.syntax_errors[Sytax_Mistakes.one_end_event.value] = {"mistakes": 0, "total": 0}

        # check start event
        if len(self.sources) > 0:
            process_ids = []
            for source in self.sources:
                self.syntax_errors[Sytax_Mistakes.one_start_event.value]["total"] += 1
                if self.model.process_graph.nodes[source]['process'] in process_ids:
                    self.syntax_errors[Sytax_Mistakes.one_start_event.value]["mistakes"] += 1
                else:
                    process_ids.append(self.model.process_graph.nodes[source]['process'])

                indegree = get_indegree_node(source, self.model.process_graph)
                outdegree = get_outdegree_node(source, self.model.process_graph)
                if outdegree < 1 or indegree > 0:
                    self.syntax_errors[Sytax_Mistakes.start_event_in_out_degree.value] = {"mistakes": 1, "total": 1}
        else:
            self.syntax_errors[Sytax_Mistakes.existence_start_event.value] = {"mistakes": 1, "total": 1}

        # check end event
        if len(self.sinks) > 0:
            process_ids = []
            for sink in self.sinks:
                self.syntax_errors[Sytax_Mistakes.one_end_event.value]["total"] += 1
                if self.model.process_graph.nodes[sink]['process'] in process_ids:
                    self.syntax_errors[Sytax_Mistakes.one_end_event.value]["mistakes"] += 1
                else:
                    process_ids.append(self.model.process_graph.nodes[sink]['process'])

                indegree = get_indegree_node(sink, self.model.process_graph)
                outdegree = get_outdegree_node(sink, self.model.process_graph)
                if outdegree > 0 or indegree < 1:
                    self.syntax_errors[Sytax_Mistakes.end_event_in_out_degree.value] = {"mistakes": 1, "total": 1}
                    break

        else:
            self.syntax_errors[Sytax_Mistakes.existence_end_event.value] = {"mistakes": 1, "total": 1}

    def check_task_event_properties(self):
        """
        Checks the correct syntrax for tasks and (intermediate) events
            - all nodes (so tasks, events and gateways) should be connected on a path between start and end event
            - Task and intermediate event (boundary events excluded) should have a indegree of 1 and a outdegree of 1
            - Boundary events should have a indegree of 0 and a outdegree of 1
            - each task should have a label
        """
        self.syntax_errors[Sytax_Mistakes.connected_nodes.value] = {"mistakes": 0, "total": 0}
        self.syntax_errors[Sytax_Mistakes.tasks_in_outdegree.value] = {"mistakes": 0, "total": 0}
        self.syntax_errors[Sytax_Mistakes.intermediate_event_in_out_degree.value] = {"mistakes": 0, "total": 0}
        self.syntax_errors[Sytax_Mistakes.labeled_tasks.value] = {"mistakes": 0, "total": 0}

        for node in self.model.process_graph.nodes:
            # Node is None, when e.g. a Flow is a source or target of another Flow
            node_type = self.model.process_graph.nodes[node]["type"] if "type" in self.model.process_graph.nodes[node] else None
            indegree = get_indegree_node(node, self.model.process_graph)
            outdegree = get_outdegree_node(node, self.model.process_graph)
            self.syntax_errors[Sytax_Mistakes.connected_nodes.value]["total"] += 1

            # check id nodes are connected
            if node_type == Event.startEvent.name:
                if outdegree < 1:
                    self.syntax_errors[Sytax_Mistakes.connected_nodes.value]["mistakes"] += 1
            elif node_type == Event.endEvent.name:
                if indegree < 1:
                    self.syntax_errors[Sytax_Mistakes.connected_nodes.value]["mistakes"] += 1
            else:
                if indegree == 0 or outdegree == 0:
                    # self.syntax_errors.add(Sytax_Mistakes.nodes_not_connected)
                    self.syntax_errors[Sytax_Mistakes.connected_nodes.value]["mistakes"] += 1

                # node is between start and end node
                elif len(self.sources) == 0 or len(self.sinks) == 0:
                    self.syntax_errors[Sytax_Mistakes.connected_nodes.value]["mistakes"] += 1
                else:
                    source_found = False
                    sink_found = False
                    for sink in self.sinks:
                        if is_reachable_with_sequence_flow(self.model, node, sink):
                            sink_found = True
                            break
                    for source in self.sources:
                        if is_reachable_with_sequence_flow(self.model, source, node):
                            source_found = True
                            break
                    if not source_found or not sink_found:
                        self.syntax_errors[Sytax_Mistakes.connected_nodes.value]["mistakes"] += 1

            if (node_type in Event.__members__
                    and node_type != Event.startEvent.name and node_type != Event.endEvent.name):
                # non-exception intermediate event
                self.syntax_errors[Sytax_Mistakes.intermediate_event_in_out_degree.value]["total"] += 1


                if node_type == Event.boundaryEvent.value:
                    if indegree != 0 or outdegree != 1:
                        self.syntax_errors[Sytax_Mistakes.intermediate_event_in_out_degree.value]["mistakes"] += 1
                if indegree != 1 or outdegree != 1:
                    self.syntax_errors[Sytax_Mistakes.intermediate_event_in_out_degree.value]["mistakes"] += 1

            if node_type in Task.__members__:
                self.syntax_errors[Sytax_Mistakes.tasks_in_outdegree.value]["total"] += 1
                self.syntax_errors[Sytax_Mistakes.labeled_tasks.value]["total"] += 1

                # check in and outdegree
                if indegree != 1 or outdegree != 1:
                    self.syntax_errors[Sytax_Mistakes.tasks_in_outdegree.value]["mistakes"] += 1

                node_name = self.model.process_graph.nodes[node]["name"]
                # check label
                if node_name == None or node_name == "":
                    self.syntax_errors[Sytax_Mistakes.labeled_tasks.value]["mistakes"] += 1

    def check_gateway_properties(self):
        """
        Checks the correct syntrax for gateways
            - each gateway split should have a corresponding join of the same type (exceptions as gateways are neglected)
            - a gateway should either be a split or join gateway
            - event split gateways must be followed by intermediate message or timer events or reveive task
        """

        gateway_types = [gateway.name for gateway in Gateway]
        gateways = [gateway for gateway, attribute in
                    self.model.process_graph.nodes(data=True) if attribute.get('type') in gateway_types]

        gateway_pairs = dict()
        not_matched_gateways = []
        join_gateways = []

        self.syntax_errors[Sytax_Mistakes.pair_gateways.value] = {"mistakes": 0, "total": 0}
        self.syntax_errors[Sytax_Mistakes.event_gateway_predecessor_successor_wrong.value] = {"mistakes": 0, "total": 0}
        self.syntax_errors[Sytax_Mistakes.gateway_in_out_degree.value] = {"mistakes": 0, "total": 0}
        for gateway in gateways:
            self.syntax_errors[Sytax_Mistakes.gateway_in_out_degree.value]["total"] += 1
            self.syntax_errors[Sytax_Mistakes.pair_gateways.value]["total"] += 1
            indegree = get_indegree_node(gateway, self.model.process_graph)
            outdegree = get_outdegree_node(gateway, self.model.process_graph)

            if (indegree > 1 and outdegree > 1) or (indegree <= 1 and outdegree <= 1):
                self.syntax_errors[Sytax_Mistakes.gateway_in_out_degree.value]["mistakes"] += 1

            if gateway not in gateway_pairs and gateway not in not_matched_gateways and indegree == 1 and outdegree >= 1:
                not_matched_gateways, gateway_pairs = self.find_corresponding_join_gateway_to_split_gateway(gateway,
                                                                                                            gateway_pairs,
                                                                                                            not_matched_gateways)
                # check if Eventbased Gateway id followed by message of timer event or receives a task
                if self.model.process_graph.nodes[gateway]["type"] == Gateway.eventBasedGateway.name:
                    self.syntax_errors[Sytax_Mistakes.event_gateway_predecessor_successor_wrong.value]["total"] += 1
                    predecessors = self.model.get_predecessor_nodes(gateway)
                    successors = self.model.get_successor_nodes(gateway)
                    event_successor = False
                    task_predecessor = False
                    for predecessor in predecessors:
                        predecessor_type = self.model.process_graph.nodes[predecessor]["type"]
                        task_predecessor = any(predecessor_type == item.value for item in Task)
                        if task_predecessor:
                            break

                    for successor in successors:
                        if "associated_event_types" in self.model.process_graph.nodes[successor]:
                            associated_event_type = self.model.process_graph.nodes[successor]["associated_event_types"]
                            event_successor = "messageEventDefinition" in associated_event_type or \
                                              "timerEventDefinition" in associated_event_type
                            if event_successor:
                                break

                    gateway_type = self.model.process_graph.nodes[gateway]["type"]
                    if gateway_type == Gateway.eventBasedGateway.name and not event_successor and not task_predecessor:
                        self.syntax_errors[Sytax_Mistakes.event_gateway_predecessor_successor_wrong.value][
                            "mistakes"] += 1

                if gateway not in gateway_pairs:
                    self.syntax_errors[Sytax_Mistakes.pair_gateways.value]["mistakes"] += 1

            else:
                join_gateways.append(gateway)

            if len(gateway_pairs) * 2 == len(gateways):
                break

        found_join_gateways = gateway_pairs.values()
        for join_gateway in join_gateways:
            if not join_gateway in found_join_gateways:
                self.syntax_errors[Sytax_Mistakes.pair_gateways.value]["mistakes"] += 1
                break

    def find_corresponding_join_gateway_to_split_gateway(self, gateway, gateway_pairs, not_matched_gateways,
                                                         visited=None):
        if not visited:
            visited = list()

        def get_not_visited_successor(successors, visited):
            for successor in successors:
                if successor not in visited:
                    return successor

            return None

        # make sure join and split is matched
        # make sure on order
        visited_suc = []
        if gateway in visited:
            not_matched_gateways.append(gateway)
            return not_matched_gateways, gateway_pairs

        visited.append(gateway)

        join_gateway = None
        gateway_type = self.model.process_graph.nodes[gateway]["type"]
        successor_type = None
        successor = self.model.get_successor_nodes(gateway)
        successor = successor[0] if successor != None and len(successor) > 0 else None
        visited_suc.append(successor)
        end = False

        # walk through the model on one arbitrary path until a fitting gateway is found
        if successor:
            while not end and not join_gateway:
                successor_type = self.model.process_graph.nodes[successor]["type"] if "type" in self.model.process_graph.nodes[successor] else None

                indegree_successor = get_indegree_node(successor, self.model.process_graph)
                outdegree_successor = get_outdegree_node(successor, self.model.process_graph)
                # found another split gateways, due to nesting the matching join gateway must be found before
                if successor_type == gateway_type and indegree_successor == 1 and outdegree_successor > 1:
                    not_matched_gateways, gateway_pairs = (self.find_corresponding_join_gateway_to_split_gateway
                                                           (successor, gateway_pairs, not_matched_gateways,
                                                            visited=visited))
                # fitting join gateway found
                if ((successor_type == gateway_type
                     or (gateway_type == Gateway.eventBasedGateway.name and
                         successor_type == Gateway.exclusiveGateway.name))
                        and indegree_successor > 1 and outdegree_successor == 1):
                    join_gateway = successor
                successors = self.model.get_successor_nodes(successor)
                successor = get_not_visited_successor(successors, visited_suc)
                visited_suc.append(successor)
                if not successor:
                    end = True

        if join_gateway:
            gateway_pairs[gateway] = join_gateway
        else:
            not_matched_gateways.append(gateway)
        return not_matched_gateways, gateway_pairs

    def check_message_flow_properties(self):
        """
        Checks the correct syntrax for message flowa
            - a message flow is only allowed between certain elements (see allowed sources and allowed targets)
            - elements need to be at least in different lanes
        """
        allowed_types_sources = [
            Event.intermediateThrowEvent.value,
            Event.endEvent.value,
            Participant.pool.value,
            Task.task.value,
            Task.subprocess.value
        ]

        allowed_types_target = [
            Event.startEvent.value,
            Event.intermediateCatchEvent.value,
            Event.boundaryEvent.value,
            Participant.pool.value,
            Task.task.value,
            Task.subprocess.value
        ]

        self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value] = {"mistakes": 0, "total": 0}
        for edge_id in self.model.message_flows:
            self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["total"] += 1
            source = self.model.message_flows[edge_id]["sourceRef"]
            target = self.model.message_flows[edge_id]["targetRef"]

            pool_source = self.model.get_pool_id_for_node(target)
            pool_target = self.model.get_pool_id_for_node(source)
            # check if elements are in the same pool/lanes
            if pool_source == None or pool_target == None:
                self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"] += 1
                continue
            else:
                for p_t in pool_target:
                    for s_t in pool_source:
                        if p_t == s_t:
                            self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"] += 1
                            continue

            source_type = self.model.get_element_type(source)
            target_type = self.model.get_element_type(target)

            # is the element allowed to be used in the context of the message flow
            if source_type not in allowed_types_sources or target_type not in allowed_types_target:
                self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"] += 1

            elif source_type == Event.intermediateThrowEvent.value or source_type == Event.startEvent.value:
                try:
                    if self.model.process_graph.nodes[source]["associated_event_types"][0]["type"] != "messageEventDefinition":
                        self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"] += 1
                except KeyError:
                    self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"] += 1

            elif target_type == Event.intermediateCatchEvent.value or target_type == Event.endEvent.value:
                try:
                    if self.model.process_graph.nodes[target]["associated_event_types"][0]["type"] != "messageEventDefinition":
                        self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"] += 1
                except KeyError:
                    self.syntax_errors[Sytax_Mistakes.wrong_message_flow.value]["mistakes"] += 1

    def check_sequence_flow_properties(self):
        """
           Checks the correct syntrax for sequence flows
               - a sequence flow is only allowed between certain elements (see allowed sources and allowed targets)
        """
        allowed_types_sources = [
            Event.intermediateThrowEvent.value,
            Event.intermediateCatchEvent.value,
            Event.boundaryEvent.value,
            Event.startEvent.value,
            Task.task.value,
            Task.subprocess.value,
            Gateway.exclusiveGateway.value,
            Gateway.parallelGateway.value,
            Gateway.eventBasedGateway.value,
            Gateway.inclusiveGateway.value,
        ]

        allowed_types_target = [
            Event.intermediateThrowEvent.value,
            Event.intermediateCatchEvent.value,
            Event.boundaryEvent.value,
            Event.endEvent.value,
            Task.task.value,
            Task.subprocess.value,
            Gateway.exclusiveGateway.value,
            Gateway.parallelGateway.value,
            Gateway.eventBasedGateway.value,
            Gateway.inclusiveGateway.value,
        ]

        self.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value] = {"mistakes": 0, "total": 0}
        for edge_id in self.model.sequence_flow_id_edge_mapping:
            self.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["total"] += 1
            source, target = self.model.sequence_flow_id_edge_mapping[edge_id]

            # focus on sequence flows
            if self.model.process_graph[source][target]["type"] != Flows.sequenceFlow.value:
                continue

            source_type = self.model.get_element_type(source)
            target_type = self.model.get_element_type(target)

            # check allowed types
            if source_type not in allowed_types_sources or target_type not in allowed_types_target \
                    or source_type == None or target_type == None:
                self.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"] += 1
                # if sequence flow is wrong for multiple reasons it should only be tracked once
                continue

            pool_source = self.model.get_pool_id_for_node(source)
            pool_target = self.model.get_pool_id_for_node(target)

            # check that elements are in the same pool
            if pool_target and pool_source:
                for p_t in pool_target:
                    for s_t in pool_source:
                        if p_t != s_t:
                            self.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"] += 1
                            continue

            elif not pool_source and pool_target or pool_source and not pool_target:
                self.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"] += 1
                continue

            # Same process
            if source_type in Event.__members__ or source_type in Task.__members__ \
                    or target_type in Event.__members__ or target_type in Task.__members__:
                if (self.model.process_graph.nodes[source]["process"]
                        != self.model.process_graph.nodes[target]["process"]):
                    self.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"] += 1
                    continue

            # boundary event
            if 'attachedToRef' in target:
                self.syntax_errors[Sytax_Mistakes.wrong_sequence_flow.value]["mistakes"] += 1

    def check_number_processes_in_pool(self):
        """
        Checks the correct syntrax for pools
            - per pool only one process is allowed
        """
        self.syntax_errors[Sytax_Mistakes.one_process_in_pool.value] = {"mistakes": 0, "total": 0}
        for process in self.model.processes:
            self.syntax_errors[Sytax_Mistakes.one_process_in_pool.value]["total"] += 1
            if "pool" in self.model.processes[process] and len(self.model.processes[process]["pool"]) > 1:
                self.syntax_errors[Sytax_Mistakes.one_process_in_pool.value]["mistakes"] += 1
