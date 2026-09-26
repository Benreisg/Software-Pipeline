"""
Internal representation of BPMNs and EPCs
"""
import networkx as nx
from bef4llm.process_models.graph_representation.node_types import Event, Flows, Participant, Task


class ProcessModel(object):
    """
    Internal representation of a process model
    """

    def __init__(self, name):
        self.process_graph = nx.DiGraph()
        self.name = name

    def get_entry_nodes(self):
        """
        Computes all entry nodes of the graph

        Returns
        -------
        entry_nodes: list of networkx nodes
            all entry nodes
        """

        result = []
        for n in self.process_graph.nodes:
            if len(list(self.process_graph.predecessors(n))) == 0:
                result.append(n)

        return result

    def get_exit_nodes(self):
        """
        Computes all exit nodes of the graph

        Returns
        -------
        exit_nodes: list of networkx nodes
            all exit nodes
        """

        result = []
        for n in self.process_graph.nodes:
            if len(list(self.process_graph.successors(n))) == 0:
                result.append(n)

        return result

    def get_nodes_by_type(self, node_type):
        """
        Returns all nodes of the graph of a specific type

        Parameters
        ----------
        node_type: str
            type to tutor

        Returns
        -------
        result_nodes : list of networkx nodes
            all nodes of given type
        """
        result = []

        for node in self.process_graph.nodes:
            if node_type == self.process_graph.nodes[node]['type']:
                result.append((node, self.process_graph.nodes(data=True)[node]))

        return result


class CollaborationModel(ProcessModel):
    """
    Internal representation of BPMNs and EPCs
    """

    def __init__(self, name):
        """
        Initializes a BPMN
        """
        self.process_graph = nx.DiGraph()
        self.processes = {}
        self.name = name
        self.collaboration_id = ""
        self.pools = {}
        self.subprocesses = {}
        self.additional_nodes = []
        self.lanes = {}
        self.message_flows = {}

        self.start_marking = []
        self.sequence_flow_id_edge_mapping = dict()
        self.subgraph_sequence_flow = None

        super(CollaborationModel, self).__init__(name=self.name)

    def get_flow_by_id(self, unique_id):
        """
        Returns an edge of the graph with given id

        Parameters
        ----------
        unique_id : str
            id to tutor

        Returns
        -------
        edge: networkx edge
            edge with the given id

        """
        for edge in self.process_graph.edges(data=True):
            if edge[2]['id'] == unique_id:
                return edge

    def get_initial_marking(self):
        if not self.start_marking:
            self.start_marking = self.compute_start_marking()
        return self.start_marking

    def map_edges_to_id(self):
        self.sequence_flow_id_edge_mapping = {data["id"]: (u, v) for u, v, data in self.process_graph.edges(data=True)}
        self.message_flow_id_edge_mapping = {id: (self.message_flows[id]["sourceRef"], self.message_flows[id]["targetRef"])
                                            for id in self.message_flows}

    def compute_start_marking(self):
        """
        Computes the start marking of the graph
        Returns
        -------
        inital marking as a list of networkx edges
        """
        if not self.sequence_flow_id_edge_mapping:
            self.map_edges_to_id()

        marking = list()
        try:
            start_nodes = self.get_nodes_by_type(node_type=Event.startEvent.value)
        except:
            return None

        for node in start_nodes:
            #if "message_flows" not in node[1]:
            if not any(self.message_flow_id_edge_mapping[id][1] == node[0] for id in self.message_flows):
                edges = node[1]["outgoing"]
                for edge_id in edges:
                    u, v = self.sequence_flow_id_edge_mapping[edge_id]
                    edge_data = self.process_graph.get_edge_data(u, v)
                    if edge_data["type"] == "sequenceFlow":
                        marking.append(edge_id)
        return marking

    def compute_new_markings(self, current_marking, fired_edge):
        """
        Computes the new marking of the BPMN based on the current marking
        Parameters
        ----------
        currentmarking: current marking in the bpmn (only one in form of a list of edges)
        fired_edge: edge where the token should be fired

        Returns
        -------
        new marking(s) as a list of list of netwrokx edges
        """
        #change to global start
        # current_marking = [edge for edge in current_marking if edge != fired_edge]

        # fire edge and get new markings
        new_markings = []
        next_node = self.process_graph.nodes[self.sequence_flow_id_edge_mapping[fired_edge][1]]
        if len(current_marking) > 1:
            current_marking.remove(fired_edge)
        else:
            current_marking = []
        # check if edges goes to end event
        if next_node["type"] == Event.endEvent.name:
            return None
        new_marked_edges = next_node["outgoing"]
        for new_marked_edge in new_marked_edges:
            current_marking.append(new_marked_edge)
            new_markings.append(current_marking)
        return new_markings

    def is_end_marking(self, current_marking: list):
        """
        Checks if the current marking is an end marking
        Parameters
        ----------
        currentmarking

        Returns
        -------
        boolean indicating if the current marking is an end marking (all edges in marking point to end event)
        """
        for edge in current_marking:
            node = self.process_graph.nodes[self.sequence_flow_id_edge_mapping[edge][1]]
            if not node["type"] == Event.endEvent.name:
                return False

            return True

    def get_successor_nodes(self, node):
        """
        Returns the successor nodes of the current node
        Parameters
        -----------
        node : str
            node ID

        Returns
        -----------
        successor_nodes: list
            successor nodes of the current node (only id of nodes)

        """
        if not self.sequence_flow_id_edge_mapping:
            self.map_edges_to_id()

        successor_nodes = []
        if "outgoing" in self.process_graph.nodes[node]:
            edges = [edge for edge in self.process_graph.nodes[node]["outgoing"]
                 if edge in self.sequence_flow_id_edge_mapping]
        else:
            edges = []

        for edge_id in edges:
            # when edge has only source or only target
            if edge_id in self.sequence_flow_id_edge_mapping:
                u, v = self.sequence_flow_id_edge_mapping[edge_id]
                successor_nodes.append(v) if v not in successor_nodes else None
        return successor_nodes

    def get_predecessor_nodes(self, node):
        """
        Returns the predecessors nodes of the current node
        Parameters
        -----------
        node : str
            node ID

        Returns
        -----------
        predecessor_nodes: list
            predecessor nodes of the current node (only id of nodes)

        """
        if not self.sequence_flow_id_edge_mapping:
            self.map_edges_to_id()
        predecessor_nodes = []
        if "incoming" in self.process_graph.nodes[node]:
            edges = [edge for edge in self.process_graph.nodes[node]["incoming"]
                 if edge in self.sequence_flow_id_edge_mapping]
        else:
            edges = []

        for edge_id in edges:
            if edge_id in self.sequence_flow_id_edge_mapping:
                u, v = self.sequence_flow_id_edge_mapping[edge_id]
                predecessor_nodes.append(u) if u not in predecessor_nodes else None
        return predecessor_nodes

    def get_pool_id_for_node(self, node):
        """
        Returns the id of the Pool the node is in. None if the node is not connected to a pool
        Parameters
        -----------
        node : str
            node ID

        Returns
        -----------
        pool: str
            id of pool

        """
        try:
            if node in self.process_graph.nodes:
                #print("node", self.process_graph.nodes[node])
                process = self.process_graph.nodes[node]["process"]
                #print("process", self.processes[process]["pool"])
                if self.processes[process]["pool"][0] in self.pools:
                    pool = self.processes[process]["pool"]
                else:
                    lane = self.lanes[self.processes[process]["pool"]]
                    pool = self.lanes[lane]["pool"]
            elif node in self.lanes:
                #print("lane")
                pool = self.lanes[node]["pool"]
            elif node in self.pools:
                #print("pool")
                pool = node
            elif node in self.processes:
                #print("process")
                pool = self.processes[node]["pool"]
            elif node in self.subprocesses:
                #print("subprocess")
                pool = self.subprocesses[node]["pool"]
            else:
                return None

            if not isinstance(pool, list):
                return [pool]
            else:
                return pool

        except KeyError:
            return None


    def get_element_type(self, node):
        """
        Returns the the type of an element (including subprocesses, participants, process, message_flow and sequence_flow)
        Parameters
        -----------
        node : str
            node ID

        Returns
        -----------
        successor_nodes: list
            successor nodes of the current node (only id of nodes)

        """
        if node in self.process_graph.nodes:
            type = self.process_graph.nodes[node]["type"] if "type" in self.process_graph.nodes[node] else None
            return type

        if node in self.lanes:
            return Participant.lane.value

        if node in self.pools:
            return Participant.pool.value

        if node in self.processes:
            return Task.process

        if node in self.subprocesses:
            return Task.subprocess

        if node in self.message_flows:
            return Flows.messageFlow

        if node in self.sequence_flow_id_edge_mapping:
            return Flows.sequenceFlow

    def init_subgraph_sequence_flows(self):
        """
        Initializes the subgraph where only sequence flows are used as edges (no message flows)
        """

        sequence_flow_edges = \
            [(u, v) for u, v, data in self.process_graph.edges(data=True) if
             data.get('type') == "sequenceFlow"]
        self.subgraph_sequence_flow = self.process_graph.edge_subgraph(sequence_flow_edges)

    def add_subproccess_nodes_to_graph(self):
        """
        Adds all nodes of the subgraphs to the to process graph
        -> needed when iterating over all nodes
        """

        for subprocess_id in self.subprocesses:
            subprocess_pg = self.subprocesses[subprocess_id].process_graph
            for node in subprocess_pg.nodes:
                if node not in self.process_graph:
                    self.process_graph.add_node(node)
                for attr in subprocess_pg.nodes[node]:
                    self.process_graph.nodes[node][attr] = subprocess_pg.nodes[node][attr]



