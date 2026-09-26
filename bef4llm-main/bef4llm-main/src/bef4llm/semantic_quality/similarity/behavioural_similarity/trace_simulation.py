"""
TraceExtraction
"""

import copy
import itertools
import collections
import string
import random
from datetime import timedelta, datetime

import networkx as nx
import pandas as pd

from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel
from bef4llm.definitions import XES_Terminology


class EPCSyntaxError(Exception):
    """Raised when the input value is too small"""
    pass


class TraceExtractor(object):
    """"
    TraceExtraction
    """

    def __init__(self, model, connector=None):
        """
        Initializes the class
        Parameters
        ----------
        model: collaboration model
            the epc
        connector: string
            The connector type that you want to add when a task has multiple outgoing edges
            Xor: exclusive Gateway
            And: parallel Gateway
            Or: inclusive Gateway
        """
        if not hasattr(model, 'process_graph'):
            model = CollaborationModel(name="sample_name")
            model.process_graph = model
            self.epc = model
        else:
            self.epc = model
        self.start_events = []
        self.gates = self.get_gates()
        self.connector = connector
        self.functions = self.get_functions()
        self.rdetection_list = []
        self.rtraces = []
        self.xorsplits = []
        self.xorjoins = []
        self.andsplits = []
        self.andjoins = []
        self.orsplits = []
        self.orjoins = []
        self.remove_events()
        self.operator_check()
        self.task_correction(self.connector)

        for g in self.gates:
            if "type" in g:
                if self.get_node_by_id(g)['type'] == "inclusiveGateway":
                    if len(list(self.epc.process_graph.predecessors(g))) > 1:
                        self.orjoins.append(g)
                    if len(list(self.epc.process_graph.successors(g))) > 1:
                        self.orsplits.append(g)
                if self.get_node_by_id(g)['type'] == "exclusiveGateway":
                    if len(list(self.epc.process_graph.predecessors(g))) > 1:
                        self.xorjoins.append(g)
                    if len(list(self.epc.process_graph.successors(g))) > 1:
                        self.xorsplits.append(g)
                if self.get_node_by_id(g)['type'] == "parallelGateway":
                    if len(list(self.epc.process_graph.predecessors(g))) > 1:
                        self.andjoins.append(g)
                    if len(list(self.epc.process_graph.successors(g))) > 1:
                        self.andsplits.append(g)

    def resolve_node(self, detection_list):
        """
            Resolves the detections according to the type of the current node
            Parameters
            ----------
            detection_list: dictionary list
                the list of detections

            Returns
            -------
            detection list
            the new detection list

            """
        new_detection_list = []
        entry = self.get_entry_nodes()
        for detection in detection_list:
            current_node = detection['current']
            ''' aktueller Knoten ist task -> an traces hängen, Nachfolger in possible packen'''
            # falls suc in trace , -> Loop, edge in edges packen

            if current_node in self.functions or current_node in entry:
                new_detection = copy.deepcopy(detection)
                new_detection['trace'].append(current_node)
                new_detection['trace_with'].append(current_node)
                new_detection['current'] = None
                new_detection['run'].append((new_detection['last'], current_node))
                new_detection['last'] = current_node

                succs = list(self.epc.process_graph.successors(current_node))
                for suc in succs:
                    if (current_node, suc) not in new_detection['edges']:
                        if suc in new_detection['trace_with']:
                            edge_list = self.get_loopy_edge(current_node, suc)
                            for e in edge_list:
                                new_detection['edges'].append(e)
                        if new_detection['possible'] != None:
                            new_detection['possible'].add(suc)
                        else:
                            new_detection['possible'] = {}
                            new_detection['possible'].add(suc)

                new_detection_list.append(new_detection)
                continue

            '''aktueller Knoten ist endevent oder endevent fehlt und current node ist None'''
            current_node_attr = self.get_node_by_id(current_node)
            current_node_type = current_node_attr['type'] if "type" in current_node_attr else None
            if current_node_type == 'endEvent' or current_node is None:
                trace = detection['trace_with']
                if trace not in self.rtraces:
                    self.rtraces.append(trace)

                continue

            '''aktueller Knoten ist XOR SPLIT, für jeden Nachfolger neue detection'''
            if current_node in self.xorsplits:
                new_detection = copy.deepcopy(detection)
                new_detection['current'] = None
                new_detection['run'].append((new_detection['last'], current_node))
                new_detection['last'] = current_node

                succs = list(self.epc.process_graph.successors(current_node))
                for suc in succs:
                    if (current_node, suc) not in new_detection['edges']:

                        d = copy.deepcopy(new_detection)
                        if suc in d['trace_with']:
                            edge_list = self.get_loopy_edge(current_node, suc)
                            for e in edge_list:
                                d['edges'].append(e)

                        d['possible'].add(suc)
                        d['trace_with'].append(current_node)
                        new_detection2 = {
                            'current': None,
                            'possible': d['possible'],
                            'trace_with': d['trace_with'],
                            'edges': d['edges'],
                            'trace': d['trace'],
                            'last': d['last'],
                            'run': d['run']

                        }
                        new_detection_list.append(new_detection2)
                continue

            '''aktueller Knoten ist XOR JOIN, Schauen ob schon eine Vorgänger ID in der trace ist, falls ja dann'''
            '''nicht weiter ,ansonsten warten bis nur noch dieser knoten in pos ist '''
            if current_node in self.xorjoins:

                new_detection = copy.deepcopy(detection)
                new_detection['current'] = None
                new_detection['run'].append((new_detection['last'], current_node))
                new_detection['last'] = current_node
                pres = list(self.epc.process_graph.predecessors(current_node))
                counter = 0
                for p in pres:
                    if p in new_detection['trace']:
                        counter += 1
                if counter <= 1:
                    result = False
                    for p in new_detection['possible']:
                        if self.is_reachable(p, current_node):
                            result = True
                    if not result:
                        suc = list(self.epc.process_graph.successors(current_node))
                        for s in suc:
                            new_detection['possible'].add(s)
                            new_detection['trace_with'].append(current_node)
                        new_detection_list.append(new_detection)
                continue

            '''aktueler Knoten ist AND Split -> neue detections mit allen nachfolgern'''
            if current_node in self.andsplits:
                new_detection = copy.deepcopy(detection)
                new_detection['current'] = None
                new_detection['run'].append((new_detection['last'], current_node))
                new_detection['last'] = current_node
                new_detection['trace_with'].append(current_node)
                sucs = list(self.epc.process_graph.successors(current_node))
                for s in sucs:
                    if (current_node, s) not in new_detection['edges']:
                        if s in new_detection['trace_with']:
                            edge_list = self.get_loopy_edge(current_node, s)
                            for e in edge_list:
                                new_detection['edges'].append(e)

                        new_detection['possible'].add(s)

                new_detection_list.append(new_detection)
                continue
            '''aktueler Knoten ist AND Join -> warten bis alle pfade angekommen sind'''
            if current_node in self.andjoins:

                new_detection = copy.deepcopy(detection)
                new_detection['current'] = None
                new_detection['run'].append((new_detection['last'], current_node))
                new_detection['last'] = current_node
                pres = list(self.epc.process_graph.predecessors(current_node))
                counter = 0
                for p in pres:
                    if (p, current_node) in new_detection['run']:
                        counter += 1

                if counter == len(pres):
                    suc = list(self.epc.process_graph.successors(current_node))
                    for s in suc:
                        new_detection['possible'].add(s)
                        new_detection['trace_with'].append(current_node)
                        new_detection_list.append(new_detection)
                    for p in pres:
                        try:
                            new_detection['run'].remove((p, current_node))
                        except ValueError:
                            continue

                if new_detection['possible']:
                    new_detection_list.append(new_detection)
                continue

            '''aktueler Knoten ist ODER Join -> Jeden Pfad weiterführen, dank der Struktur keine Probleme,'''
            '''aber warten bis pos nodes leer ist'''
            if current_node in self.orjoins:
                if detection['possible']:
                    continue

                new_detection = copy.deepcopy(detection)
                new_detection['current'] = None
                new_detection['run'].append((new_detection['last'], current_node))
                new_detection['last'] = current_node

                suc = list(self.epc.process_graph.successors(current_node))
                for s in suc:
                    new_detection['possible'].add(s)
                    new_detection['trace_with'].append(current_node)
                new_detection_list.append(new_detection)
                continue
            '''aktueler Knoten ist ODER Split -> Für jede Kombination der Nachfolger detection erstellen'''
            if current_node in self.orsplits:

                new_detection = copy.deepcopy(detection)
                new_detection['current'] = None
                new_detection['trace_with'].append(current_node)
                new_detection['run'].append((new_detection['last'], current_node))
                new_detection['last'] = current_node
                succs = list(self.epc.process_graph.successors(current_node))

                all_combi_list = self.all_list_variants(succs)
                for frozenset in all_combi_list:
                    legal = True
                    pos = set(frozenset)
                    for p in pos:
                        if (current_node, p) not in new_detection['edges']:
                            if p in new_detection['trace_with']:
                                edge_list = self.get_loopy_edge(current_node, p)
                                for e in edge_list:
                                    new_detection['edges'].append(e)
                        else:
                            legal = False

                    if legal:
                        new_detection2 = {'current': None,
                                          'possible': pos,
                                          'trace_with': new_detection['trace_with'],
                                          'edges': new_detection['edges'],
                                          'trace': new_detection['trace'],
                                          'last': new_detection['last'],
                                          'run': new_detection['run']
                                          }

                        new_detection_list.append(new_detection2)

        return new_detection_list

    def main(self, start_nodes):
        """
            Executes the trace extraction
            Parameters
            ----------
            start_nodes
                the start_nodes of the model

            Returns
            -------
            trace list
            the list of traces

        """
        print(start_nodes)

        detection_list = []
        '''Alle Kombinationen der Startnodes (für unterschiedliche erste Gatter) '''
        all_combi = self.all_list_variants(start_nodes)
        for frozenset in all_combi:
            pos = set(frozenset)

            detection = {'current': None,
                         'possible': pos,
                         'trace_with': [],
                         'edges': [],
                         'trace': [],
                         'last': None,
                         'run': []

                         }
            detection_list.append(detection)

        while True:
            detectionlist_tuple = self.next_node(detection_list)
            detection_list = detectionlist_tuple[1]
            something_done = detectionlist_tuple[0]
            if not something_done:
                for detection in detection_list:
                    trace = detection['trace_with']
                    if trace not in self.rtraces:
                        self.rtraces.append(trace)
                self.edit_traces()
                self.remove_duplicates()
                return self.rtraces
            detection_list = self.resolve_node(detection_list)

    def next_node(self, detection_list):
        """
            generates the next detection list and the next current nodes for each detection
            Parameters
            ----------
            detection_list
                the current detection list

            Returns
            -------
            detection list
            the new detection list

            """
        new_detection_list = []
        somethingdone = False
        for detection in detection_list:
            posnodes = detection['possible']
            if not posnodes:
                trace = detection['trace_with']
                if trace not in self.rtraces:
                    self.rtraces.append(detection['trace_with'])
            for posnode in posnodes:
                somethingdone = True
                n = copy.deepcopy(detection)
                n['possible'].remove(posnode)

                if n['possible'] == None:
                    n['possible'] = []
                new_detection = {
                    'current': posnode,
                    'possible': n['possible'],
                    'trace_with': n['trace_with'],
                    'edges': n['edges'],
                    'trace': n['trace'],
                    'run': n['run'],
                    'last': n['last']

                }
                new_detection_list.append(new_detection)

        if somethingdone:
            return [True, new_detection_list]
        else:
            return [False, detection_list]

    def remove_events(self):
        """
            Removes all useless events
        """
        # außer start/end knoten
        remove_list = []
        for node in self.epc.process_graph.nodes(data='type'):
            if self.get_super_type(node[1]) == 'event' and node[1] != 'startEvent' and node[1] != 'endEvent':
                pre = list(self.epc.process_graph.predecessors(node[0]))
                suc = list(self.epc.process_graph.successors(node[0]))
                for p in pre:
                    self.epc.process_graph.remove_edge(p, node[0])
                    for s in suc:
                        self.epc.process_graph.add_edge(p, s)
                for s in suc:
                    self.epc.process_graph.remove_edge(node[0], s)
                remove_list.append(node[0])
        for n in remove_list:
            self.epc.process_graph.remove_node(n)

    def edit_traces(self):
        """
        Removes all nodes that are no task

        """
        r_traces_new = []

        for trace in self.rtraces:
            new_trace = []
            for id_ in trace:
                node = self.get_node_by_id(id_)
                node_type = node["type"] if "type" in node else None
                if node_type and node_type == 'task':
                    new_trace.append(id_)
            r_traces_new.append(new_trace)
        self.rtraces = r_traces_new

    @classmethod
    def all_list_variants(cls, start_nodes_list):
        """
            generates all combinations of the IDs in a list
            Parameters
            ----------
            start_nodes_list
                the list with all start nodes

            Returns
            -------
            list
            list of all combinations of those Ids

        """

        liste = start_nodes_list
        f = []
        ret = set()
        for i in range(2, len(liste) + 1):
            g = list(itertools.combinations(liste, i))
            f.extend(g)
        for g in liste:
            f.append((g, g))
        for g in f:
            n = frozenset(g)

            ret.add(n)

        return ret

    def operator_check(self):
        """
        Handles the operator methods
        """
        self.granularity()
        self.reduce_edges()
        self.granularity()

    def is_reachable(self, id_, reach_node):
        """
        checks if a node is reachable for another node with given id
        Parameters
        ----------
        id_
            the id of the node
        reach_node
            the id of the node you want to reach

        Returns
        -------
        bool
        True if the node is reachable else False

        """

        set_ = nx.descendants(self.epc.process_graph, id_)
        result = False
        if reach_node in set_:
            result = True
        return result

    def remove_duplicates(self):
        """
        removes duplicates in traces

        """
        new = []
        for trace in self.rtraces:
            if trace not in new:
                new.append(trace)
        if new == [[]]:
            new = []
        self.rtraces = new

    def get_loopy_edge(self, currentnode, suc):
        """
        Gets the first edge that led to a loop
        Parameters
        ----------
        currentnode
            id of the current node
        suc
            id of the successor node

        Returns
        -------
        list
        the edge in a list

        """
        retlist = []
        succs = list(self.epc.process_graph.successors(currentnode))
        if len(succs) > 1:
            retlist.append((currentnode, suc))
            return retlist
        else:
            for p in list(self.epc.process_graph.predecessors(currentnode)):
                liste = self.get_loopy_edge(p, currentnode)

                retlist.extend(liste)
        return retlist

    def add_gate(self, connector, task):
        """
        Adds a gate after a task with multiple outgoing edges
        Parameters
        ----------
        task
            the task where its added
        connector
            the type of the connector you want to add
        """
        succ = list(self.epc.process_graph.successors(task))
        if len(succ) > 1:
            if self.connector is None:
                raise EPCSyntaxError

            type_ = ""
            id_ = self.get_new_node_id()

            if connector == "And":
                type_ = "parallelGateway"

            elif connector == "Or":
                type_ = "inclusiveGateway"
            elif connector == "Xor":
                type_ = "exclusiveGateway"
            self.gates.append(id_)
            self.epc.process_graph.add_node(id_, id=id_, type=type_)
            for s in succ:
                self.epc.process_graph.remove_edge(task, s)
                self.epc.process_graph.add_edge(id_, s)
            self.epc.process_graph.add_edge(task, id_)

    def task_correction(self, connector):
        """
        Adds a connector after a task with multiple outgoing edges
        Parameters
        ----------
        connector
            the type of the connector you want to add
        """

        for task in self.functions:
            self.add_gate(connector, task)
        for start in self.start_events:
            self.add_gate(connector, start)

    def granularity(self):
        """
        Checks if every operator has either only one incoming or outgoing edge and adds operators to reach that goal

        """
        appendlist = []
        removelist = []
        for gate in self.gates:
            gate_attr = self.get_node_by_id(gate)
            type_ = gate_attr["type"] if "type" in gate_attr else None
            pred = list(self.epc.process_graph.predecessors(gate))
            succ = list(self.epc.process_graph.successors(gate))
            if len(pred) == 1 and len(succ) == 1:
                '''wenn beides 1 ist, keine logik in OPerator vorhanden -> removen '''
                p = pred[0]
                s = succ[0]
                self.epc.process_graph.remove_edge(p, gate)
                self.epc.process_graph.remove_edge(gate, s)
                self.epc.process_graph.add_edge(p, s)
                self.epc.process_graph.remove_node(gate)
                removelist.append(gate)


            elif len(pred) == len(succ):

                '''neues gate wird join, alter split'''
                new_id = self.get_new_node_id()
                self.epc.process_graph.add_node(new_id, type=type_)
                for p in pred:
                    self.epc.process_graph.remove_edge(p, gate)
                    self.epc.process_graph.add_edge(p, new_id)
                self.epc.process_graph.add_edge(new_id, gate)
                appendlist.append(new_id)
            elif len(succ) != 1 and len(pred) > 2:
                '''neues gate wird join, alter split'''
                new_id = self.get_new_node_id()
                self.epc.process_graph.add_node(new_id, id=new_id, type=type_)
                for p in pred:
                    self.epc.process_graph.remove_edge(p, gate)
                    self.epc.process_graph.add_edge(p, new_id)
                self.epc.process_graph.add_edge(new_id, gate)
                appendlist.append(new_id)


            elif len(pred) != 1 and len(succ) > 2:
                new_id = self.get_new_node_id()
                self.epc.process_graph.add_node(new_id, type=type_)
                for s in succ:
                    self.epc.process_graph.remove_edge(gate, s)
                    self.epc.process_graph.add_edge(new_id, s)
                self.epc.process_graph.add_edge(gate, new_id)
                appendlist.append(new_id)
        for a in appendlist:
            self.gates.append(a)
        for a in removelist:
            self.gates.remove(a)

    def reduce_edges(self):
        """
        removes edges that point at the same successing node
        """

        def check_duplicates(n, succs):
            """
            Gets the first edge that led to a loop
            Parameters
            ----------
            n
                nodeID
            succs
                list of successors

            """
            duplicates = [item for item, count in collections.Counter(succs).items() if count > 1]
            if duplicates:
                for d in duplicates:
                    self.epc.process_graph.remove_edge(n, d)
                check_duplicates(n, succs)
            else:
                return

        nodes = list(self.epc.process_graph.nodes(data='type'))
        for n in nodes:
            id_ = n[0]
            succs = list(self.epc.process_graph.successors(id_))
            check_duplicates(id_, succs)

    def get_gates(self):
        """
        gets the gates of a model
        Returns
        -------
        list
        the gates

        """
        gates = []
        nodes = self.epc.process_graph.nodes(data='type')
        for n in nodes:
            if n[1] == "inclusiveGateway" or n[1] == "exclusiveGateway" or n[1] == "parallelGateway" \
                    or n[1] == "eventBasedGateway" or n[1] == "complexGateway":
                gates.append(n[0])
        return gates

    def get_functions(self):
        """
        gets the functions of a model
        Returns
        -------
        list
        the functions
        """
        functions = []
        nodes = self.epc.process_graph.nodes(data='type')
        for n in nodes:


            if n[1] == 'task' :
                functions.append(n[0])
            if n[1] == "startEvent":
                self.start_events.append(n[0])
        return functions

    def get_node_by_id(self, id_):
        """
        Gets the node object given the id
        Parameters
        ----------
        id_
            id of the node

        Returns
        -------
        node
        the node with that id

        """
        return self.epc.process_graph.nodes()[id_]

    def get_new_node_id(self):
        """
        Gets a new nodeID

        Returns
        -------
        string
        new node iD

        """
        ids = []
        nodes = self.epc.process_graph.nodes(data='type')
        for n in nodes:
            ids.append(n[0])

        def get_random_string(length):
            """
            Generates a random string
            Parameters
            -------
            length
                the length of a new string
            Returns
            -------
            string
            random string
            """
            letters = string.ascii_lowercase
            result_str = ''.join(random.choice(letters) for _ in range(length))
            return result_str

        int_ids = []
        for inte in self.epc.process_graph.nodes(data='id'):
            # if isinstance(inte[0],str):

            try:
                x = int(inte[0])
                int_ids.append(x)
            except ValueError:
                new_string = get_random_string(7)
                while new_string in ids:
                    new_string = get_random_string(7)

                return new_string

        maximum = max(int_ids)
        new_id = str(maximum + 1)
        return new_id

    def get_entry_nodes(self):
        """
        Computes all entry nodes of the graph

        Returns
        -------
        entry_nodes: list of networkx nodes
            all entry nodes
        """

        result = []
        for n in self.epc.process_graph.nodes:
            if len(list(self.epc.process_graph.predecessors(n))) == 0:
                result.append(n)

        return result

    @classmethod
    def get_super_type(cls, type_):
        """
        Gets the super type of a type

        Returns
        -------
        event:string
            if type is a event
        task:string
            if type is a function
        gate:string
            if type is a gate
        """
        tuplee = ["startEvent", "intermediateCatchEvent", "endEvent", "intermediateThrowEvent", "boundaryEvent",
                  "event"]
        if type_ in tuplee:
            return 'event'
        elif type_ == 'task':
            return 'task'
        else:
            return 'gate'

    def get_traces_as_pandas(self, traces):
        traces_df = []

        for caseID, trace in enumerate(traces, start=1):
            # Loop over each task in the current case
            for eventID, task in enumerate(trace, start=1):
                # Create a timestamp for each event (e.g., adding 1 hour per event as an example)
                timestamp = datetime.now()
                # Append a row of data
                traces_df.append({XES_Terminology.CASE_ID.value: caseID,
                                    XES_Terminology.TIMESTAMP.value: timestamp,
                                    XES_Terminology.EVENT_ID.value: task,
                                    XES_Terminology.RESOURCE.value: "user123"})
        return pd.DataFrame(traces_df)

class TraceExtractorBPMN(TraceExtractor):
    def __init__(self, model, connector=None):
        new_bpmn = copy.copy(model)
        print([(u, v, data) for u, v, data in model.process_graph.edges(data=True)])
        sequence_flow_edges = \
            [(u, v) for u, v, data in model.process_graph.edges(data=True) if
             data["id"] not in model.message_flows]

        new_bpmn.process_graph = nx.DiGraph(model.process_graph.edge_subgraph(sequence_flow_edges))
        print(new_bpmn.process_graph.nodes)
        super().__init__(new_bpmn, connector)
