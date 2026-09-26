"""
Imports .bpmn files

Notes
_____
Based on [1] https://www.omg.org/spec/BPMN/2.0/
"""
from xml.etree import ElementTree as ET
from bef4llm.process_models.graph_representation.node_types import Gateway, Task, Event, Others
from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel
import os
import uuid



def load_bpmn_from_directory(directory, recursive=False, randomize_ids=False):
    """
    Loads a directory with .bpmn files and creates internal representations of the BPMNs (collaboration_model).
    Parameters
    ----------
    directory : str
        path to the directory
    recursive : boolean, optional
        if set true, imports subdirectories
    randomize_ids : bool, optional
        if set true, randomizes all id's used in the bpmns

    Returns
    -------
    bpmns : list of collaboration models
        imported BPMNs

    """
    bpmns = []

    for root, dirs, files in os.walk(directory):
        bpmns += [load_diagram_from_xml(os.path.join(root, f), randomize_ids) for f in files if f.endswith('.bpmn')]
        if not recursive:
            break
    return bpmns


def load_diagram_from_et_tree(tree:ET, randomize_ids=False, name="CollaborationModel"):
    """
        Loads a .bpmn files and creates internal representations of the BPMN (collaboration_model).
        Parameters
        ----------
        xml_string : str
            xml_string to be parsed
        randomize_ids : bool, optional
            if set true, randomizes all id's used in the bpmn
        name : str, optional
            name of the collaboration model

        Returns
        -------
        bpmn : collaboration_model
            imported BPMN

        """
    bpmn = CollaborationModel(name)

    for _, el in tree:
        prefix, has_namespace, postfix = el.tag.partition('}')
        if has_namespace:
            # strip all namespaces
            el.tag = postfix
    root = tree.root

    message_flows = []
    process_pool_map = {}
    id_map = {}

    for collaboration in root.findall('collaboration'):

        if "id" not in collaboration.attrib or collaboration.attrib == {} or not collaboration.attrib:
            collaboration_id = str(uuid.uuid4())
        else:
            collaboration_id = collaboration.attrib['id']


        bpmn.collaboration_id = collaboration_id

        # find all participants
        for participant in collaboration.findall('participant'):
            pool = {}
            pool_id = participant.attrib['id']
            for attribute in participant.attrib:
                pool[attribute] = participant.attrib[attribute]

            if 'processRef' in pool.keys():
                process_pool_map[pool['processRef']] = pool_id

            bpmn.pools[pool_id] = pool

        for message_flow in collaboration.findall('messageFlow'):
            message_flows.append(message_flow)

    # add processes
    for process in root.findall('process'):

        current = {}
        process_id = process.attrib['id']

        for attribute in process.attrib:
            current[attribute] = process.attrib[attribute]

        if process_id in process_pool_map:
            id_map.update(parse_process(bpmn, process_id, process_pool_map[process_id], process, randomize_ids))
            bpmn.processes[process_id] = current
            try:
                bpmn.processes[process_id]['pool'].append(process_pool_map[process_id])
            except KeyError:
                bpmn.processes[process_id]['pool'] = [process_pool_map[process_id]]
        else:
            id_map.update(parse_process(bpmn, process_id, None, process, randomize_ids))
            bpmn.processes[process_id] = current

    # add message flows
    for message_flow in message_flows:
        if not ('sourceRef' in message_flow.attrib and 'targetRef' in message_flow.attrib):
            continue

        edge_id = message_flow.attrib['id']

        source = message_flow.attrib['sourceRef']
        target = message_flow.attrib['targetRef']

        if randomize_ids:
            if source in id_map.keys():
                source = id_map[source]
            if target in id_map.keys():
                target = id_map[target]

        bpmn.message_flows[edge_id] = {'id': edge_id, 'sourceRef': source, 'targetRef': target}

        if bpmn.process_graph.has_node(source) and bpmn.process_graph.has_node(target):
            bpmn.process_graph.add_edge(source, target)
            bpmn.process_graph[source][target]["type"] = 'messageFlow'

            for attribute in message_flow.attrib:
                if attribute != 'sourceRef' and attribute != 'targetRef':
                    bpmn.process_graph[source][target][attribute] = message_flow.attrib[attribute]

        else:
            unspecific_message_flow = {}

            for attribute in message_flow.attrib:
                unspecific_message_flow[attribute] = message_flow.attrib[attribute]

            if bpmn.process_graph.has_node(source):
                # Node to Pool

                if 'message_flows' not in bpmn.process_graph.nodes[source]:
                    bpmn.process_graph.nodes[source]['message_flows'] = [unspecific_message_flow]
                else:
                    bpmn.process_graph.nodes[source]['message_flows'].append(unspecific_message_flow)

            else:
                # Nodes in subprocesses

                node_in_sub = False

                for sub_process_key in bpmn.subprocesses.keys():
                    sub_process = bpmn.subprocesses[sub_process_key]
                    if sub_process.process_graph.has_node(source):
                        node_in_sub = True
                        if 'message_flows' not in sub_process.process_graph.nodes[source]:
                            sub_process.process_graph.nodes[source]['message_flows'] = [unspecific_message_flow]
                        else:
                            sub_process.process_graph.nodes[source]['message_flows'].append(unspecific_message_flow)

                        break

                if not node_in_sub:
                    # Pool to Node or Pool to Pool
                    if 'message_flows' not in bpmn.pools[source]:
                        bpmn.pools[source]['message_flows'] = [unspecific_message_flow]
                    else:
                        bpmn.pools[source]['message_flows'].append(unspecific_message_flow)

    bpmn.map_edges_to_id()
    return bpmn

def load_diagram_from_xml(filepath, randomize_ids=False):
    """
    Loads a .bpmn files and creates internal representations of the BPMN (collaboration_model).
    Parameters
    ----------
    filepath : str
        path to the .bpmn file
    randomize_ids : bool, optional
        if set true, randomizes all id's used in the bpmn

    Returns
    -------
    bpmn : collaboration_model
        imported BPMN

    """
    filename, _ = os.path.splitext(os.path.basename(filepath))
    tree = ET.iterparse(filepath)
    return load_diagram_from_et_tree(tree=tree, randomize_ids=randomize_ids, name=filename)



def parse_lanes(bpmn, tag, node_lane_map, parent, pool_id):

    """
    Recursively parses lanes of a bpmn process and adds it to a existing collaboration model

    Parameters
    ----------
    bpmn : collaboration_model
        given collaboration model
    lane_set : ET xml element
        lane_set xml element
    node_lane_map : dict
        dictionary of all flownodes contained in a lane
    parent : ET xml element
        parent lane_set xml element (None if no parent exists)
    pool_id : str
        id of the current pool
    """

    for lane_set in tag.findall('laneSet'):

        if 'id' in lane_set.attrib:
            lane_set_id = lane_set.attrib['id']
        else:
            lane_set_id = str(uuid.uuid1())

        l = {'id': lane_set_id, 'isSet': True}

        if parent is None:
            l['pool'] = pool_id

        else:
            l['parent'] = parent['id']

            if 'children' not in parent:
                parent['children'] = [lane_set_id]
            else:
                parent['children'].append(lane_set_id)

        for attribute in lane_set.attrib:
            l[attribute] = lane_set.attrib[attribute]

        parse_lanes(bpmn, lane_set, node_lane_map, l, pool_id)

        bpmn.lanes[lane_set_id] = l

    for childLaneSet in tag.findall('childLaneSet'):
        child_lane_set_id = childLaneSet.attrib['id']

        l = {'id': child_lane_set_id, 'isSet': True}

        if parent is None:
            l['pool'] = pool_id

        else:
            l['parent'] = parent['id']

            if 'children' not in parent:
                parent['children'] = [child_lane_set_id]
            else:
                parent['children'].append(child_lane_set_id)

        for attribute in childLaneSet.attrib:
            l[attribute] = childLaneSet.attrib[attribute]

        parse_lanes(bpmn, childLaneSet, node_lane_map, l, pool_id)

        bpmn.lanes[child_lane_set_id] = l

    for lane in tag.findall('lane'):
        lane_id = lane.attrib['id']

        l = {'id': lane_id, 'isSet': False}

        if parent is None:
            l['pool'] = pool_id

        else:
            l['parent'] = parent['id']

            if 'children' not in parent:
                parent['children'] = [lane_id]
            else:
                parent['children'].append(lane_id)

        for attribute in lane.attrib:
            l[attribute] = lane.attrib[attribute]

        l['flowNodeRefs'] = []

        for node_ref in lane.findall('flowNodeRef'):
            node_lane_map[node_ref.text] = lane_id

        parse_lanes(bpmn, lane, node_lane_map, l, pool_id)

        bpmn.lanes[lane_id] = l


def parse_process(bpmn, process_id, pool_id, process, randomize_ids):

    """
    Parses a process of a bpmn and adds it to an existing collaboration model

    Parameters
    ----------
    bpmn : collaboration_model
        given collaboration model
    process_id : str
        id of the current process
    pool_id : str
        id of the current pool
    process : ET xml element
        process xml element
    randomize_ids : boolean
        if set true, randomizes all id's used in the process
    """
    node_lane_map = {}

    if pool_id is not None:
        #for lane_set in process.findall('laneSet'):
        parse_lanes(bpmn, process, node_lane_map, None, pool_id)

    id_map = {}

    for child in process:

        tag_name = child.tag

        if tag_name == Task.task.value \
                or tag_name == Task.userTask.value \
                or tag_name == Task.serviceTask.value \
                or tag_name == Task.manualTask.value \
                or tag_name == Task.sendTask.value \
                or tag_name == Task.receiveTask.value \
                or tag_name == Task.businessRuleTask.value \
                or tag_name == Task.callActivity.value\
                or tag_name == Task.scriptTask.value\
                or tag_name == Event.startEvent.value \
                or tag_name == Event.intermediateCatchEvent.value \
                or tag_name == Event.endEvent.value \
                or tag_name == Event.intermediateThrowEvent.value \
                or tag_name == Event.boundaryEvent.value \
                or tag_name == Others.subProcess.value \
                or tag_name == Gateway.inclusiveGateway.value \
                or tag_name == Gateway.exclusiveGateway.value \
                or tag_name == Gateway.parallelGateway.value \
                or tag_name == Gateway.eventBasedGateway.value \
                or tag_name == Gateway.complexGateway.value:
                # or tag_name == consts.Consts.business_rule_task \
                # or tag_name == consts.Consts.receive_task \
                # or tag_name == "callActivity":

            id_pair = import_flow_node(bpmn, process_id, child, randomize_ids)

            if randomize_ids:
                id_map[id_pair[0]] = id_pair[1]


        elif tag_name == Others.dataStoreReference.value \
                or tag_name == Others.textAnnotation.value \
                or tag_name == Others.dataObject.value:
            import_additional_node(bpmn, child)

        elif tag_name == "association":

            if not ('sourceRef' in child.attrib and 'targetRef' in child.attrib):
                continue

            source_ref = child.attrib['sourceRef']
            target_ref = child.attrib['targetRef']

            if bpmn.process_graph.has_node(source_ref):
                bpmn.process_graph.nodes[source_ref]['outgoing_association'] = target_ref
            elif bpmn.process_graph.has_node(target_ref):
                bpmn.process_graph.nodes[target_ref]['incoming_association'] = source_ref

        elif tag_name == Event.boundaryEvent.value:
            boundary_event_id = child.attrib['id']
            attached_to = child.attrib['attachedToRef']

            boundary_event = {
                'type': tag_name,
                'id': boundary_event_id,
                'attachedTo': attached_to
            }
            bpmn.additional_nodes.append(boundary_event)

            try:
                bpmn.process_graph.nodes[attached_to]['boundaryEvent'].append(boundary_event_id)
            except KeyError:
                bpmn.process_graph.nodes[attached_to]['boundaryEvent'] = [boundary_event_id]

    for seq_flow in process.findall('sequenceFlow'):

        if not ('sourceRef' in seq_flow.attrib and 'targetRef' in seq_flow.attrib):
            continue

        source_id = seq_flow.attrib['sourceRef']
        target_id = seq_flow.attrib['targetRef']

        if randomize_ids:
            source_id = id_map[source_id]
            target_id = id_map[target_id]

        flow_id = seq_flow.attrib['id']

        if not bpmn.process_graph.has_node(source_id):
            for additional_node in bpmn.additional_nodes:
                if additional_node['id'] == source_id:
                    if additional_node['type'] != 'boundaryEvent':
                        print('something wrong')
                    else:
                        source_id = additional_node['attachedTo']
                        bpmn.process_graph.add_edge(source_id, target_id, boundary=additional_node['id'], id=flow_id)
        else:
            bpmn.process_graph.add_edge(source_id, target_id, id=flow_id, type="sequenceFlow")

    for node_id in node_lane_map.keys():

        lane_id = node_lane_map[node_id]

        if randomize_ids:
            node_id = id_map[node_id]

        if bpmn.process_graph.has_node(node_id):
            bpmn.process_graph.nodes[node_id]['lane'] = lane_id
            bpmn.lanes[lane_id]['flowNodeRefs'].append(node_id)

    return id_map


def import_flow_node(bpmn, process_id, node, randomize_ids=False):
    """
    Parses a flow node of a bpmn process model and adds it to an existing collaboration model

    Parameters
    ----------
    bpmn : collaboration_model
        given collaboration model
    process_id : str
        id of the current process
    node : ET xml element
        flow node xml element
    randomize_ids : boolean, optional
        if set true, randomizes all id's used in the flow node
    """
    node_id = node.attrib['id']

    id_pair = [node_id, None]

    if randomize_ids:
        node_id = str(uuid.uuid1())
        id_pair[1] = node_id

    bpmn.process_graph.add_node(node_id)
    bpmn.process_graph.nodes[node_id]["type"] = node.tag

    bpmn.process_graph.nodes[node_id]["process"] = process_id

    for attribute in node.attrib:
        bpmn.process_graph.nodes[node_id][attribute] = node.attrib[attribute]

    bpmn.process_graph.nodes[node_id]["id"] = node_id

    # add name attribute if not exists
    if "name" not in bpmn.process_graph.nodes[node_id]:
        bpmn.process_graph.nodes[node_id]['name'] = ""

    data_input = []
    data_output = []

    for data_input_association in node.findall('dataInputAssociation'):
        for input_data in data_input_association.findall('sourceRef'):
            data_input.append(input_data.text)

    for data_output_association in node.findall('dataOutputAssociation'):
        for output_data in data_output_association.findall('targetRef'):
            data_output.append(output_data.text)

    if len(data_input) > 0:
        bpmn.process_graph.nodes[node_id]['data_input'] = data_input

    if len(data_output) > 0:
        bpmn.process_graph.nodes[node_id]['data_output'] = data_output

    if node.tag == 'subProcess':

        sub_process = CollaborationModel(name=node.attrib['id'])

        current = {}
        sub_process_id = node.attrib['id']

        for attribute in node.attrib:
            current[attribute] = node.attrib[attribute]

        parse_process(sub_process, sub_process_id, None, node, randomize_ids)
        sub_process.processes[process_id] = current

        bpmn.subprocesses[sub_process_id] = sub_process


    incomings = []
    outgoings = []

    for incoming in node.findall('incoming'):
        incomings.append(incoming.text)

    for outgoing in node.findall('outgoing'):
        outgoings.append(outgoing.text)
    event_type = []
    # bpmn standard chapter 10.4.5
    event_types = ["cancelEventDefinition", "compensationEventDefinition", "conditionalEventDefinition",
                   "errorEventDefinition", "escalationEventDefinition", "messageEventDefinition", "linkEventDefinition",
                   "signalEventDefinition", "terminateEventDefinition", "timerEventDefinition"]
    for event_type_ in event_types:
        for event in node.findall(event_type_):
            event_type_dict = {"type": event_type_}
            if 'id' in event.attrib:
                event_type_dict['id'] = event.attrib['id']
            event_type.append(event_type_dict)
    if event_type:
        bpmn.process_graph.nodes[node_id]["associated_event_types"] = event_type

    bpmn.process_graph.nodes[node_id]["incoming"] = incomings
    bpmn.process_graph.nodes[node_id]["outgoing"] = outgoings

    return id_pair


def import_additional_node(bpmn, node):
    """
    Parses a non-flow node of a bpmn process model and adds it to an existing collaboration model

    Parameters
    ----------
    bpmn : collaboration_model
        given collaboration model
    node : ET xml element
        non-flow node xml element
    """
    node_type = node.tag
    a_node = {'type': node_type}

    for attribute in node.attrib:
        a_node[attribute] = node.attrib[attribute]

    if node_type == 'textAnnotation':
        for text_tag in node.findall('text'):
            content = text_tag.text
            a_node['content'] = content

    bpmn.additional_nodes.append(a_node)

