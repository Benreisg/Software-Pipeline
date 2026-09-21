"""
Enums for all node types of collaboration models
"""
from enum import Enum

class Task(Enum):
    """
    Enum of all task-related nodes in a collaboration model
    """
    # EPC Nodes
    function = "function"

    # BPMN Nodes
    task = "task"
    userTask = "userTask"
    serviceTask = "serviceTask"
    manualTask = "manualTask"
    sendTask = "sendTask"
    scriptTask = "scriptTask"
    receiveTask = "receiveTask"
    businessRuleTask = "businessRuleTask"
    callActivity = "callActivity" # subprocess is not implemented!!!
    process = "process"
    subprocess = "subprocess"

    # Petri Net Nodes
    transition = "transition"

class Event(Enum):
    """
    Enum of all event-related nodes in a collaboration model
    """
    # EPC Nodes
    event = "event"

    # BPMN Nodes
    startEvent = "startEvent"
    intermediateCatchEvent = "intermediateCatchEvent"
    endEvent = "endEvent"
    intermediateThrowEvent = "intermediateThrowEvent"
    boundaryEvent = "boundaryEvent"

    # Petri Net Nodes
    place = "place"

class Gateway(Enum):
    """
    Enum of all gateway-related nodes in a collaboration model
    """
    OR = 1
    AND = 2
    XOR = 3

    # BPMN
    inclusiveGateway = "inclusiveGateway"
    exclusiveGateway = "exclusiveGateway"
    parallelGateway = "parallelGateway"
    eventBasedGateway = "eventBasedGateway"
    complexGateway = "complexGateway"

class Participant(Enum):
    """
    Enum of all participant-related nodes in a collaboration model
    """
    participant = None
    lane = "lane"
    pool = "pool"

class Flows(Enum):
    """
    Enum of all flow-related nodes in a collaboration model
    """
    sequenceFlow = "sequenceFlow"
    messageFlow = "messageFlow"




class Others(Enum):
    """
    Enum of all other nodes in a collaboration model
    """
    # BPMN Nodes
    subProcess = "subProcess"
    dataStoreReference = "dataStoreReference"
    textAnnotation = "textAnnotation"
    dataObject = "dataObject"
