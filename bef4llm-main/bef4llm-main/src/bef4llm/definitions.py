from enum import Enum
from bef4llm.resource_controller.path_helper import get_folder_path
import os

"""
Pragmatic quality
"""
class Pragmatic_Subgroups(Enum):
    """
    Enum for all subgroups of Pragmatic Subgroups
    """
    size = "size"
    density = "density"
    connector_interplay = "connector_interplay"
    partionability = "partionability"
    cyclicity = "cyclicity"
    concurrency = "concurrency"
    other_metrics = "other_metrics"

class Pragmatic_Metrics(Enum):
    """
    Enum for all metrics of Pragmatic Metrics
    """
    total_numbers_of_gateways = "total_numbers_of_gateways"
    total_numbers_of_nodes = "total_numbers_of_nodes"
    total_numbers_of_sequence_flows = "total_numbers_of_sequence_flows"
    total_numbers_of_message_flows = "total_numbers_of_message_flows"
    diameter = "diameter"
    density = "density"
    average_gateway_degree = "average_gateway_degree"
    connectivity_coefficient = "connectivity_coefficient"
    gateway_heterogeneity = "gateway_heterogeneity"
    control_flow_complexity = "control_flow_complexity"
    sequentiality = "sequentiality"
    seperatibility = "seperatibility"
    depth = "depth"
    token_split = "token_split"
    cross_connectivity = "cross_connectivity"

# dict with thresholds for all pragmatic quality metrics
thresholds_pragmatic_quality_metrics = {
            Pragmatic_Subgroups.size.value:
                {Pragmatic_Metrics.total_numbers_of_nodes.value: [29.9, 43.7, 58.1, 81.1],
                 Pragmatic_Metrics.total_numbers_of_gateways.value: [1.42, 3.36, 5.3, 6.49],
                 Pragmatic_Metrics.total_numbers_of_sequence_flows.value: [19.4, 34.8, 50.2, 74.8],
                 Pragmatic_Metrics.total_numbers_of_message_flows.value: [1.09, 7.15, 13.2, 22.8],
                 Pragmatic_Metrics.diameter.value: [7.92, 12.2, 16.5, 23.4]},
            Pragmatic_Subgroups.density.value:
                #{Pragmatic_Mistakes_Metrics.density.value: [0.1361169, 0.357143, 0.741667, 0.233333],
                {Pragmatic_Metrics.density.value: [0.1361169, 0.357143, 0.741667, 2.33333],
                 Pragmatic_Metrics.average_gateway_degree.value: [3.67, 3.88, 4.06, 4.18],
                 Pragmatic_Metrics.connectivity_coefficient.value: [0.37, 0.9, 1.43, 2.28]},
            Pragmatic_Subgroups.connector_interplay.value:
                {Pragmatic_Metrics.gateway_heterogeneity.value: [0.62, 0.79, 0.92, 0.94],
                 #"gm": [], -> take since in syntax?
                 Pragmatic_Metrics.control_flow_complexity.value: [13, 22, 37, 51],
                 Pragmatic_Metrics.cross_connectivity.value: [0.112903, 0.061814, 0.030407, 0.007996]},
            Pragmatic_Subgroups.partionability.value:
                {Pragmatic_Metrics.seperatibility.value: [1.24, 0.71, 0.37, 0.03],
                 Pragmatic_Metrics.sequentiality.value: [1.07, 0.7, 0.48, 0.25],
                 Pragmatic_Metrics.depth.value: [0.42, 1.72, 3.02, 5.09]},
            Pragmatic_Subgroups.cyclicity.value: {},
            Pragmatic_Subgroups.concurrency.value:
                {Pragmatic_Metrics.token_split.value: [0.12, 0.21, 0.6, 1.36]}
        }

"""
Syntactic quality
"""
class Sytax_Mistakes(Enum):
    existence_start_event = "existence_start_event"
    existence_end_event = "existence_end_event"
    start_event_in_out_degree = "start_event_in_out_degree"
    end_event_in_out_degree = "end_event_in_out_degree"
    #sound_process_model = "sound_process_model"
    labeled_tasks = "labeled_tasks"
    connected_nodes = "connected_nodes"
    tasks_in_outdegree = "tasks_in_outdegree"
    intermediate_event_in_out_degree = "intermediate_event_in_out_degree"
    gateway_in_out_degree = "gateway_in_out_degree"
    pair_gateways = "pair_gateways"
    event_gateway_predecessor_successor_wrong = "event_gateway_predecessor_successor_wrong"
    one_start_event = "one_start_event"
    one_end_event = "one_end_event"
    wrong_sequence_flow = "wrong_sequence_flow"
    wrong_message_flow = "wrong_message_flow"
    one_process_in_pool = "one_process_in_pool"


"""
Semantic quality - Similarity 
"""

class Similarity_Groups(Enum):
    """
    Enum for all subgroups of Similarity Groups (semantic quality)
    """
    natural_language = "natural_language"
    graph_structure = "graph_structure"
    behaviour = "behaviour"

class Similarity_Metrics(Enum):
    """
    Enum for all metrics of Similarity Metrics (Semantic quality)
    """
    label_sim_syntactic = "label_sim_syntactic"
    label_sim_semantic = "label_sim_semantic"
    label_sim_context = "label_sim_context"
    graph_edit_distance = "graph_edit_distance"
    causal_footprint = "causal_footprint"
    dependency_graph = "dependency_graph"
    common_percentage = "common_percentage"

# other
class XES_Terminology(Enum):
    """
    Enum for all XES Terminology
    """
    CASE_PREFIX = "case:"
    EVENT_ID = "concept:name"
    CASE_ID = CASE_PREFIX + EVENT_ID
    RESOURCE = "org:resource"
    TRANSITION = "lifecycle:transition"
    TIMESTAMP = "time:timestamp"

    EVENT_START = "start"
    EVENT_END = "end"

class Folder(Enum):
    """
    Enum is a list of folders with the corresponding paths.
    """

    DATA = os.path.join("data")
    RESOURCE_CONTROLLER = os.path.join("src", "bef4llm", "resource_controller")
    RMM_CORE = os.path.join("src", "bef4llm")
    TEST = "test"

class SapSamConstants(Enum):
    """
    Enum for SapSam Constants (Dataset)
    """
    CSV_PATH = get_folder_path(Folder.DATA) + "/models/sapsam/csv"
    BPMN_PATH = get_folder_path(Folder.DATA) + "/models/sapsam/bpmn"
    MODEL_PATH = get_folder_path(Folder.DATA) + "/models"
    SAP_MODEL_PATH = get_folder_path(Folder.DATA) + "/models/sapsam"
    DATA_PATH = get_folder_path(Folder.DATA)

    BPMN2_NAMESPACE = "http://b3mn.org/stencilset/bpmn2.0#"

    # login data
    email = "chla00001@stud.uni-saarland.de"
    pw = "PetriNet234"
    tenant_id = '17989c1295ae48d895fdc47f2d75bc37'
    system_instance = 'https://academic.signavio.com'

class BPMNSize(Enum):
    """
    Enum for BPMN Sizes
    """
    very_small = "very small"
    small = "small"
    medium = "medium"
    large = "large"
    very_large = "very large"
