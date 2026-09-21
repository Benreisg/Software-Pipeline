from bef4llm.semantic_quality.similarity.behavioural_similarity.trace_simulation import TraceExtractorBPMN
from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel

""" not working - change in traces needed"""

class PrincipalTransitions:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        pass

    def principal_transition_simialrity(self):
        transitions1 = self.create_principal_transition_sets(self.model1)
        transitions1 = self.create_principal_transition_sets(self.model2)


    def create_principal_transition_sets(self, model:CollaborationModel):
        print("principal transitions")
        trace_extractor = TraceExtractorBPMN(model, connector="Xor")
        traces = trace_extractor.main(model.get_entry_nodes())
        print(traces)

        # get loops
        prefix_loop = set()
        repeat_part_loop = set()

        #for trace in traces:
            #for task in trace:
                #suc = get_successor_by_type(model.process_graph, task, [t for t in Task.__members__])


        return None

