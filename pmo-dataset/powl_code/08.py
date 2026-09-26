from utils.model_generation import ModelGenerator

gen = ModelGenerator()

report_incident = gen.activity("Report incident")
log_report = gen.activity("Log report into tracking system")
assign_report = gen.activity("Assign report to appropriate team")
gather_necessary = gen.activity("Gather necessary information")
identify_cause = gen.activity("Identify cause of incident")
propose_corrective = gen.activity("Propose corrective actions")
implement_fix = gen.activity("Implement fix")
change_policy = gen.activity("Change policy")
conduct_training = gen.activity("Conduct training")
conduct_followup = gen.activity("Conduct follow-up")
close_incident = gen.activity("Close incident report")
notify_all = gen.activity("Notify all stakeholders")

xor_1 = gen.xor(implement_fix, change_policy, conduct_training)

dependencies = [
    (report_incident, log_report),
    (log_report, assign_report),
    (assign_report, gather_necessary),
    (assign_report, identify_cause),
    (gather_necessary, propose_corrective),
    (identify_cause, propose_corrective),
    (propose_corrective, xor_1),
    (xor_1, conduct_followup),
    (conduct_followup, close_incident),
    (close_incident, notify_all),
]

final_model = gen.partial_order(dependencies=dependencies)
