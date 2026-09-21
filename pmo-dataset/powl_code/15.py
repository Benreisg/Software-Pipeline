from utils.model_generation import ModelGenerator

gen = ModelGenerator()

schedule_compliance = gen.activity("Schedule compliance audit")
conduct_selfassessment = gen.activity("Conduct self-assessment")
gather_evidence = gen.activity("Gather evidence")
prepare_documentation = gen.activity("Prepare documentation")
conduct_external = gen.activity("Conduct external audit")
identify_gaps = gen.activity("Identify gaps or issues")
make_necessary = gen.activity("Make necessary corrections or improvements")
conduct_final = gen.activity("Conduct final audit")
award_certification = gen.activity("Award certification")
issue_official = gen.activity("Issue official documents")

xor_1_branch_1 = gen.partial_order(dependencies = [(award_certification, issue_official)])
xor_1 = gen.xor(xor_1_branch_1, None)

dependencies = [
    (schedule_compliance, conduct_selfassessment),
    (schedule_compliance, gather_evidence),
    (schedule_compliance, prepare_documentation),
    (conduct_selfassessment, conduct_external),
    (gather_evidence, conduct_external),
    (prepare_documentation, conduct_external),
    (conduct_external, identify_gaps),
    (identify_gaps, make_necessary),
    (make_necessary, conduct_final),
    (conduct_final, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
