from utils.model_generation import ModelGenerator

gen = ModelGenerator()

identify_need = gen.activity("Identify need for new supplier or vendor")
issue_request = gen.activity("Issue request for proposals (RFP)")
receive_supplier = gen.activity("Receive supplier proposals")
evaluate_proposal = gen.activity("Evaluate proposal")
conduct_interview = gen.activity("Conduct interview")
conduct_site = gen.activity("Conduct site visit")
select_supplier = gen.activity("Select supplier")
begin_contract = gen.activity("Begin contract negotiations")
sign_contract = gen.activity("Sign contract")
onboard_supplier = gen.activity("Onboard supplier")
execute_contract = gen.activity("Execute contract")

xor_1 = gen.xor(None, conduct_interview, conduct_site)

loop_1_branch_1 = gen.partial_order(dependencies = [(evaluate_proposal, xor_1)])
loop_1 = gen.loop(loop_1_branch_1, None)

dependencies = [
    (identify_need, issue_request),
    (issue_request, receive_supplier),
    (receive_supplier, loop_1),
    (loop_1, select_supplier),
    (select_supplier, begin_contract),
    (begin_contract, sign_contract),
    (sign_contract, onboard_supplier),
    (onboard_supplier, execute_contract),
]

final_model = gen.partial_order(dependencies=dependencies)
