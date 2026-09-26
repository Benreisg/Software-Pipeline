from utils.model_generation import ModelGenerator

gen = ModelGenerator()

file_complaint = gen.activity("File complaint")
log_complaint = gen.activity("Log complaint")
assign_complaint = gen.activity("Assign complaint to relevant department")
review_complaint = gen.activity("Review complaint details")
approve_and = gen.activity("Approve and notify customer")
process_reimbursement = gen.activity("Process reimbursement")
reject_and = gen.activity("Reject and notify customer")
resolve_complaint = gen.activity("Resolve complaint")
provide_feedback = gen.activity("Provide feedback")

xor_1_branch_1 = gen.partial_order(dependencies = [(approve_and, process_reimbursement)])
xor_1 = gen.xor(xor_1_branch_1, reject_and)

xor_2 = gen.xor(provide_feedback, None)

dependencies = [
    (file_complaint, log_complaint),
    (log_complaint, assign_complaint),
    (assign_complaint, review_complaint),
    (review_complaint, xor_1),
    (xor_1, resolve_complaint),
    (resolve_complaint, xor_2),
]

final_model = gen.partial_order(dependencies=dependencies)
