from utils.model_generation import ModelGenerator

gen = ModelGenerator()

customer_signs = gen.activity("Customer signs up")
generate_account = gen.activity("Generate account")
assign_access = gen.activity("Assign access")
set_automatic = gen.activity("Set automatic triggers for billing cycles")
send_renewal = gen.activity("Send renewal notifications")
send_regular = gen.activity("Send regular updates")
send_product = gen.activity("Send product enhancements")
customer_submits = gen.activity("Customer submits cancellation request")
deactivate_subscription = gen.activity("Deactivate subscription")
apply_refund = gen.activity("apply refund")
apply_charges = gen.activity("apply charges")
settle_final = gen.activity("Settle final account balance")

loop_1_branch_1 = gen.partial_order(
    dependencies = [
        (send_renewal, ),
        (send_regular, ),
        (send_product, ),
    ]
)
loop_1 = gen.loop(loop_1_branch_1, None)

xor_2 = gen.xor(apply_refund, apply_charges, None)

xor_1_branch_2 = gen.partial_order(
    dependencies = [
        (customer_submits, deactivate_subscription),
        (customer_submits, xor_2),
        (xor_2, settle_final),
    ]
)
xor_1 = gen.xor(None, xor_1_branch_2)

dependencies = [
    (customer_signs, generate_account),
    (generate_account, assign_access),
    (generate_account, set_automatic),
    (assign_access, loop_1),
    (set_automatic, loop_1),
    (loop_1, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
