from utils.model_generation import ModelGenerator

gen = ModelGenerator()

receive_customer = gen.activity("Receive customer inquiry")
address_customer = gen.activity("Address customer concerns or questions")
collect_customer = gen.activity("Collect customer information")
guide_customer = gen.activity("Guide customer in selecting product/service")
provide_quote = gen.activity("Provide quote")
place_order = gen.activity("Place order")
record_order = gen.activity("Record order in system")
send_order = gen.activity("Send order confirmation to customer")

xor_2_branch_2 = gen.partial_order(
    dependencies = [
        (place_order, record_order),
        (record_order, send_order),
    ]
)
xor_2 = gen.xor(None, xor_2_branch_2)

xor_1_branch_2 = gen.partial_order(
    dependencies = [
        (guide_customer, provide_quote),
        (provide_quote, xor_2),
    ]
)
xor_1 = gen.xor(None, xor_1_branch_2)

dependencies = [
    (receive_customer, address_customer),
    (receive_customer, collect_customer),
    (address_customer, xor_1),
    (collect_customer, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
