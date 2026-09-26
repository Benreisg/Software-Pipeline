from utils.model_generation import ModelGenerator

gen = ModelGenerator()

place_order = gen.activity("Place order")
record_order = gen.activity("Record order details")
process_payment = gen.activity("Process payment")
check_stock = gen.activity("Check stock availability")
initiate_backorder = gen.activity("Initiate back-order")
inform_customer = gen.activity("Inform customer about delay")
receive_backorder = gen.activity("Receive back-order")
pick_items = gen.activity("Pick items from warehouse")
perform_quality = gen.activity("Perform quality control checks")
package_items = gen.activity("Package items")
gift_wrap = gen.activity("Gift wrap items")
prepare_shipping = gen.activity("Prepare shipping documents and labels")
prepare_documentation = gen.activity("Prepare documentation for international orders")
dispatch_order = gen.activity("Dispatch order")
update_inventory = gen.activity("Update inventory levels")
send_shipping = gen.activity("Send shipping confirmation to customer")
confirm_delivery = gen.activity("Confirm delivery")
send_followup = gen.activity("Send follow-up email")
customer_reports = gen.activity("Customer reports issue")
send_return = gen.activity("Send return shipping label")
receive_returned = gen.activity("Receive returned items")
inspect_returned = gen.activity("Inspect returned items")
process_refund = gen.activity("Process refund")
process_replacement = gen.activity("Process replacement")
notify_about = gen.activity("Notify about failed payment")

xor_2_branch_1 = gen.partial_order(
    dependencies = [
        (initiate_backorder, inform_customer),
        (inform_customer, receive_backorder),
    ]
)
xor_2 = gen.xor(xor_2_branch_1, None)

xor_3 = gen.xor(None, gift_wrap)

xor_4 = gen.xor(None, prepare_documentation)

xor_6 = gen.xor(process_refund, process_replacement)

xor_5_branch_1 = gen.partial_order(
    dependencies = [
        (customer_reports, send_return),
        (send_return, receive_returned),
        (receive_returned, inspect_returned),
        (inspect_returned, xor_6),
    ]
)
xor_5 = gen.xor(xor_5_branch_1, None)

xor_1_branch_1 = gen.partial_order(
    dependencies = [
        (check_stock, xor_2),
        (xor_2, pick_items),
        (pick_items, perform_quality),
        (perform_quality, package_items),
        (package_items, xor_3),
        (perform_quality, prepare_shipping),
        (perform_quality, xor_4),
        (xor_3, dispatch_order),
        (prepare_shipping, dispatch_order),
        (xor_4, dispatch_order),
        (dispatch_order, update_inventory),
        (dispatch_order, send_shipping),
        (update_inventory, confirm_delivery),
        (send_shipping, confirm_delivery),
        (confirm_delivery, send_followup),
        (send_followup, xor_5),
    ]
)
xor_1 = gen.xor(xor_1_branch_1, notify_about)

dependencies = [
    (place_order, record_order),
    (record_order, process_payment),
    (process_payment, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
