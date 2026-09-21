from utils.model_generation import ModelGenerator

gen = ModelGenerator()

customer_places = gen.activity("Customer places order online")
customer_places_1 = gen.activity("Customer places order over the phone")
generate_and = gen.activity("Generate and send order confirmation")
generate_shipping = gen.activity("Generate shipping label")
pick_and = gen.activity("Pick and pack items")
hand_over = gen.activity("Hand over order to logistics provider")
send_tracking = gen.activity("Send tracking information to customer")
monitor_shipment = gen.activity("Monitor shipment")
successful_delivery = gen.activity("Successful delivery")
process_customer = gen.activity("Process customer feedback or returns")

xor_1 = gen.xor(customer_places, customer_places_1)

loop_1 = gen.loop(monitor_shipment, None)

xor_2 = gen.xor(process_customer, None)

dependencies = [
    (xor_1, generate_and),
    (generate_and, generate_shipping),
    (generate_and, pick_and),
    (generate_shipping, hand_over),
    (pick_and, hand_over),
    (hand_over, send_tracking),
    (send_tracking, loop_1),
    (loop_1, successful_delivery),
    (successful_delivery, xor_2),
]

final_model = gen.partial_order(dependencies=dependencies)
