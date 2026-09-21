from utils.model_generation import ModelGenerator

gen = ModelGenerator()

check_current = gen.activity("Check current inventory level")
send_a = gen.activity("Send a manual alert")
send_an = gen.activity("Send an automated alert")
place_order = gen.activity("Place order with suppliers")
update_inventory = gen.activity("Update inventory system with expected delivery dates")
receive_stock = gen.activity("Receive stock")
record_stock = gen.activity("Record stock in system")
inspect_stock = gen.activity("Inspect stock for quality")
place_stock = gen.activity("Place stock on shelves")
place_stock_1 = gen.activity("Place stock in storage")
update_inventory_1 = gen.activity("Update inventory levels")

loop_1 = gen.loop(check_current, None)

xor_1 = gen.xor(send_a, send_an)

xor_2 = gen.xor(place_stock, place_stock_1)

dependencies = [
    (loop_1, xor_1),
    (xor_1, place_order),
    (place_order, update_inventory),
    (update_inventory, receive_stock),
    (receive_stock, record_stock),
    (receive_stock, inspect_stock),
    (inspect_stock, xor_2),
    (record_stock, update_inventory_1),
    (xor_2, update_inventory_1),
]

final_model = gen.partial_order(dependencies=dependencies)
