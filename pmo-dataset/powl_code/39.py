from utils.model_generation import ModelGenerator

gen = ModelGenerator()

create_account = gen.activity("Create account")
log_into = gen.activity("Log into account")
select_device = gen.activity("Select device model")
select_type = gen.activity("Select type of hardware Issue")
list_required = gen.activity("List required materials")
list_required_1 = gen.activity("List required tools")
offer_ordering = gen.activity("Offer ordering choices")
contact_friends = gen.activity("Contact friends")
get_secondhand = gen.activity("Get second-hand tools and materials from friends")
purchase_online = gen.activity("purchase online")
display_repair = gen.activity("Display repair instructions")
repair_device = gen.activity("Repair device")
send_device = gen.activity("Send device to an expert")
write_review = gen.activity("Write review")
upload_video = gen.activity("Upload video of reparation steps")

xor_1 = gen.xor(create_account, None)

loop_1 = gen.loop(log_into, None)

xor_3_branch_2 = gen.partial_order(dependencies = [(contact_friends, get_secondhand)])
xor_3 = gen.xor(None, xor_3_branch_2)

xor_4 = gen.xor(purchase_online, None)

xor_2_branch_1 = gen.partial_order(dependencies = [(xor_3, xor_4)])
xor_2 = gen.xor(xor_2_branch_1, None)

xor_6 = gen.xor(write_review, None)

xor_7 = gen.xor(None, upload_video)

xor_5_branch_2 = gen.partial_order(dependencies = [(xor_6, xor_7)])
xor_5 = gen.xor(send_device, xor_5_branch_2)

dependencies = [
    (xor_1, loop_1),
    (loop_1, select_device),
    (select_device, select_type),
    (select_type, list_required),
    (select_type, list_required_1),
    (select_type, offer_ordering),
    (list_required, xor_2),
    (list_required_1, xor_2),
    (offer_ordering, xor_2),
    (xor_2, display_repair),
    (display_repair, repair_device),
    (repair_device, xor_5),
]

final_model = gen.partial_order(dependencies=dependencies)
