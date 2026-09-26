from utils.model_generation import ModelGenerator

gen = ModelGenerator()

select_type = gen.activity("Select type of order")
select_menu = gen.activity("Select menu")
select_type_1 = gen.activity("Select type of beverage")
prepare_beverage = gen.activity("Prepare beverage")
select_side = gen.activity("Select side")
prepare_potato = gen.activity("Prepare potato wedges")
prepare_french = gen.activity("Prepare French fries")
select_burger = gen.activity("Select burger")
make_payment = gen.activity("Make payment")
enter_customers = gen.activity("Enter customer's name")
update_status = gen.activity("Update status")
check_status = gen.activity("Check status")
prepare_burger = gen.activity("Prepare burger")
notify_customer = gen.activity("Notify customer")
deliver_via = gen.activity("Deliver via conveyor belt")

xor_2 = gen.xor(prepare_potato, prepare_french)

xor_1_branch_1 = gen.partial_order(
    dependencies = [
        (select_menu, select_type_1),
        (select_type_1, prepare_beverage),
        (select_type_1, select_side),
        (prepare_beverage, xor_2),
        (select_side, xor_2),
    ]
)
xor_1 = gen.xor(xor_1_branch_1, select_burger)

loop_1_branch_1 = gen.partial_order(dependencies = [(update_status, check_status)])
loop_1 = gen.loop(loop_1_branch_1, None)

dependencies = [
    (select_type, xor_1),
    (xor_1, make_payment),
    (make_payment, enter_customers),
    (enter_customers, loop_1),
    (enter_customers, prepare_burger),
    (loop_1, notify_customer),
    (prepare_burger, notify_customer),
    (notify_customer, deliver_via),
]

final_model = gen.partial_order(dependencies=dependencies)
