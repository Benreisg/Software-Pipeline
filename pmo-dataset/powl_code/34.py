from utils.model_generation import ModelGenerator

gen = ModelGenerator()

develop_basic = gen.activity("Develop basic design")
order_lego = gen.activity("Order Lego brick sets")
give_the = gen.activity("Give the lego sets to the children")
sort_the = gen.activity("Sort the parts into containers")
reorder_parts = gen.activity("Reorder parts")
build_the = gen.activity("Build the next subcomponent")
test_subcomponent = gen.activity("Test subcomponent 1")
test_subcomponent_1 = gen.activity("Test subcomponent 4")
test_subcomponent_2 = gen.activity("Test subcomponent 2")
test_subcomponent_3 = gen.activity("Test subcomponent 3")
assemble_subcomponents = gen.activity("Assemble subcomponents")

xor_1_branch_1 = gen.partial_order(
    dependencies = [
        (order_lego, give_the),
        (give_the, sort_the),
    ]
)
xor_1 = gen.xor(xor_1_branch_1, None)

xor_2 = gen.xor(reorder_parts, None)

loop_2_branch_1 = gen.partial_order(dependencies = [(xor_2, build_the)])
loop_2 = gen.loop(loop_2_branch_1, None)

loop_1_branch_1 = gen.partial_order(
    dependencies = [
        (develop_basic, xor_1),
        (xor_1, loop_2),
        (loop_2, test_subcomponent),
        (loop_2, test_subcomponent_1),
        (loop_2, test_subcomponent_2),
        (loop_2, test_subcomponent_3),
    ]
)
loop_1 = gen.loop(loop_1_branch_1, None)

dependencies = [(loop_1, assemble_subcomponents)]

final_model = gen.partial_order(dependencies=dependencies)
