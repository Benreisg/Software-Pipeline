from utils.model_generation import ModelGenerator

gen = ModelGenerator()

identify_development = gen.activity("Identify development needs or career aspirations")
create_personal = gen.activity("Create personal development plan")
work_on = gen.activity("Work on skill enhancement")
receive_feedback = gen.activity("Receive feedback and evaluation from supervisors")
consider_employee = gen.activity("Consider employee for promotion or new role")
conducts_formal = gen.activity("Conducts formal performance review")
approve_promotion = gen.activity("Approve promotion")
set_new = gen.activity("Set new responsibilities")
adjust_compensation = gen.activity("Adjust compensation")
transition_into = gen.activity("Transition into new role")

loop_1 = gen.loop(work_on, None)

loop_2 = gen.loop(receive_feedback, None)

xor_1_branch_1 = gen.partial_order(
    dependencies = [
        (approve_promotion, set_new),
        (approve_promotion, adjust_compensation),
        (set_new, transition_into),
        (adjust_compensation, transition_into),
    ]
)
xor_1 = gen.xor(xor_1_branch_1, None)

dependencies = [
    (identify_development, create_personal),
    (create_personal, loop_1),
    (create_personal, loop_2),
    (loop_1, consider_employee),
    (loop_2, consider_employee),
    (consider_employee, conducts_formal),
    (conducts_formal, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
