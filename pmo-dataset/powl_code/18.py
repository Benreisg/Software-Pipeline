from utils.model_generation import ModelGenerator

gen = ModelGenerator()

submit_application = gen.activity("Submit application online")
review_application = gen.activity("Review application and documents")
notify_applicant = gen.activity("Notify applicant of missing documents")
provide_missing = gen.activity("Provide missing documents")
evaluate_application = gen.activity("Evaluate application by admissions committee")
process_fees = gen.activity("Process fees or waivers")
send_rejection = gen.activity("Send rejection letter")
send_acceptance = gen.activity("Send acceptance letter")
cancel_application = gen.activity("Cancel application")
confirm_enrollment = gen.activity("Confirm enrollment")
send_orientation = gen.activity("Send orientation materials")
set_up = gen.activity("Set up IT accounts")
assist_with = gen.activity("Assist with visa processing")
meet_with = gen.activity("Meet with academic advisor")
select_courses = gen.activity("Select courses")
resolve_schedule = gen.activity("Resolve schedule conflicts")
obtain_student = gen.activity("Obtain student ID card")
begin_attending = gen.activity("Begin attending classes")
adddrop_courses = gen.activity("Add/drop courses")
post_grades = gen.activity("Post grades")
review_grades = gen.activity("Review grades online")
submit_appeal = gen.activity("Submit appeal form")
meet_with_1 = gen.activity("Meet with appeals committee")
await_decision = gen.activity("Await decision")
withdraw = gen.activity("Withdraw")
graduate = gen.activity("Graduate")

xor_1_branch_1 = gen.partial_order(dependencies = [(notify_applicant, provide_missing)])
xor_1 = gen.xor(xor_1_branch_1, None)

xor_4 = gen.xor(assist_with, None)

xor_5 = gen.xor(None, adddrop_courses)

xor_6_branch_1 = gen.partial_order(
    dependencies = [
        (submit_appeal, meet_with_1),
        (meet_with_1, await_decision),
    ]
)
xor_6 = gen.xor(xor_6_branch_1, None)

loop_1_branch_1 = gen.partial_order(
    dependencies = [
        (begin_attending, xor_5),
        (xor_5, post_grades),
        (post_grades, review_grades),
        (review_grades, xor_6),
    ]
)
loop_1 = gen.loop(loop_1_branch_1, None)

xor_7 = gen.xor(withdraw, graduate)

xor_3_branch_2 = gen.partial_order(
    dependencies = [
        (confirm_enrollment, send_orientation),
        (confirm_enrollment, set_up),
        (send_orientation, xor_4),
        (set_up, xor_4),
        (xor_4, meet_with),
        (meet_with, select_courses),
        (select_courses, resolve_schedule),
        (xor_4, obtain_student),
        (resolve_schedule, loop_1),
        (obtain_student, loop_1),
        (loop_1, xor_7),
    ]
)
xor_3 = gen.xor(cancel_application, xor_3_branch_2)

xor_2_branch_2 = gen.partial_order(dependencies = [(send_acceptance, xor_3)])
xor_2 = gen.xor(send_rejection, xor_2_branch_2)

dependencies = [
    (submit_application, review_application),
    (review_application, xor_1),
    (xor_1, evaluate_application),
    (xor_1, process_fees),
    (evaluate_application, xor_2),
    (process_fees, xor_2),
]

final_model = gen.partial_order(dependencies=dependencies)
