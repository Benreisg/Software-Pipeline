from utils.model_generation import ModelGenerator

gen = ModelGenerator()

employee_submits = gen.activity("Employee submits a vacation request")
requirement_is = gen.activity("Requirement is registered")
employees_supervisor = gen.activity("Employee's supervisor receive the request")
supervisor_approve = gen.activity("Supervisor approve or reject the request")
application_is = gen.activity("Application is returned to employee")
employee_can = gen.activity("Employee can check rejection reasons")
notification_is = gen.activity("Notification is generated to the HR Representative")
hr_representative = gen.activity("HR Representative complete the respective management procedures")

xor_1_branch_1 = gen.partial_order(dependencies = [(application_is, employee_can)])
xor_1_branch_2 = gen.partial_order(dependencies = [(notification_is, hr_representative)])
xor_1 = gen.xor(xor_1_branch_1, xor_1_branch_2)

dependencies = [
    (employee_submits, requirement_is),
    (requirement_is, employees_supervisor),
    (employees_supervisor, supervisor_approve),
    (supervisor_approve, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
