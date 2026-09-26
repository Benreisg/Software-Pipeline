from utils.model_generation import ModelGenerator

gen = ModelGenerator()

inq_transmits = gen.activity("INQ transmits the transaction data request")
ip_checks = gen.activity("IP checks the request of the INQ")
ip_answers = gen.activity("IP answers the question of the INQ")

dependencies = [
    (inq_transmits, ip_checks),
    (ip_checks, ip_answers),
]

final_model = gen.partial_order(dependencies=dependencies)
