from utils.model_generation import ModelGenerator

gen = ModelGenerator()

mspn_sends = gen.activity("MSPN sends a dismissal to MSPO")
mspo_reviews = gen.activity("MSPO reviews the dismissal")
mspo_rejects = gen.activity("MSPO rejects the dismissal of the MSPN")
mspo_confirms = gen.activity("MSPO confirms the dismissal of the MSPN")

xor_1 = gen.xor(mspo_rejects, mspo_confirms)

dependencies = [
    (mspn_sends, mspo_reviews),
    (mspo_reviews, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
