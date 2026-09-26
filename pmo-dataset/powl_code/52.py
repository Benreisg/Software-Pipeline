from utils.model_generation import ModelGenerator

gen = ModelGenerator()

mpon_sents = gen.activity("MPON sents the dismissal to the MPOO")
mpoo_reviews = gen.activity("MPOO reviews the dismissal")
mpoo_opposes = gen.activity("MPOO opposes the dismissal")
mpoo_confirmes = gen.activity("MPOO confirmes the dismissal")

xor_1 = gen.xor(mpoo_opposes, mpoo_confirmes)

dependencies = [
    (mpon_sents, mpoo_reviews),
    (mpoo_reviews, xor_1),
]

final_model = gen.partial_order(dependencies=dependencies)
