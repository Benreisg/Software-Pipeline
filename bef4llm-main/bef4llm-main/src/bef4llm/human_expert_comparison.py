from bef4llm.llm_comparison.quality_check import *
from bef4llm.resource_controller.path_helper import look_for_directory
from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language

def get_expert_scores(analyse_method):
    """
    Computes quality scores for expert dataset

    Parameters
    ----------
        analyse_method: str
            method of analysis (detail, metric_score, quality_group_score)

    Returns
    -------
    quality_scores: dict
        quality scores for expert dataset
    """
    def get_invalid_models(models):
        not_valid_xml = []
        for model in models:
            if not validate_bpmn(model):
                not_valid_xml.append(model)

        return not_valid_xml

    def extract_files(model_pairs):
        dir_path = look_for_directory("Expert_BPMN")
        model_dir = "experts_modelling"
        ground_truth_dir = "ground_truth_bpmn"

        for expert in os.listdir(os.path.join(dir_path, model_dir)):
            # exclude hidden directories
            if not expert.startswith("."):
                for model in os.listdir(os.path.join(dir_path, model_dir, expert)):
                    if model.startswith("."):
                        continue
                    if model not in model_pairs:
                        if "schufa" not in model:
                            ground_truth_bpmn_model = os.path.join(dir_path, ground_truth_dir, model)
                        else:
                            ground_truth_bpmn_model = [os.path.join(dir_path, ground_truth_dir, model.split(".")[0] + "1.bpmn")]
                            ground_truth_bpmn_model.append(os.path.join(dir_path, ground_truth_dir,
                                                                   model.split(".")[0] + "2.bpmn"))

                        model_pairs[model] = dict()
                        model_pairs[model]["ground truth"] = ground_truth_bpmn_model
                        model_pairs[model]["models"] = [os.path.join(dir_path, model_dir, expert, model)]
                    else:
                        model_pairs[model]["models"].append(os.path.join(dir_path, model_dir, expert, model))

        return model_pairs

    bpmn_model_pairs = dict()
    bpmn_model_pairs = extract_files(model_pairs=bpmn_model_pairs)

    if analyse_method == "quality_group_score":
        quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline",
                          "syntatic quality": [], "pragmatic quality": [], "semantic quality": []}
    else:
        if analyse_method == "detail":
            quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline", "syntatic quality": []}
        else:
            quality_scores = {"not_valid": 0, "valid": 0, "llm": "camunda_baseline"}

    for model in tqdm(bpmn_model_pairs):
        if not model.startswith("."):
            models = bpmn_model_pairs[model]["models"]
            ground_truth_bpmn_model = bpmn_model_pairs[model]["ground truth"]

            not_valid_xml = get_invalid_models(models)

            syn_scores, not_valid_xml = make_syntactic_check(models=models,
                                                 not_valid=not_valid_xml,
                                                 analyse_mode=analyse_method)
            prag_scores, not_valid_xml = make_pragmatic_check(models=models,
                                                  not_valid=not_valid_xml,
                                                  analyse_mode=analyse_method)
            semantic_score, not_valid_xml = make_semantic_check(
                models=models,
                reference_model=ground_truth_bpmn_model,
                lang=Language.ENGLISH,  # when multiple language implementation, this must be changed
                analyse_mode=analyse_method)


            if analyse_method == "quality_group_score":
                quality_scores["syntatic quality"].extend(syn_scores)
                quality_scores["pragmatic quality"].extend(prag_scores)
                quality_scores["semantic quality"].extend(semantic_score)

            else:
                if analyse_method == "detail":
                    quality_scores["syntatic quality"].extend(syn_scores)
                else:
                    for key in syn_scores:
                        if key not in quality_scores:
                            quality_scores[key] = []
                        quality_scores[key].extend(syn_scores[key])
                for key in prag_scores:
                    if key not in quality_scores:
                        quality_scores[key] = []
                    quality_scores[key].extend(prag_scores[key])

                for key in semantic_score:
                    if key not in quality_scores:
                        quality_scores[key] = []
                    quality_scores[key].extend(semantic_score[key])

            quality_scores["not_valid"] += len(not_valid_xml)
            quality_scores["valid"] += len(models) - len(not_valid_xml)


    for key in quality_scores:
        if isinstance(quality_scores[key], list):
            print(key, quality_scores[key])
            quality_scores[key] = mean(quality_scores[key])

    return quality_scores



