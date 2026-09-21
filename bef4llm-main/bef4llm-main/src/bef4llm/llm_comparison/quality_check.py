from bef4llm.process_models.importer.bpmn_importer import load_bpmn_from_directory, load_diagram_from_xml
from bef4llm.semantic_quality.semantic_quality_check import SemanticQualityCheckBPMN
from bef4llm.synactic_quality.synactic_quality_check import SyntacticQualityCheckBPMN
from bef4llm.pragmatic_quality.pragmatic_quality_check import PragmaticQualityCheckBPMN
from bef4llm.process_models.graph_representation.collaboration_model import CollaborationModel
from bef4llm.validation.validation import validate_bpmn
from bef4llm.definitions import *

import pandas as pd
from tqdm import tqdm
from statistics import mean


def make_syntactic_check(models, not_valid=None, analyse_mode="detail"):
    """
    Computes the syntactic quality check for a given set of models.

    Parameters
    ----------
    models : list, str
        List of models to check or dir where models are located.
    not_valid : list
        list of models, that should not be checked (either path to model or object)
    analyse_mode : str
        Granuarity in which the syntactic quality should be computed (detail, quality_group_score, metric_score)

    Returns
    -------
    score_list_syn: list of scores with syntactic quality score for each model.
    or score_metric_dict: dict of metrics of syntactic quality, with scores for each model

    """
    if not not_valid:
        not_valid = []

    not_analysed = []
    score_list_syn = []
    score_metric_dict = {Sytax_Mistakes[metric].value: [] for metric in Sytax_Mistakes.__members__}

    if not isinstance(models, list) and os.path.isdir(models):
        model_dir = models
        models = [os.path.join(model_dir, file) for file in os.listdir(model_dir)]

    if len(models) > 0 and isinstance(models[0], str) and os.path.isfile(models[0]):
        new_models = []
        for model in models:
            if model not in not_valid and model.endswith('.bpmn') and model not in not_valid:
                try:
                    new_models.append(load_diagram_from_xml(model))
                except:
                    not_analysed.append(model)
            else:
                not_analysed.append(model)

        models = new_models.copy()


    for model in models:
        try:
            if model.process_graph.number_of_nodes() > 1 and model not in not_valid:
                    syn_check = SyntacticQualityCheckBPMN(model)
                    if analyse_mode == "detail" or analyse_mode == "quality_group_score":
                        score = syn_check.syntax_check()
                        score_list_syn.append(score)
                    elif analyse_mode == "metric_score":
                        metric_score = syn_check.syntax_check_metric_results()
                        for metric in score_metric_dict:
                            score_metric_dict[metric].append(metric_score[metric])

        except Exception as e:
            print(e)
            not_analysed.append(model)

    if analyse_mode == "detail" or analyse_mode == "quality_group_score":
        return score_list_syn, not_analysed
    elif analyse_mode == "metric_score":
        return score_metric_dict, not_analysed


def make_semantic_check(models, reference_model, lang, not_valid=None, analyse_mode="detail"):
    """
        Computes the semantic quality for a given set of models.

        Parameters
        ----------
        models : list, str
            List of models to check or dir where models are located.
        not_valid : list
            list of models, that should not be checked (either path to model or object)
        analyse_mode : str
            Granularity in which the syntactic quality should be computed (detail, quality_group_score, metric_score)

        Returns
        -------
        score_list_sem: list of scores with semantic quality score for each model.
        or score_group_dict: dict of groups of semantic quality, with scores for each model
        or score_metric_dict: dict of metrics of semantic quality, with scores for each model

        not_analysed: list of models, that could not be analysed
        """
    if not not_valid:
        not_valid = []

    if not isinstance(models, list) and os.path.isdir(models):
        model_dir = models
        models = [os.path.join(model_dir, file) for file in os.listdir(model_dir)]

    not_analysed = []
    score_list_sem = []
    score_metric_dict = {Similarity_Metrics[metric].value: [] for metric in Similarity_Metrics.__members__}
    score_group_dict = {Similarity_Groups[group].value: [] for group in Similarity_Groups.__members__}


    for model in models:
        try:
            if model not in not_valid and model.endswith('.bpmn'):
                if analyse_mode == "quality_group_score":
                    score = semantic_check_single_model(model, reference_model, lang, analyse_mode=analyse_mode)
                    if score:
                        score_list_sem.append(score)
                    else:
                        not_analysed.append(model)

                elif analyse_mode == "detail":
                    score_groups = semantic_check_single_model(model, reference_model, lang, analyse_mode=analyse_mode)
                    if score_groups:
                        for group in score_groups:
                            score_group_dict[group].append(score_groups[group])
                    else:
                        not_analysed.append(model)

                elif analyse_mode == "metric_score":
                    score_metrics = semantic_check_single_model(model, reference_model, lang, analyse_mode=analyse_mode)
                    if score_metrics:
                        for metric in score_metric_dict:
                            score_metric_dict[metric].append(score_metrics[metric])
                    else:
                        not_analysed.append(model)

        except Exception as e:
            print(e)
            not_analysed.append(model)

    if analyse_mode == "quality_group_score":
        return score_list_sem, not_analysed
    elif analyse_mode == "metric_score":
        return score_metric_dict, not_analysed
    elif analyse_mode == "detail":
        return score_group_dict, not_analysed


def semantic_check_single_model(model, reference_model, lang, analyse_mode="detail"):
    """
    make semantic check for single model (one model and one or multiple ground truth models)
    Parameters
    ----------
    model : CollaborationModel, str
        model to be checked as a path where it is located, or CollaborationModel
    reference_model : list of CollaborationModel or str
        models to be checked as a list of paths where they located, or a list of CollaborationModel
    lang : Language
        language of the textual description
    analyse_mode : str
        Granularity in which the syntactic quality should be computed (detail, quality_group_score, metric_score)
    """
    score_metric_dict = {Similarity_Metrics[metric].value: [] for metric in Similarity_Metrics.__members__}
    score_group_dict = {Similarity_Groups[group].value: [] for group in Similarity_Groups.__members__}
    try:
        if isinstance(model, str):
            bpmn = load_diagram_from_xml(model)
        else:
            bpmn = model

        if not bpmn.process_graph.number_of_nodes() > 1:
            return None

        if isinstance(reference_model, list) and isinstance(reference_model[0], str):
            reference_model = [load_diagram_from_xml(model) for model in reference_model]
        elif isinstance(reference_model, str):
            reference_model = [load_diagram_from_xml(reference_model)]

        if isinstance(reference_model, CollaborationModel):
            sem_check = SemanticQualityCheckBPMN(model=bpmn, reference_model=reference_model, lang=lang)
            if analyse_mode == "quality_group_score":
                return sem_check.semantic_quality_check()
            elif analyse_mode == "metric_score":
                return score_metric_dict.update(sem_check.semantic_quality_check_metric_results())
            elif analyse_mode == "detail":
                return score_group_dict.update(sem_check.semantic_quality_check_detailed())


        elif isinstance(reference_model, list):
            score_list_sem = []
            #score_metric_dict = {Similarity_Metrics[metric].value: [] for metric in Similarity_Metrics.__members__}
            #score_group_dict = {Similarity_Groups[group].value: [] for group in Similarity_Groups.__members__}
            for ref_model in reference_model:
                sem_check = SemanticQualityCheckBPMN(model=bpmn, reference_model=ref_model, lang=lang)
                if analyse_mode == "quality_group_score":
                    score = sem_check.semantic_quality_check()
                    score_list_sem.append(score)
                if analyse_mode == "detail":
                    group_scores = sem_check.semantic_quality_check_detailed()
                    for group in group_scores:
                        score_group_dict[group].append(group_scores[group])
                if analyse_mode == "metric_score":
                    metric_score = sem_check.semantic_quality_check_metric_results()
                    for metric in score_metric_dict:
                        score_metric_dict[metric].append(metric_score[metric])


            if analyse_mode == "quality_group_score":
                return mean(score_list_sem)
            elif analyse_mode == "metric_score":
                for group in score_metric_dict:
                    score_metric_dict[group] = mean(score_metric_dict[group])
                return score_metric_dict
            elif analyse_mode == "detail":
                for group in score_group_dict:
                    score_group_dict[group] = mean(score_group_dict[group])
                return score_group_dict
    except Exception as e:
        print(e)
        return None


def make_pragmatic_check(models, not_valid=None, analyse_mode="detail"):
    """
   Computes the pragmatic quality for a given set of models.

   Parameters
   ----------
   models : list, str
       List of models to check or dir where models are located.
   not_valid : list
       list of models, that should not be checked (either path to model or object)
   analyse_mode : str
       Granularity in which the pragmatic quality should be computed (detail, quality_group_score, metric_score)

   Returns
   -------
   score_list_prag: list of scores with pragmatic quality score for each model.
   or score_group_dict: dict of groups of pragmatic quality, with scores for each model
   or score_metric_dict: dict of metrics of pragmatic quality, with scores for each model

   not_analysed: list of models, that could not be analysed
   """

    not_analysed = []
    if not not_valid:
        not_valid = []

    if not isinstance(models, list) and os.path.isdir(models):
        model_dir = models
        models = [os.path.join(model_dir, file) for file in os.listdir(model_dir)]

    if len(models) > 0 and isinstance(models[0], str) and os.path.isfile(models[0]):
        new_models = []
        for model in models:
            if model not in not_valid and model.endswith('.bpmn') and model not in not_valid:
                try:
                    new_models.append(load_diagram_from_xml(model))
                except Exception as e:
                    not_analysed.append(model)
            else:
                not_analysed.append(model)

        models = new_models.copy()



    score_list_prag = []
    score_metric_dict = {Pragmatic_Metrics[metric].value: [] for metric in Pragmatic_Metrics.__members__}
    score_group_dict = {Pragmatic_Subgroups[group].value: [] for group in Pragmatic_Subgroups.__members__}
    for model in models:
        if model not in not_valid:
            try:
                if model.process_graph.number_of_nodes() > 1:
                    prag_check = PragmaticQualityCheckBPMN(model)
                    if analyse_mode == "quality_group_score":
                        score = prag_check.pragmatic_quality_check()
                        score_list_prag.append(score)
                    elif analyse_mode == "metric_score":
                        metric_score = prag_check.pragmatic_quality_check_metric_results()
                        for metric in score_metric_dict:
                            score_metric_dict[metric].append(metric_score[metric])
                    elif analyse_mode == "detail":
                        group_score = prag_check.pragmatic_quality_check_detailed()
                        for metric in score_group_dict:
                            score_group_dict[metric].append(group_score[metric])
                else:
                    not_analysed.append(model)


            except Exception as e:
                print(e)
                not_analysed.append(model)


    if analyse_mode == "quality_group_score":
        return score_list_prag, not_analysed
    elif analyse_mode == "metric_score":
        return score_metric_dict, not_analysed
    elif analyse_mode == "detail":
        return score_group_dict, not_analysed


def compute_overall_quality_llms(datasets, test_llm_dir, analyse_method="detaiil", target_file=None):
    """
    computes the quality scores for multiple LLMs based on the models they generated for multiple datasets
    Parameters:
    -----------
    dataset: dict
        dict with all datasets that should be checked for each LLM
    test_llm_dir: str
        directory to the process models modelled by different llms
    analyse_mode : str
       Granularity in which the pragmatic quality should be computed (detail, quality_group_score, metric_score)
    target_file: str
        file path to save the result (pandas dataframe)

    Returns:
    --------
    df: pandas dataframe
        Dataframe with LLMs as rows and quality dimensions/ quality dimension subgroups/ metrics as columns
    """
    quality_scores = init_quality_scores_dict(analyse_method, overall_quality_dict=True)

    for llm in tqdm(os.listdir(test_llm_dir)):
        if llm.startswith("."):
            continue

        quality_scores["llm"].append(llm)
        if llm.startswith("."):
            continue

        quality_scores_llm = init_quality_scores_dict(analyse_method)

        print(llm)
        num_valid = 0
        num_invalid = 0
        num_not_analysed = 0
        for dataset in os.listdir(os.path.join(test_llm_dir, llm)):
            not_analysed_xml = []

            if dataset.startswith(".") or os.path.isfile(f"{test_llm_dir}/{llm}/{dataset}"):
                continue

            model_dir = f"{test_llm_dir}/{llm}/{dataset}"
            print(dataset)

            not_valid_xml = get_invalid_models(model_dir)

            quality_scores_llm, na = get_quality_per_dataset(quality_scores_dict=quality_scores_llm,
                                                         models=model_dir,
                                                         analyse_method=analyse_method,
                                                         dataset=datasets[dataset],
                                                         not_valid_xml=not_valid_xml)

            not_analysed_xml.extend(na)

            num_valid += len(os.listdir(model_dir)) - len(set(not_valid_xml))
            num_invalid += len(set(not_valid_xml))
            num_not_analysed += len(set(not_analysed_xml))




        # save mean value over all tested BPMN
        for key in quality_scores_llm:
            if len(quality_scores_llm[key]) != 0:
                if key in quality_scores:
                    quality_scores[key].append(mean(quality_scores_llm[key]))
                else:
                    quality_scores[key] = [mean(quality_scores_llm[key])]
            else:
                quality_scores[key].append(None)

        quality_scores["validity"].append(num_valid/105)

    df = pd.DataFrame(quality_scores)
    if analyse_method == "quality_group_score":
        df = df.loc[:, ["llm", "syntactic quality", "pragmatic quality", "semantic quality", "validity"]]
    df.to_csv(target_file, sep=";")

    return df

def get_quality_per_dataset(quality_scores_dict, analyse_method, dataset, not_valid_xml, models):
    """
    computes the quality scores for one LLM and one dataset

    Parameters:
    -----------
    quality_scores_dict: dict
        dict with current quality scores for one LLM, which is extended
    analyse_mode : str
      Granularity in which the pragmatic quality should be computed (detail, quality_group_score, metric_score)
    dataset : str
        name of dataset for which the quality scores should be computed
    not_valid_xml: list
        dirs to BPMN XML, which are not valid
    models: list
        dir to folder containing the BPMN XML files for this dataset

    Returns:
    --------
    df: pandas dataframe
       Dataframe with LLMs as rows and quality dimensions/ quality dimension subgroups/ metrics as columns
    """

    not_analysed_xml = []
    syn_scores, na = make_syntactic_check(models=models,
                                          not_valid=not_valid_xml,
                                          analyse_mode=analyse_method)

    not_analysed_xml.extend(na)
    prag_scores, na = make_pragmatic_check(models=models,
                                           not_valid=not_valid_xml,
                                           analyse_mode=analyse_method)
    not_analysed_xml.extend(na)

    if analyse_method == "quality_group_score":
        quality_scores_dict["syntactic quality"].extend(syn_scores)
        quality_scores_dict["pragmatic quality"].extend(prag_scores)

        if isinstance(models, str):
            models = [os.path.join(models, model)for model in os.listdir(models)]
        for model in models:
            #if isinstance(model, str):
                #model_name = model.split("/")[-1].split(".")[0]
                #if not os.path.isfile(f"{model}/{model_name}.bpmn"):
                #    continue
                #model = f"{model}/{model_name}.bpmn"
            #else:
            #    model_name = model.name
            if isinstance(model, str):
                model_name = model.split("/")[-1].split(".")[0]
            else:
                model_name = model.name
            if model not in not_valid_xml:
                semantic_score = []
                semantic_score = semantic_check_single_model(
                    model=model,
                    reference_model=dataset[model_name][2],
                    lang=dataset[model_name][0],
                    analyse_mode=analyse_method)
                if semantic_score:
                    quality_scores_dict["semantic quality"].append(semantic_score)
    else:
        # syntactic scores
        if analyse_method == "detail":
            quality_scores_dict["syntactic quality"].extend(syn_scores)
        else:
            for key in syn_scores:
                if key not in quality_scores_dict:
                    quality_scores_dict[key] = []
                quality_scores_dict[key].extend(syn_scores[key])

        # pragmatic scores
        for key in prag_scores:
            if key not in quality_scores_dict:
                quality_scores_dict[key] = []
            quality_scores_dict[key].extend(prag_scores[key])


        # semantic scores
        if isinstance(models, str):
            models = [os.path.join(models, model) for model in os.listdir(models)]
        for model in models:
            if isinstance(model, str):
                model_name = model.split("/")[-1].split(".")[0]
            else:
                model_name = model.name
            if model not in not_valid_xml:
                semantic_scores = semantic_check_single_model(
                                model=model,
                                reference_model=dataset[model_name][2],
                                lang=dataset[model_name][0],
                                analyse_mode=analyse_method)
                if semantic_scores:
                    for key in semantic_scores:
                        if key not in quality_scores_dict:
                            quality_scores_dict[key] = []

                        quality_scores_dict[key].append(semantic_scores[key])
                else:
                    not_analysed_xml.append(model)

    return quality_scores_dict, not_analysed_xml

def get_invalid_models(model_dir):
    """
    Makes a list of BPMN XML, that are invalid

    Parameters
    ----------
    model_dir: str
        dir to the BPMN XML files

    Returns
    -------
    not_valid_xml
    """
    not_valid_xml = []
    for model in os.listdir(model_dir):
        if model.endswith(".bpmn"):
            with open(f"{model_dir}/{model}") as f:
                content = f.read()
            if content and content != "":
                if not validate_bpmn(f"{model_dir}/{model}"):
                    not_valid_xml.append(f"{model_dir}/{model}")
            else:
                not_valid_xml.append(f"{model_dir}/{model}")



    return not_valid_xml

def init_quality_scores_dict(analyse_method, overall_quality_dict=False):
    """
    Initalizes the dict for the quality scores

    Parameters
    ----------
    analyse_method: str
        Indicates on which granularity the quality scores should be computed (detail, quality_group_score, metric_score)
    overall_quality_dict: bool
        Indicates if the dict is used in context of multiple or one LLM
    """
    if analyse_method == "quality_group_score":
        quality_scores = {"syntactic quality": [], "pragmatic quality": [], "semantic quality": []}
    elif analyse_method == "metric_score":
        quality_scores = {Sytax_Mistakes[metric].value: [] for metric in Sytax_Mistakes.__members__}
        quality_scores.update({Pragmatic_Metrics[metric].value: [] for metric in Pragmatic_Metrics.__members__})
        quality_scores.update({Similarity_Metrics[metric].value: [] for metric in Similarity_Metrics.__members__})

    elif analyse_method == "detail":
        quality_scores = {"syntactic quality": []}
        quality_scores.update(
            {Pragmatic_Subgroups[metric].value: [] for metric in Pragmatic_Subgroups.__members__})
        quality_scores.update({Similarity_Groups[metric].value: [] for metric in Similarity_Groups.__members__})
    else:
        quality_scores = dict()

    if overall_quality_dict:
        quality_scores["llm"] = []
        quality_scores["validity"] = []


    return quality_scores

def get_metric_results_per_process_model(datasets, llm_dir, analyse_method, run, target_file=None):
    """
    returns a dataframe with the score for all metric for each process model (for all files in the given llm_dir).
    So per LLM (given by llm_dir) a csv file is created and for all valid BPMNs the metric results are inserted.

    Parameters
    ----------
    dataset: dict
        dict with all datasets
    llm_dir: str
        directory to the process models modelled by different llms are stored
    analyse_method: str
        Indicates on which granularity the quality scores should be computed (detail, quality_group_score, metric_score)
    target_file: str
        file path to save the result (pandas dataframe)

    Returns
    -------
    df: pandas dataframe
        one row per bpmn, metrics as columns
    """
    quality_scores = init_quality_scores_dict(analyse_method, overall_quality_dict=False)
    quality_scores["process model"] = []
    quality_scores["run"] = []

    for dataset in os.listdir(llm_dir):
        if os.path.isdir(f"{llm_dir}/{dataset}"):
            not_valid_xml = get_invalid_models(os.path.join(llm_dir, dataset))
            for model_name in os.listdir(os.path.join(llm_dir, dataset)):
                if os.path.join(llm_dir, dataset, model_name) not in not_valid_xml:
                    try:
                        model = load_diagram_from_xml(os.path.join(llm_dir, dataset, model_name))
                        model_name = model_name.split(".")[0]
                        scores = dict()
                        if analyse_method == "metric_score":
                            syn_check = SyntacticQualityCheckBPMN(model)
                            scores.update(syn_check.syntax_check_metric_results())

                            prag_check = PragmaticQualityCheckBPMN(model)
                            scores.update(prag_check.pragmatic_quality_check_metric_results())

                            scores.update(semantic_check_single_model(
                                model=model,
                                reference_model=datasets[dataset][model_name][2],
                                lang=datasets[dataset][model_name][0],
                                analyse_mode=analyse_method))

                        elif analyse_method == "detail":
                            syn_check = SyntacticQualityCheckBPMN(model)
                            scores["syntactic quality"] = [syn_check.syntax_check()]

                            prag_check = PragmaticQualityCheckBPMN(model)
                            scores.update(prag_check.pragmatic_quality_check_detailed())

                            scores.update(semantic_check_single_model(
                                model=model,
                                reference_model=datasets[dataset][model_name][2],
                                lang=datasets[dataset][model_name][0],
                                analyse_mode=analyse_method))

                        elif analyse_method == "quality_group_score":
                            syn_check = SyntacticQualityCheckBPMN(model)
                            scores["syntactic quality"] = syn_check.syntax_check()

                            prag_check = PragmaticQualityCheckBPMN(model)
                            scores["pragmatic quality"] = prag_check.pragmatic_quality_check()

                            scores["semantic quality"] = semantic_check_single_model(
                                model=model,
                                reference_model=datasets[dataset][model_name][2],
                                lang=datasets[dataset][model_name][0],
                                analyse_mode=analyse_method)

                        quality_scores["process model"].append(model_name)
                        quality_scores["run"].append(f"run {run}")
                        for score in scores:
                            quality_scores[score].append(scores[score])
                    except Exception as e:
                        print(e)

    df = pd.DataFrame(quality_scores)

    # determines if file needs to be attached
    if not os.path.isfile(target_file):
        df.to_csv(target_file, sep=";", index=False)
    else:
        df.to_csv(target_file, sep=";", mode='a', index=False, header=False)
    return df
