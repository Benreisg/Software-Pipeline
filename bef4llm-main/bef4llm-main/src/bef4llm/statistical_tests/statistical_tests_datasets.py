import copy

from bef4llm.llm_comparison import prepare_datasets, quality_check
from bef4llm.definitions import *

import pandas as pd

def calcualte_metric_results_per_bpmn_for_each_LLM(llms, datasets):
    """
    Generates the files containing the metric results for each BPMN.
    For each LLM one file is generated, and saved in the folder llm_metric_results_runx as file name_of_llm.csv

    Parameters
    -----------
    llms : list
        list of LLM tags
    datasets : dictionary
    """
    missing_llms = []
    for llm in llms:
        if not os.path.isdir(f"{get_folder_path(Folder.DATA)}/llm_metric_results_run1/{llm}.csv"):
            missing_llms.append(llm)

    for llm in llms:
        for run in range(1, 6):
            print("run ", run)

            quality_check.get_metric_results_per_process_model(datasets=datasets,
                                                     test_llm_dir=f"{get_folder_path(Folder.DATA)}/llm_metric_results_run{run}/{llm}",
                                                     target_file=f"{get_folder_path(Folder.DATA)}/llm_metric_results_run{run}/{llm}.csv",
                                                     analyse_method="metric_score",
                                                     run=run)


def generate_data_for_statistical_tests():
    """
    Generates data set for statistical tests.
    For each metric a seperate csv file is created. Each file contains all each generated BPMN of each tested LLM with the given metric score.
    BPMN that are not valid or not generated have a None or NaN value for the metric.

    The data are saved in the folder "data_statistical_tests"
    """
    folder_path = f"{get_folder_path(Folder.DATA)}/data_statistical_tests"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    bpmn_models = []

    datasets = dict()
    datasets["camunda"] = prepare_datasets.prepare_camunda()
    datasets["bpmn_and_text"] = prepare_datasets.prepare_text_and_bpmn()
    datasets["lre_new"] = prepare_datasets.prepare_lre_new()
    datasets["lre_old"] = prepare_datasets.prepare_lre_old()

    for dataset in datasets:
        bpmn_models.extend([key for key in datasets[dataset].keys()])

    bpmn_models_id = []
    for bpmn in bpmn_models:
        for i in range(1, 6):
            bpmn_models_id.append(f"{bpmn}_run {i}")

    llm_dict = {
        "qwen3:14b-q8_0": None,
        "llama3.3:70b-instruct-q8_0": None,
        "llama3.1:8b-instruct-q8_0": None,
        "qwen2.5:14b-instruct-q8_0": None,
        "deepseek-r1:14b-qwen-distill-q8_0": None,
        "phi4:14b-q8_0": None,
        "qwen2.5:32b-instruct-q8_0": None,
        "qwen3:30b-a3b-q8_0": None,
        "deepseek-r1:70b-llama-distill-q8_0": None,
        "falcon3:10b-instruct-q8_0": None,
        "qwen3:235b-a22b": None
    }

    metric_dicts_group = {
        "syntactic quality": {bpmn: copy.deepcopy(llm_dict) for bpmn in bpmn_models_id},
        "pragmatic quality": {bpmn: copy.deepcopy(llm_dict) for bpmn in bpmn_models_id},
        "semantic quality": {bpmn: copy.deepcopy(llm_dict) for bpmn in bpmn_models_id},
    }

    # calculate metric scores
    calcualte_metric_results_per_bpmn_for_each_LLM(llms=llm_dict, datasets=datasets)

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        llm = file.split(".csv")[0]
        df = pd.read_csv(f"{folder_path}/{file}", sep=";")

        for metric in metric_dicts_group:
            for i, data in df.iterrows():
                # row with column names
                if data["process model"] == "process model":
                    continue

                # save data for one BPMN
                bpmn = data["process model"]
                run = data["run"]
                metric_dicts_group[metric][f"{bpmn}_{run.strip()}"][llm] = data[metric]

    # create one file per metric, add BPMN analysis (for not valid BPMNs just an empty list is saved)
    metric_data = dict()
    for metric in metric_dicts_group:
        metric_data[metric] = {"bpmn": []}
        for bpmn in bpmn_models_id:
            metric_data[metric]["bpmn"].append(bpmn)
            for llm in metric_dicts_group[metric][bpmn]:
                if llm not in metric_data[metric]:
                    metric_data[metric][llm] = [metric_dicts_group[metric][bpmn][llm]]
                else:
                    metric_data[metric][llm].append(metric_dicts_group[metric][bpmn][llm])

    target_dir = f"{get_folder_path(Folder.DATA)}/Analysis Results/statistical_tests_group"
    for metric in metric_data:
        df = pd.DataFrame.from_dict(metric_data[metric])
        df.to_csv(target_dir + f"/{metric}.csv", sep=";", index=False)
        print(f"created {metric}.csv")
