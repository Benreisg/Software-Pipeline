import argparse
import re
import sys

from bef4llm.llm_comparison import prepare_datasets, quality_check
from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path

import os
from tqdm import tqdm
from bef4llm.llm_comparison.generate_bpmns import Benchmark


def generate_bpmn(run, expert_dataset=False):
    """
    automates the process of generating BPMNs with multiple LLMs.
    All BPMNs are saved in the folder llm_runx, with x being the iteration of the experiment, and sorted by LLM and dataset.
    Each BPMN is saved in a separate BPMN-XML file.

    Prameters
    run: int
        indicates the number of the iteration
    expert_dataset: bool
        if true, the generation of BPMNs of only made for the expert dataset
    """
    llms = [
        "llama3.3:70b-instruct-q8_0",
        "llama3.2:1b-instruct-q8_0",
        "llama3.1:8b-instruct-q8_0",
        "qwen2.5:1.5b-instruct-q8_0",
        "qwen2.5:14b-instruct-q8_0",
        "qwen2.5:32b-instruct-q8_0",
        "deepseek-r1:1.5b-qwen-distill-q8_0",
        "deepseek-r1:14b-qwen-distill-q8_0",
        "deepseek-r1:8b-llama-distill-q8_0",
        "deepseek-r1:70b-llama-distill-q8_0",
        "phi4:14b-q8_0",
        "falcon3:10b-instruct-q8_0",
        "falcon3:3b-instruct-q8_0",
        "qwen3:30b-a3b-q8_0",
        "qwen3:14b-q8_0",
        "qwen3:1.7b-q8_0",
        "qwen3:235b-a22b",
    ]

    datasets = dict()
    if not expert_dataset:
        datasets["camunda"] = prepare_datasets.prepare_camunda()
        datasets["bpmn_and_text"] = prepare_datasets.prepare_text_and_bpmn()
        datasets["lre_new"] = prepare_datasets.prepare_lre_new()
        datasets["lre_old"] = prepare_datasets.prepare_lre_old()
    else:
        datasets["experts"] = prepare_datasets.prepare_experts_comparison()

    print("Number of textual descriptions: ", sum([len(datasets[d].keys()) for d in datasets]))

    for llm in llms:
        print(llm)
        if not os.path.isdir(f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}"):
            os.makedirs(f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}")

        not_modelled_models = []

        for dataset in tqdm(datasets):
            target_dir = f"{get_folder_path(Folder.DATA)}/llm_run{run}/{llm}/{dataset}"
            print("Start test with dataset:", dataset)
            benchmark = Benchmark(datasets[dataset], llm=llm)
            not_modelled = benchmark.model_processes(target_dir)
            not_modelled_models.extend(not_modelled)

        print("Due to a timeout the following models are not modelled or the invalid model is not corrected:",
              not_modelled_models)

        #with open(f"{get_folder_path(Folder.DATA)}/test_llm_temp_01_run{run}/{llm}/not_modelled.txt", "w") as f:
        #    f.write(str(not_modelled_models))



def check_quality_llms(run, evaluation, expert_dataset=False):
    """
    Allows to analyze the quality of the LLM-generated BPMNs.
    The analysis results are saved in a csv file llm_results_runx_evaluation (x= iteration, evaluation = evaluation type) in the folder llm_runx
    run: int
        indicates the number of the iteration
    evaluation: string
        We allow for three types of granualrity to assess the quality of the LLM-generated BPMNs.
        quality_group_score: scores in the four groups sytactic, pragmatic and semantic quality, and validity
        detail: here the subgroup scores for the quality dimensions are calculated
        metrics: here each metric result is given
    expert_dataset:
        if true, the analysis is made for the expert comparison
    """
    datasets = dict()
    datasets["camunda"] = prepare_datasets.prepare_camunda()
    datasets["bpmn_and_text"] = prepare_datasets.prepare_text_and_bpmn()
    datasets["lre_new"] = prepare_datasets.prepare_lre_new()
    datasets["lre_old"] = prepare_datasets.prepare_lre_old()

    df = quality_check.compute_overall_quality_llms(datasets=datasets,
                                                    test_llm_dir=f"{get_folder_path(Folder.DATA)}/llm_run{run}",
                                                    target_file=f"{get_folder_path(Folder.DATA)}//llm_run{run}/llm_results_run{run}_{evaluation}.csv",
                                                    analyse_method=evaluation)

def human_expert_comparison(run, evaluation, expert_dataset=False):
    """
    Allows to analyze the quality of the LLM-generated BPMNs and compare the results to human experts.
    This is based on the dataset of the human experts.
    The analysis results are saved in a csv file llm_results_runx_evaluation (x= iteration, evaluation = evaluation type) in the folder llm_runx
    run: int
        indicates the number of the iteration
    evaluation: string
        We allow for three types of granualrity to assess the quality of the LLM-generated BPMNs.
        quality_group_score: scores in the four groups sytactic, pragmatic and semantic quality, and validity
        detail: here the subgroup scores for the quality dimensions are calculated
        metrics: here each metric result is given
    expert_dataset:
        if true, the analysis is made for the expert comparison
    """
    datasets = dict()
    datasets["experts"] = prepare_datasets.prepare_experts_comparison()

    df = quality_check.compute_overall_quality_llms(datasets=datasets,
                                                    test_llm_dir=f"{get_folder_path(Folder.DATA)}/expert_comparison/human_experts",
                                                    target_file=f"{get_folder_path(Folder.DATA)}/expert_comparison/human_experts/expert_results_{evaluation}.csv",
                                                    analyse_method=evaluation)

    df = quality_check.compute_overall_quality_llms(datasets=datasets,
                                                    test_llm_dir=f"{get_folder_path(Folder.DATA)}/llm_run{run}",
                                                    target_file=f"{get_folder_path(Folder.DATA)}//llm_run{run}/llm_results_run{run}_{evaluation}.csv",
                                                    analyse_method=evaluation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-function script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # arguments for quality check
    quality_parser = subparsers.add_parser("check_quality_llms")
    quality_parser.add_argument("run", type=str)
    quality_parser.add_argument("evaluation", type=str)

    quality_parser = subparsers.add_parser("generate_bpmn")
    quality_parser.add_argument("run", type=str)

    args = parser.parse_args()
    if args.command == "generate_bpmn":
        generate_bpmn(run=args.run)
    elif args.command == "check_quality_llms":
        check_quality_llms(run=args.run, evaluation=args.evaluation)
    elif args.command == "human_expert_comparison":
        human_expert_comparison(run=args.run, evaluation=args.evaluation)
