# BEF4LLM

This repository contains the source code accompanying the paper:

*Assessing the Business Process Modeling Competences of Large Language Models*

Using these metrics, we developed a pipeline to analyze how well large language models (LLMs) can generate BPMN diagrams from textual descriptions. Our Evaluation used 105 text-BPMN pairs.

For comparison, we also assess the performance of human experts in process modeling using a smaller dataset of 9 text-model pairs.

## Installation
The installation via pip is not yet tested, so we recommend to clone the project and work with it. 
All required packages can be installed by using the requirements.txt, with the most import packages being: 
- ollama
- pandas
- networkx

**Note:** The LLM connection is based on Ollama, which in our setup runs on a private server and cannot be made publicly accessible.
To run the pipeline, you will need to set up your own Ollama service. Instructions for adapting the repository to your own Ollama instance can be found in the next paragraph "Assessing the abilities of LLMs to generate BPMNs".

## Structure and usage of the repository
The contribution of this repository is threefold, it allows assessing the ability of LLMs to generate BPMNs, a comparison to human experts and provides statistical test to analyze the results. 

### Assessing the abilities of LLMs to generate BPMNs 
To assess LLM's abilities to generate BPMNs two steps are necessary: The generation of the BPMNs, and the analysis of the LLM-generated BPMNs

Step 1: Generation of BPMNs by LLMs
We decided to use Ollama as an environment to run the LLMs. This enables easy setup and testing of multiple models in the same environment.
To be able to run generate BPMNs with a LLM, you need to set up Ollama on your machine/server. There you can download all LLMs you want to test. To run the LLMs you need to update the ```__init__``` method of the ConnectLLM class in ```llm_connection/connect_llms.py```.
Set the host parameter in the Client initialization to your Ollama server’s IP address. We do not provide a server. 
To run the generation, you can use the function ```generate_bpmn``` in ```compare_llms.py```. The list llms indicates which LLMs are used to generate LLMs. If you want to add a new LLM the ollame tag must be enetered.

Step 2: Assessment
To check the quality of the LLMs, the script ```check_quality_llms``` in ```compare_llms.py``` can be used. 
We assess the quality of BPMNs in the four categories of syntactic quality, pragmatic quality, semantic quality and validity. For each category a seperate score is computed.
Further, we allow for a more detailed analysis, as the evaluation method can be specified: 
- quality_group_score - scores for the four main categories
- detail - subgroup scores within each dimension
- metrics - individual metric results

### Comparison to human experts
The comparison to human experts is done in a similar way to the "Assessing the abilities of LLMs to generate BPMNs". The BPMNs are generated with the function generate_bpmns, with the parameter ```expert_dataset``` being set to ```True```. 
After that, the LLM-generated BPMNs and the BPMNs modeled by the experts can be assessed via ```check_quality_expert_dataset```.
The expert datset can be found in the folder ```data_human_comparison```.

### Statistical tests
The code for the statistical test can be found in the folder ```statistical_tests``` and is currently only available for the quality dimensions (syntactic, pragmatic and semantic quality). 
To run the statistical test for a quality dimension, you just need to run the corresponding file, e.g. for the syntactic quality dimension, you need to run ```syntactic_quality_analysis.py```
Further, we need to create a "dataset" for the statistical test, which is automatically created at the beginning of the tests. For each quality dimension, one file is created containing the score of each BPMN of each run for each LLM. If the BPMN is invalid or not generated, it is indicated by a "none" or "NaN" value.