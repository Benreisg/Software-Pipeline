from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import look_for_directory, get_folder_path
from bef4llm.process_models.importer.bpmn_importer import load_bpmn_from_directory, load_diagram_from_xml
from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language
from bef4llm.datasets.model_collection import download_any_format, DatasetCollection, download_git_folder
import os
from tika import parser

"""
In this file different datasets are prepared to be usable for the llm_comparison, so the text-model pairs are returned
Text model pairs should always be returned in the following format, since for some texts there are multiple models given:
{
    text_id_1: (language, text, [model1, model2, ...]),
    text_id_2: (language, text, [model1, model2, ...]),
    ....
}
"""
def prepare_camunda():
    """
    Prepares Camunda dataset for llm_comparison with model text pairs
    --------
    Returns:
        text_model_pairs: list
            list of tuples: (file_name, language, text, reference models)
    """

    def clean_text_camunda(text_path):
        """
        cleans the text, i.e. deletes Hints or instructions naming specific pools that should be modelled.
        adds specific formal like, background:..., description:...
        """
        txt_text_path = text_path.split(".")[0] + ".txt"
        if not os.path.exists(txt_text_path):
            text = parser.from_file(text_path)["content"]
            text = text.replace("\n", "")
            # Exercise 1-3 english
            if "Please model the following process" in text:
                rest, text = text.split("Please model the following process")

                if text.startswith(":"):
                    text = text[1:]

                background = ""

                # extract background information without heading etc and add to end
                if "Background" in rest:
                    if "Exercise:" in rest:
                        rest = rest.replace("Exercise: ", "")

                    background = "Background" + rest.split("Background")[1]

                if "Please use" in text:
                    text = text.split("Please use")[0]

                text = text + background
                text = text.replace("Background", "\nBackground: \n")

            # Exercise 4 english
            if "Exercise: Create a model of the following optimised process" in text:
                rest, text = text.split("Exercise: Create a model of the following optimised process")
                background = ""
                if "Background" in rest:
                    background = "\nBackground: \n" + rest.split("Background")[1]

                if "Chef (meal preparation)" in text:
                    text = text.split("Chef (meal preparation)")[1]

                text = text + background

            # Exercisr 1, 3 german
            if "Bitte modellieren Sie folgende Prozessbeschreibung:" in text:
                text = text.split("Bitte modellieren Sie folgende Prozessbeschreibung:")[1]
                text = text.replace("Hintergrund:", "\nHintergrund: \n")

            #exercise 7
            if "Aufgabe: Modellieren Sie folgenden Prozess" in text:
                rest, text = text.split("Aufgabe: Modellieren Sie folgenden Prozess")
                if rest.startswith(":"):
                    rest = text[1:]

                if "Hinweis:" in text:
                    text = text.split("Hinweis:")[0]

                background = "\nHintergund: \n"
                if "Hintergrund" in rest:
                    background = background + rest.split("Hintergrund")[1]

                text = text + background

            # exercise 8
            if "Aufgabe: Modellieren Sie folgenden optimierten Prozess" in text:
                rest, text = text.split("Aufgabe: Modellieren Sie folgenden optimierten Prozess")

                if "Koch (Mahlzeitzubereitung)" in text:
                    text = text.split("Koch (Mahlzeitzubereitung)")[1]

                background = "\nHintergrund: \n"
                if "Hintergrund" in rest:
                    background = background + rest.split("Hintergrund")[1]

                text = text + background

            with open(txt_text_path, "w") as text_file:
                text_file.write(text)

        else:
            # write text to txt file, such that tika service must not be used every time
            with open(txt_text_path, "r") as f:
                text = f.read()

        return text

    def extract_files(lang, text_model_pairs):
        dir_path = look_for_directory("BPMN for Research")
        model_dir = ""
        text_dir = ""
        if lang == Language.ENGLISH:
            dir_path = os.path.join(dir_path, "English")
            model_dir = "03-Solution"
            text_dir = "01-Exercise"
        elif lang == Language.GERMAN:
            dir_path = os.path.join(dir_path, "German")
            model_dir = "03-Musterlösung"
            text_dir = "01-Aufgabenstellung"

        for exercise in os.listdir(dir_path):
            # exclude hidden directories
            if not exercise.startswith('.'):
                ex_dir_path = os.path.join(dir_path, exercise)
                text_file = os.listdir(f"{ex_dir_path}/{text_dir}")[0]
                text_path = f"{ex_dir_path}/{text_dir}/{text_file}"
                #text = parser.from_file(f"{ex_dir_path}/{text_dir}/{text_file}")["content"]
                clean_text = clean_text_camunda(text_path)
                models = load_bpmn_from_directory(f"{ex_dir_path}/{model_dir}")
                text_model_pairs[exercise] = (lang, clean_text, models)

        return text_model_pairs

    # add dowlooad files if not yet done
    if not look_for_directory("BPMN for Research"):
        data = DatasetCollection.CAMUNDA.value
        download_any_format(url=data["link"],
                            file_name=data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), data["destination_dir"]),
                            compressed=True)
    text_model_pairs = dict()
    extract_files(Language.ENGLISH, text_model_pairs)
    extract_files(Language.GERMAN, text_model_pairs)

    return text_model_pairs

def prepare_text_and_bpmn():
    """
    Prepares text and bpmn dataset for llm_comparison with model text pairs
    --------
    Returns:
        text_model_pairs: list
            list of tuples: (file_name, language, text, reference models)
    """

    def clean_text_text_and_bpmn(text):
        lines = text.splitlines()
        filtered_lines = [line for line in lines if not line.startswith("Title") or not line.startswith("Category")]
        return "\n".join(filtered_lines)

    if not look_for_directory("text_and_bpmn"):
        data = DatasetCollection.TEXT_AND_BPMN.value
        download_any_format(url=data["link"],
                            file_name=data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), data["destination_dir"]),
                            compressed=True)

    path = f"{get_folder_path(Folder.DATA)}/models/text_and_bpmn/bpmn"

    text_model_pairs = dict()
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            with open(f"{path}/{file}", 'r') as f:
                text = f.read()

            filename = file.split(".")[0]
            model_list = []
            for model in os.listdir(os.path.join(path, filename)):
                # iter over all models
                if model.endswith(".txt"):
                    continue

                model_name = model.split(".")[0]

                # model quality is not documented
                if not os.path.isfile(f"{path}/{filename}/{model_name}.quality.txt"):
                    continue

                # select only models with quality 5
                with open(f"{path}/{filename}/{model_name}.quality.txt", 'r') as f:
                    quality = f.read()

                if quality == "5":
                    model = load_diagram_from_xml(f"{path}/{filename}/{model}")
                    model_list.append(model)

            text = clean_text_text_and_bpmn(text)
            text_model_pairs[filename] = (Language.ENGLISH, str(text), model_list)

    return text_model_pairs

def prepare_lre_new():
    """
    Helper function for text_model_pairs_lre

    Returns:
    -------
    text_model_pairs: list
        list of tuples: (file_name, language, text, reference models)
    """
    lre_new_data = DatasetCollection.LRE_NEW.value
    if not look_for_directory("lre_new"):
        download_git_folder(url=lre_new_data["link"],
                            file_name=lre_new_data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), lre_new_data["destination_dir"]),
                            download_folder=lre_new_data["download_folder"],
                            download_dir=lre_new_data["download_dir"],
                            delete_folder=lre_new_data["delete_folder"])

    lre_new_folder = os.path.join(get_folder_path(Folder.DATA), lre_new_data["destination_dir"])
    text_model_pairs = text_model_pairs_lre(os.path.join(lre_new_folder, "NewDataset"))

    return text_model_pairs




def prepare_lre_old():
    """
    Helper function for text_model_pairs_lre

    Returns:
    -------
    text_model_pairs: list
        list of tuples: (file_name, language, text, reference models)
    """
    lre_original_data = DatasetCollection.LRE_ORIGINAL.value
    if not look_for_directory("lre_original"):
        download_git_folder(url=lre_original_data["link"],
                            file_name=lre_original_data["filename"],
                            dest_dir=os.path.join(get_folder_path(Folder.DATA), lre_original_data["destination_dir"]),
                            download_folder=lre_original_data["download_folder"],
                            download_dir=lre_original_data["download_dir"],
                            delete_folder=lre_original_data["delete_folder"])

    lre_original_folder = os.path.join(get_folder_path(Folder.DATA), lre_original_data["destination_dir"])
    text_model_pairs = text_model_pairs_lre(os.path.join(lre_original_folder, "OriginalDataset"))

    return text_model_pairs


def text_model_pairs_lre(lre_folder):
    """
    Prepares LRE (orginial and new) dataset for llm_comparison with model text pairs

    Parameters
    --------
    lre_folder: str
        Indicates which LRE dataset should be prepared

    Returns:
    -------
    text_model_pairs: list
        list of tuples: (file_name, language, text, reference models)
    """
    camunda = ["Dispatch-of-goods", "Recourse", "Self-service-restaurant"]
    text_model_pairs = dict()
    for file in os.listdir(f"{lre_folder}/Texts"):
        try:
            if not file.startswith("."):
                filename = file.split(".")[0]

                # some camunda files are added -> no duplicates
                if filename in camunda and not camunda:
                    continue

                with open(f"{lre_folder}/Texts/{file}", 'r') as f:
                    text = f.read()

                text = text.replace("\n", " ")

                model = load_diagram_from_xml(f"{lre_folder}/Models/{filename}.bpmn")

                text_model_pairs[filename] = (Language.ENGLISH, text, [model])
        except Exception as e:
            print(f"For file {file} no model text_pair was built due to an error: {e}")

    return text_model_pairs

def prepare_experts_comparison():
    """
    Prepares Expert dataset for human comparison with model text pairs
    --------
    Returns:
        text_model_pairs: list
            list of tuples: (file_name, language, text, reference models)
        """
    expert_folder = os.path.join(get_folder_path(Folder.DATA), "models", "Expert_BPMN")
    text_model_pairs = dict()
    for file in os.listdir(os.path.join(expert_folder, "textual_descriptions")):
        try:
            if not file.startswith("."):
                filename = file.split(".")[0]

                with open(f"{expert_folder}/textual_descriptions/{file}", 'r') as f:
                    text = f.read()

                text = text.replace("\n", " ")

                if filename == "schufa":
                    model_1 = load_diagram_from_xml(f"{expert_folder}/ground_truth_bpmn/{filename}1.bpmn")
                    model_2 = load_diagram_from_xml(f"{expert_folder}/ground_truth_bpmn/{filename}2.bpmn")
                    model = [model_1, model_2]
                else:
                    model = load_diagram_from_xml(f"{expert_folder}/ground_truth_bpmn/{filename}.bpmn")



                text_model_pairs[filename] = (Language.ENGLISH, text, [model])

        except Exception as e:
            print(f"For file {file} no model text_pair was built due to an error: {e}")

    print(text_model_pairs)
    return text_model_pairs

















