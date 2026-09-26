import re

from bef4llm.llm_comparison import prepare_datasets
from bef4llm.llm_connection.connect_llms import ConnectLLMs
import bef4llm.llm_comparison.promting_helper as prompts
from bef4llm.validation.validation import validate_bpmn_with_error_message, validate_bpmn
from bef4llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language
from bef4llm.definitions import Folder
from bef4llm.resource_controller.path_helper import get_folder_path
from bef4llm.process_models.importer.bpmn_importer import load_diagram_from_xml

import os
from tqdm import tqdm
import concurrent.futures


class Benchmark():
    """
    Class for generating BPMN diagrams, based on given Datasets
    """
    def __init__(self, dataset, llm:str, sys_msg=None):

        self.text_model_pairs = dataset
        self.syntax_scores = []
        self.pragmatic_scores = []
        self.semantic_scores = []
        self.sys_msg = None
        self.llm_model = llm
        self.llm = None
        self.current_lang = list(self.text_model_pairs.values())[0][0]
        if not sys_msg:
            self.change_sys_msg_lang(self.current_lang)
        # init llm


    def model_processes(self, target_dir):
        """
        Here for each text-model pair in the dataset a coresponding process model is modeled by the LLM

        Parameters
        ----------
        target_dir : str
            directory, where to BPMN is saved to

        Returns:
        -------
        not_modelled: list
            list of BPMN names, that are not modelled (e.g. due to timeout)
        """
        # validation check mit einem Loop
        not_modelled = []
        # not_valid_first_time = []
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        for pair in tqdm(self.text_model_pairs):
            if not os.path.isfile(f"{target_dir}/{pair}.bpmn"):
                # check if system message is in right language
                lang, text, r_models = self.text_model_pairs[pair]

                # adapt language of the system message
                #if self.current_lang != lang:
                    #self.change_sys_msg(lang)

                model_succ = self.llm_modelling(text, pair, target_dir)

                if not model_succ:
                    not_modelled.append(pair)


        return not_modelled


    def llm_modelling(self, text, pair, target_dir):
        """
        Actual interaction with the llm. To ensure the xml is valid, a validation loop is added
        If the xml is still not valid after an second try a empty file is saved.
        Each process model is saved in a file in the target directory.

        Parameters
        ----------
        text: str
            textual description of the BPMN
        pair: str
            name of the text-model pair
        target_dir: str
            directory, where to BPMN is saved to

        Returns:
        -------
        not_modelled: list
            list of BPMN names, that are not modelled (e.g. due to timeout)
        """
        # adapt prompt here
        prompt = (prompts.modeling_prompt + text).strip()
        response = self.chat_with_timeout(prompt=prompt)
        if not response:
            print(f"No process is modelled due to timeout: Model = {pair}")
            return False, False


        xml = self.get_xml_from_response(response)
        with open(f"{target_dir}/{pair}.bpmn", 'w') as f:
            f.write(xml)

        error, valid = validate_bpmn_with_error_message(f"{target_dir}/{pair}.bpmn")

        # validation loop
        if not valid:
            # adapt prompting language to textual description
            #if self.current_lang == Language.GERMAN:
                #prompt = prompts.inavlid_xml_ger + "\n" + error
            #else:
                #prompt = prompts.inavlid_xml_eng + "\n" + error

            prompt = prompts.inavlid_xml + "\n" + error

            response = self.chat_with_timeout(prompt=prompt)
            if not response:
                print(f"Process model is not corrected due to timeout: Model = {pair}")
                return False

            xml = self.get_xml_from_response(response)

            with open(f"{target_dir}/{pair}.bpmn", 'w') as f:
                f.write(xml)

        # ensures that each process model is modelled without interferences from previous modelling
        self.llm.reset_chat_history()


        return True

    def change_sys_msg_lang(self, lang, timeout=300):
        """
        Allows to adapt the language of the system message

        Parameters
        ----------
        lang: Language
            indicates the language of the system message
        """
        if lang == Language.ENGLISH:
            self.sys_msg = prompts.sys_msg # + "\\no_think" disable thinking mode
        elif lang == Language.GERMAN:
            raise Exception("Language is not implemented yet")

        if not self.llm:
            self.llm = ConnectLLMs(llm_modell=self.llm_model, sys_msg=self.sys_msg, timeout=timeout)
        else:
            self.llm.init_sys_role(self.sys_msg)
        self.current_lang = lang

    def get_xml_from_response(self, response):
        """
        Filters the xml from the response (content value in reponse)

        Parameters
        ----------
        response: str
            content string from the response from LLM
        """

        xml = response
        try:
            # deepseek specific
            if "</think>" in response:
                response = response.split("</think>")[-1]

            if not response.startswith("<?xml") or not response.endswith("definitions>"):
                if "<?xml" in response and "definitions>" in response:
                    xml = re.search(r"<\?xml[\s\S]*?definitions>", response).group(0)

                elif "<?xml" in response and ":process>" in response:
                    xml = re.search(r"<\?xml[\s\S]*?process>", response).group(0)
                    xml = xml + "\n" + "</bpmn:definitions>"
            else:
                xml = response

            return xml
        except Exception as e:
            print("Error while getting xml from response ", e)
            return response

    def chat_with_timeout(self, prompt):
        """
        Chat with LLM.
        Implemented timeout due to problem of endless generation in Ollama

        Parameters:
        -----------
        prompt: str
            user prompt, that should be sent to the LLM

        Returns
        -------
        response: str
            Content string of response from the LLM
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                response = self.llm.chat_with_history(role="user", content=prompt)
                return response
            except TimeoutError:
                return None
            # catch possible exceptions
            except Exception as e:
                print(e)
                return None

