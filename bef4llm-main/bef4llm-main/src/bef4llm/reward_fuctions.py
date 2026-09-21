from epmc_llm.KICoPro.translation.bpil_json_v3_to_xml import TranslationBpilToXmlV3
from epmc_llm.synactic_quality.synactic_quality_check import SyntacticQualityCheckBPMN
from epmc_llm.semantic_quality.semantic_quality_check import SemanticQualityCheckBPMN
from epmc_llm.pragmatic_quality.pragmatic_quality_check import PragmaticQualityCheckBPMN
from epmc_llm.semantic_quality.similarity.language_similarity.lanuage_utils import Language
from epmc_llm.process_models.importer.bpmn_importer import load_diagram_from_et_tree
from epmc_llm.validation.validation import validate_bpmn

from xml.etree import ElementTree as ET


import io

def translate_and_import_bpil(bpil):
    """
    Translates and imports a BPIL (Business Process Instruction Language) document into a BPMN diagram.
    If the input is not already in XML format, it will first translate the BPIL into XML
    using the `TranslationBpilToXmlV3` translator. The resulting XML or input XML is then validated as a BPMN-compliant
    document. Upon successful validation, the XML input is parsed into a BPMN diagram model.

    Parameters
    ----------
    bpil : str
        The input BPIL document or XML string to be translated and imported.

    Returns
    -------
    model : Any
        A BPMN diagram model object generated from the provided BPIL or XML input.

    Raises
    ------
    Exception
        If the provided BPIL or XML input is not valid BPMN after validation.
    """
    if not bpil.startswith("<?xml"):
        translator = TranslationBpilToXmlV3()
        xml = translator.translate_bpil_to_xml(bpil)

        if not validate_bpmn(xml):
            raise Exception

        f = io.StringIO(ET.tostring(xml, encoding="utf-8", xml_declaration=True).decode())
    else:
        if not validate_bpmn(bpil):
            raise Exception
        f = io.StringIO(bpil)

    tree = ET.iterparse(f)
    model = load_diagram_from_et_tree(tree)
    return model

def reward_total(bpil, ground_truth=None, lang="en"):
    """
    Compute the total reward score for a given BPIL model.

    This function translates and imports the provided BPIL model into a BPMN structure,
    then computes its syntactic and pragmatic quality scores. If a ground truth model is
    provided, it additionally computes the semantic quality score of the input model
    against the ground truth, and averages these scores to compute the overall reward.

    Parameters
    ----------
    bpil : str
        The input BPIL model to translate and evaluate.
    ground_truth : str, optional
        The reference ground truth BPIL model for semantic quality evaluation. Default is None.

    Returns
    -------
    float
        The average reward score, computed from the syntactic, pragmatic, and optionally semantic scores.
        Returns 0 if an exception occurs during processing.
    """
    try:
        bpmn = translate_and_import_bpil(bpil)

        syntactic_check = SyntacticQualityCheckBPMN(bpmn)
        syn_score = syntactic_check.compute_syntax_score()

        pragmatic_check = PragmaticQualityCheckBPMN(bpmn)
        prag_score = pragmatic_check.pragmatic_quality_check()

        if ground_truth is not None:
            if lang == "de":
                lang = Language.GERMAN
            else:
                lang = Language.ENGLISH

            gt_bpmn = translate_and_import_bpil(ground_truth)
            semantic_check = SemanticQualityCheckBPMN(model=bpmn, reference_model=gt_bpmn, lang=lang)
            semantic_score = semantic_check.semantic_quality_check()

            return (syn_score + prag_score + semantic_score) / 3

        else:
            return (syn_score + prag_score) / 2

    except Exception as e:
        print(e)
        return 0

def reward_syntactic(bpil):
    """
    Calculate and return the syntactic quality score of a given BPIL (Business Process
    Improvement Language) model.

    This function translates the BPIL model into a BPMN (Business Process Model and
    Notation) representation, then evaluates the syntactic quality of the resulting BPMN
    model. If an error occurs during the process, it safely handles the exception and returns
    a default score of 0.

    Parameters
    ----------
    bpil : str
        A string representation of the BPIL model to be evaluated.

    Returns
    -------
    float
        The computed syntactic quality score of the given model, or 0 if an exception is
        encountered.
    """
    try:
        bpmn = translate_and_import_bpil(bpil)

        syntactic_check = SyntacticQualityCheckBPMN(bpmn)
        syn_score = syntactic_check.compute_syntax_score()

        return syn_score

    except Exception as e:
        return 0

def reward_pragmatic(bpil):
    """
    Calculates the pragmatic quality score of a BPMN process derived from BPIL.

    This function takes a BPIL (Business Process Intermediate Language) process model,
    translates it into a BPMN (Business Process Model and Notation) format, and evaluates
    its pragmatic quality. The pragmatic quality is determined through specific checks
    implemented in the `PragmaticQualityCheckBPMN` class. If any error occurs during the
    process, a default score of 0 is returned.

    Parameters
    ----------
    bpil : str
        A BPIL representation of the processmodel to be evaluated.

    Returns
    -------
    int
        Pragmatic quality score of the BPMN-derived model. Returns 0 in case of any
        exceptions during processing or evaluation.
    """
    try:
        bpmn = translate_and_import_bpil(bpil)

        pragmatic_check = PragmaticQualityCheckBPMN(bpmn)
        prag_score = pragmatic_check.pragmatic_quality_check()

        return prag_score

    except Exception as e:
        return 0

def reward_semantic(bpil, ground_truth, lang="en"):
    """
    Calculates the semantic similarity score between a given BPIL process model and its
    ground truth using semantic quality checks.

    Parameters
    ----------
    bpil : Any
        The BPIL process model to be evaluated.
    ground_truth : Any
        The ground truth BPIL process model to compare against.

    Returns
    -------
    float
        A semantic similarity score based on the comparison between `bpil` and
        `ground_truth`. Returns 0 if an error occurs during the computation.

    Raises
    ------
    Exception
        If an error occurs in translating, importing, or performing the semantic quality
        check.
    """
    try:
        bpmn = translate_and_import_bpil(bpil)
        gt_bpmn = translate_and_import_bpil(ground_truth)

        if lang == "de":
            lang = Language.GERMAN
        else: lang = Language.ENGLISH

        semantic_check = SemanticQualityCheckBPMN(bpmn, gt_bpmn, lang=lang)
        semantic_score = semantic_check.semantic_quality_check()

        return semantic_score

    except Exception as e:
        return 0