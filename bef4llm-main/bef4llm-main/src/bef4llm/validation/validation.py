from bef4llm.resource_controller.path_helper import look_for_directory, look_for_file
from xml.etree import ElementTree
import xmlschema


def validate_bpmn(bpmn_path):
    """
    Validates whether a bpmn is valid based on the XSD schema.

    Parameters
    ----------
    bpmn_path : str
        path to BPMN xml file

    Returns
    -------
    bool
        True if valid, False otherwise

    """

    xsd_file = look_for_file("BPMN20.xsd")
    schema = xmlschema.XMLSchema(xsd_file)

    try:
        if schema.is_valid(bpmn_path):
            #print("The BPMN is valid")
            return True
        else:
            #errors = schema.validate(bpmn_path)
            #print(errors)
            return False
    except xmlschema.XMLSchemaValidationError as e:
        return False
    except xmlschema.XMLSchemaParseError as e:
        return False
    except ElementTree.ParseError as e:
        return False
    except Exception as e:
        print(e)
        return False


def validate_bpmn_with_error_message(bpmn_path):
    """
    Validates whether a bpmn is valid based on the XSD schema.
    If BPMN XML is not valid, a list with all errors is returned.

    Parameters
    ----------
    bpmn_path : str
        path to BPMN xml file

    Returns
    -------
    list of str
        list of error messages
    bool
        True if valid, False otherwise

    """
    xsd_file = look_for_file("BPMN20.xsd")
    schema = xmlschema.XMLSchema(xsd_file)

    try:
        #schema.validate(bpmn_path)
        errors = list(schema.iter_errors(bpmn_path))
        if errors == None or len(errors) == 0:
            return "", True

        else:
            error_msg = ""
            for error in errors:
                error_msg += f"- {error.reason}\\n"

            return error_msg, False

    except xmlschema.XMLSchemaValidationError as e:
        return e.reason, False
    except xmlschema.XMLSchemaParseError as e:
        return e.message, False
    except ElementTree.ParseError as e:
        return e.msg, False
    except Exception as e:
        return str(e), False