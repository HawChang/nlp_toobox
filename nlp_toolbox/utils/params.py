# -*- coding: utf-8 -*
"""maintain a dictionary of parameters"""

import json
import logging


def evaluate_file(filename):
    """evaluate_file"""
    logging.info(filename)
    with open(filename, "r") as evaluation_file:
        return evaluation_file.read()


def from_file(filename):
    """from_file"""
    if filename is None:
        return dict()
    file_dict = json.loads(evaluate_file(filename), strict=False)
    #logging.info(json.dumps(file_dict, indent=4, sort_keys=True))
    return file_dict


def replace_none(params):
    """replace_none"""
    if params == "None":
        return None
    elif isinstance(params, dict):
        for key, value in params.items():
            params[key] = replace_none(value)
            if key == "split_char" and isinstance(value, str):
                try:
                    value = chr(int(value, base=16))
                    logging.debug("ord(value): {} ".format(ord(value)))
                except Exception:
                    pass
                params[key] = value
        return params
    elif isinstance(params, list):
        return [replace_none(value) for value in params]
    return params
