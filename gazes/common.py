"""Contains various function used throughout this project."""
from typing import Dict
import os
import json

from . import settings


def get_secrets(entry_name: str, secret_file_name: str = 'secret') -> Dict[str, str]:  # noqa: E501
    """
    Open the secrets file and return the requested entry.
    """
    with open(os.path.join(settings.root_dir, secret_file_name)) as f:
        return json.load(f)[entry_name]


def get_configs(entry_name: str, config_file_name: str = 'config',
                config_default_file_name: str = 'default.config'):
    """
    Open the config file and return the requested entry.
    If no config file is found, open default.config.
    """

    try:
        with open(os.path.join(settings.root_dir, config_file_name)) as f:
            content = json.load(f)
    except FileNotFoundError:
        with open(os.path.join(settings.root_dir, config_default_file_name)) as f:  # noqa: E501
            content = json.load(f)
    return content[entry_name]


def search_dict(dictionary, search_for, nested=False):
    """
    Search if dictionary value contains certain string search_for. If
    nested=True multiple levels are traversed.
    """
    for k in dictionary:
        if nested:
            for v in dictionary[k]:
                if search_for in v:
                    return k
                elif v in search_for:
                    return k
        else:
            if search_for in dictionary[k]:
                return k
            elif dictionary[k] in search_for:
                return k
    return None
