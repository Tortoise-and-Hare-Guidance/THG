import os
import contextlib

import yaml


def chunk(l: list, n: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def deep_update(source: dict, overrides: dict):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value


@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield
