# Python built-in modules
import os
from itertools import combinations
import copy

# Other modules
import pytest
import yaml


def get_all_keys(dictionary, parent_key=""):
    keys = []
    for key, value in dictionary.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        keys.append(current_key)
        if isinstance(value, dict):
            keys.extend(get_all_keys(value, current_key))
    return keys


def test_verify_the_number_of_translations_same():
    with open(
        os.path.join("src/scratch/config", "translation.yaml"), "r"
    ) as config_file:
        config = yaml.safe_load(config_file)

    language = list(config.keys())

    # title, album_info, ... (key 개수 확인)
    for lang1, lang2 in combinations(language, 2):
        lang1_key = get_all_keys(config[lang1])
        lang2_key = get_all_keys(config[lang2])

        assert len(lang1_key) == len(lang2_key)

        for key in lang1_key:
            inner1 = copy.deepcopy(config[lang1])
            inner2 = copy.deepcopy(config[lang2])
            split_key = key.split(".")
            for sk in split_key:
                inner1 = inner1[sk]
                inner2 = inner2[sk]

            assert type(inner1) == type(inner2)

            if type(inner1) == list:
                assert len(inner1) == len(inner2)
