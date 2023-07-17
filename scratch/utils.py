# Python built-in modules
import os

# Other modules
import yaml


def load_yaml(yaml_path: str, key: str = None) -> dict:
    yaml_path = os.path.join("config", yaml_path)
    with open(yaml_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config[key] if key else config
