import yaml


def load_cfg(path="config/config.yaml"):
    with open(path, "r") as config_file:
        return yaml.safe_load(config_file)
