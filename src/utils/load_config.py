import yaml


def load_cfg():
    with open(
        "config/config.yaml",
        "r",
    ) as config_file:
        return yaml.safe_load(config_file)