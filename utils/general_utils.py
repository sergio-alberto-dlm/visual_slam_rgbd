import yaml

def load_config(path):
    """
    Loads config file.
    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg = yaml.full_load(f)

    return cfg 