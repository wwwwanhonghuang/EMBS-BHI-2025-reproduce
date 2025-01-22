


import yaml
def load_yaml_config(yaml_file):
    """
    Load a YAML configuration file.

    Args:
        yaml_file (str): Path to the YAML file.
    
    Returns:
        dict: Parsed YAML data as a Python dictionary.
    """
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)