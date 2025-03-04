import etdmap
import etdtransform
import yaml
import os


def load_config_file(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def find_config_file():
    """
    Find the config.yaml file. Priority:
    1. Current working directory (project directory).
    2. Directory where the script is located.
    Returns the path to the config.yaml if found.
    """
    # Check in the current working directory
    cwd_config = os.path.join(os.getcwd(), 'config.yaml')
    if os.path.exists(cwd_config):
        return cwd_config
    
    # Check in the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_config = os.path.join(script_dir, 'config.yaml')
    if os.path.exists(script_config):
        return script_config
    
    raise FileNotFoundError("config.yaml not found in project directory or script location.")

def init_config(local_options_name = "local_configuration"):
    config_path = find_config_file()
    config = load_config_file(config_path)
    return set_config(config)

def set_config(config, local_options_name = "local_configuration"):
    local_options = {}    
    package_options = {
        'etdmap_configuration': etdmap.options,
        'etdtransform_configuration': etdtransform.options
    }

    # Set the option for the analysis repo, etdmap and etdtransform
    # It is a dynamic way of doint the following for all configurations 
    # in the config.yaml: 
    # etdmap.options.mapped_folder_path = config['mapped_folder_path']
    for section, settings in config.items():
        # set the options for the packages (etdmap & etdtransform)
        if section in package_options: 
            section_options = package_options[section]
            # Dynamically set attributes for the corresponding object
            for key, value in settings.items():
                # Only assign if the value is not an alias reference
                setattr(section_options, key, value)
        # set the options for the workflow repo
        elif section == 'analysis_configuration':
            for key, value in settings.items():
                local_options[key] = value

    return local_options