import json
import os

def load_config() -> dict:
    """
    Load configuration from config.json in the project root.
    Returns a dictionary with configuration values and absolute paths.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(root_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Resolve paths to absolute
    # We iterate over the keys to ensure we don't break if keys are missing, 
    # but ideally we should validate the config.
    if 'paths' in config:
        for key, value in config['paths'].items():
            # If it's already absolute, join handles it (on Windows it might be tricky if mixed separators, but usually fine)
            # If it's relative, it joins with root_dir
            config['paths'][key] = os.path.normpath(os.path.join(root_dir, value))
            
    return config
