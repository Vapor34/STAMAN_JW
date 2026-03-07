"""
MuJoCo simulator utilities

Provides basic interface for loading and initializing MuJoCo models
"""

import mujoco


def mujoco_load(model_path):
    """
    Load a MuJoCo model from an XML file path
    
    Args:
        model_path (str): Path to the MuJoCo XML model file
        
    Returns:
        tuple: (model, data) - MuJoCo model and data objects
        
    Raises:
        RuntimeError: If the model file cannot be loaded
    """
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        raise RuntimeError(f"Error: failed to load mujoco model: {model_path}") from e
    data = mujoco.MjData(model)
    return model, data
