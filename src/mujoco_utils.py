import mujoco
import mujoco.viewer
import numpy as np
import time

def mujoco_load(model_path):
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        raise RuntimeError(f"Error: failed to load mujoco model: {model_path}")
    data = mujoco.MjData(model)
    return model, data

