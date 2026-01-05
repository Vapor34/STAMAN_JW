import mujoco
import mujoco.viewer
import numpy as np
import time

model_path = "shadow_hand/scene_left.xml"

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_start = time.time()
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


            # 维持频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


except Exception as e:
    print(f"发生错误：{e}")
