#!/usr/bin/env python3
"""
快速测试 GraspControl 类是否有问题
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import numpy as np
from src.grasp_control import GraspControl
from src.grasp_planner import GraspPlanner
from src.grasp_state import StateStruct

try:

    
    print("✓ 所有导入成功")
    
    # 加载模型
    model_path = "shadow_hand/scene_left.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"✓ 模型加载成功: {model_path}")
    
    # 创建 MuJoCo data
    data = mujoco.MjData(model)
    print(f"✓ MuJoCo data 创建成功")

    # for i in range(model.nbody):
    #     body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    #     print(f"Body {i}: {body_name}")
    #     if body_name is not None:
    #         pos = data.xpos[i]
    #         quaternion = data.xquat[i]
    #         print(f"  Position: {pos}, Quaternion: {quaternion}")
    #         comp = pos.tolist() + quaternion.tolist()
    #         print(f"  Combined: {comp}")


    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 创建初始状态
            state = StateStruct(model, data, position=[0.0, 0.0, 0.3], quaternion=[1.0, 0.0, 0.0, 0.0], grasp=0.5, spread=0.5)
            print(f"✓ 初始状态创建成功")



            state.get_state_dic()


            
            # 创建 GraspPlanner
            # planner = GraspPlanner(state, model, data, "bottle", hand_body_prefix='lh_')
            # print(f"✓ GraspPlanner 创建成功")
            
            # 创建 GraspControl
            grasp_ctrl = GraspControl(model, data)
            print(f"✓ GraspControl 创建成功")

            print("act_names")
            print(grasp_ctrl.act_name_list)
            print()

            print("act_name_to_id")
            print(grasp_ctrl.act_name_to_id)

            set_val = np.random.uniform(1.3, 0.01)
            set_val = np.clip(set_val, 0.0, 1.5708)
            set_act_name = "lh_FFJ0"
            print(f"original value: {grasp_ctrl.get_act_val(set_act_name)} ")
            grasp_ctrl.set_act_val(set_act_name, set_val)
            print(f"value now: {grasp_ctrl.get_act_val(set_act_name)} ")



            mujoco.mj_step(model, data)
            

            print("\n✅ 所有测试通过！")

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        
except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
