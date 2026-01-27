#!/usr/bin/env python3
"""
快速测试 GraspControl 类是否有问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mujoco
    import numpy as np
    from src.grasp_control import GraspControl
    from src.grasp_planner import GraspPlanner
    from src.grasp_state import StateStruct
    
    print("✓ 所有导入成功")
    
    # 加载模型
    model_path = "shadow_hand/scene_left.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"✓ 模型加载成功: {model_path}")
    
    # 创建 MuJoCo data
    data = mujoco.MjData(model)
    print(f"✓ MuJoCo data 创建成功")
    
    # 创建初始状态
    state = StateStruct(position=[0.0, 0.0, 0.3], quaternion=[1.0, 0.0, 0.0, 0.0], grasp=0.5, spread=0.5)
    print(f"✓ 初始状态创建成功")
    
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
    print(grasp_ctrl.act_name_dic)

    set_val = 0.5
    set_act_name = "lh_FFJ0"
    print(f"original value: {grasp_ctrl.get_act_val(set_act_name)} ")
    grasp_ctrl.set_act_val(set_act_name, set_val)
    print(f"value now: {grasp_ctrl.get_act_val(set_act_name)} ")

    
    print("\n✅ 所有测试通过！")
    
except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
