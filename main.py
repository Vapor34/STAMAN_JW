import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer
import argparse

from src.mujoco_utils import mujoco_load
from src.grasp_state import StateStruct
from src.grasp_planner import GraspPlanner
from src.grasp_control import GraspControl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="shadow_hand/scene_left.xml", help='mujoco models path')
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    model, data = mujoco_load(args.model_path)
    controler = GraspControl(model, data)

    # 初始状态 - 使用 StateStruct
    initial_state = StateStruct(
        position=[0.4, 0.4, 0.3],
        quaternion=[1.0, 0.0, 0.0, 0.0]
    )
    initial_guess = initial_state.to_array()

    planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')
    planner.steps = 5000

    # 动画控制变量
    planning_done = False
    target_state = initial_state.copy()  # 规划得到的目标状态
    current_grasp_val = 0.0
    execution_start_time = 0.0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 规划阶段
            if data.time < 0.002:
                print("\n[状态] 正在规划最佳抓取点...")
                
                # 从初始状态开始扰动
                planner.state = initial_guess.copy()
                planner.state[0:3] += np.random.uniform(-0.05, 0.05, 3)
                best_pose, energy = planner.anneal()

                
                # 将规划结果加载到 StateStruct
                target_state.from_array(best_pose)
                # target_state.grasp = 0.8
                print(f"[完成] 目标 Grasp: {target_state.grasp:.2f}, Spread: {target_state.spread:.2f}")
                print(f"[完成] 目标位置: ({target_state.x:.3f}, {target_state.y:.3f}, {target_state.z:.3f})")
                
                current_grasp_val = 0.0
                execution_start_time = time.time()
                planning_done = True
                
                data.time = 0.002

            # 执行阶段
            if planning_done:
                anim_time = time.time() - execution_start_time
                
                # 0.0s ~ 1.0s：手掌瞬移
                data.qpos[0:3] = target_state.get_position()
                data.qpos[3:7] = target_state.get_quaternion()
                
                # 1.0s ~ 3.0s：手指闭合
                grasp_start_time = 1.0
                grasp_duration = 2.0
                
                if anim_time > grasp_start_time:
                    grasp_progress = (anim_time - grasp_start_time) / grasp_duration
                    grasp_progress = np.clip(grasp_progress, 0.0, 1.0)
                    
                    current_grasp_val = target_state.grasp * grasp_progress
                    current_curl = target_state.curl * grasp_progress
                    current_thumb_flex = target_state.thumb_flex * grasp_progress
                else:
                    current_grasp_val = 0.0
                    current_curl = 0.0
                    current_thumb_base = 0.0
                    current_thumb_flex = 0.0
                    

                # 构造当前执行状态用于计算控制命令
                exec_state = StateStruct(
                    position=target_state.get_position(),
                    quaternion=target_state.get_quaternion(),
                    grasp=current_grasp_val,
                    curl=current_curl,
                    spread=target_state.spread,
                    thumb_base=target_state.thumb_base,
                    thumb_flex=current_thumb_flex,
                )


                # =====特定关节的固定角度（可选覆盖） =====
                # 如果需要固定某些拇指关节的角度，在这里指定
                controler.set_act_val('lh_THJ5', 0.5)  # 拇指末端关节
                controler.set_act_val('lh_THJ4', 1.0)  # 拇指近端关节
                # ====================================
                

                # 设置关节执行器的控制信号
                controler.set_hand_state(exec_state)
                controler.print_all_act_val()
                
                
                mujoco.mj_step(model, data)

            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
