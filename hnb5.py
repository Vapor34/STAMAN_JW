import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer
import argparse

from src.mujoco_utils import mujoco_load
from src.grasp_state import StateStruct
from src.grasp_planner import GraspPlanner, get_synergy_mapping, get_tendon_actuator_map


def qpos_to_ctrl(model, data, planner, target_pose):
    """
    【正确版本】协同变量 → 目标关节角度（位置伺服）
    
    【关键认识】Shadow Hand使用 <position> 执行器（位置伺服），不是 <motor>
    • 控制输入语义：目标关节角度（rad），在 ctrlrange 范围内
    • 执行器内部：自动使用PID将目标角度转换为力矩
    • 错误做法：把 grasp_val 当做力度乘数（这是 <motor> 的方式）
    
    【正确做法】
    • 定义每个关节类型的最大摆动角度
    • grasp_val [0,1] 线性映射到 [0, max_angle]
    • 位置伺服自动处理动力学

    target_pose: 9D数组 [x, y, z, qw, qx, qy, qz, grasp, spread]
    """
    #### grasping encoding: (x y z qw qx qy qz alpha beta) 
    ctrl_cmd = np.zeros(model.nu)
    grasp_val = np.clip(target_pose[7], 0.0, 1.0) # tanh sigmoid 归一化函数
    spread_val = np.clip(target_pose[8], -0.2, 0.3)
    
    # 【参数】每类关节的最大控制角度
    # 这些值应该在各自的 ctrlrange 范围内
    flex_max_angle = 1.571        # J3 弯曲最大角度 (rad)，ctrlrange=[-0.262, 1.571]
    abd_max_angle = 0.349         # J4 侧摆最大角度 (rad)，ctrlrange=[-0.349, 0.349]
    thumb_max_angles = [1.0472, 1.22173, 0.20944, 0.698132, 1.5708] # 各拇指关节的最大角度
    tendon_max_angle = 3.14

    tendon_actuator_map = get_tendon_actuator_map(model)

    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_adr = model.jnt_qposadr[jnt_id]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        
        # 获取该执行器的物理限制
        ctrl_range = model.actuator_ctrlrange[i]
        
        if jnt_adr in planner.flex_adrs: # 【四指弯曲】grasp_val [0,1] → 目标角度 [0, flex_max_angle]
            val = grasp_val * flex_max_angle
                
        elif jnt_adr in planner.thumb_adrs: # 【大拇指】根据拇指内索引选择对应的最大角度
            try:
                thumb_idx = planner.thumb_adrs.index(jnt_adr)
                max_angle = thumb_max_angles[thumb_idx] if thumb_idx < len(thumb_max_angles) else 0.8
            except (ValueError, IndexError):
                max_angle = 0.8
            val = grasp_val * max_angle
            
        elif jnt_adr in planner.abd_adrs: # 【四指侧摆】spread_val 已经是角度偏移
            val = spread_val
        else:
            val = 0.0

        ctrl_cmd[i] = np.clip(val, ctrl_range[0], ctrl_range[1])

    for td_name, actuator_idx in tendon_actuator_map.items():
        ctrl_cmd[actuator_idx] = grasp_val * tendon_max_angle               

    return ctrl_cmd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="shadow_hand/scene_left.xml", help='mujoco models path')
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    model, data = mujoco_load(args.model_path)

    # 初始状态 - 使用 StateStruct
    initial_state = StateStruct(
        position=[0.4, 0.4, 0.3],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        grasp=0.0,
        spread=0.0
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
                target_state.grasp = 0.8
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
                else:
                    current_grasp_val = 0.0

                # 构造当前执行状态用于计算控制命令
                exec_state = StateStruct(
                    position=target_state.get_position(),
                    quaternion=target_state.get_quaternion(),
                    grasp=current_grasp_val,
                    spread=target_state.spread
                )
                print(f"exec_state grasp: {exec_state.grasp:.3f}")
                # 设置关节执行器的控制信号
                joint_ctrl = qpos_to_ctrl(model, data, planner, exec_state.to_array())
                data.ctrl = joint_ctrl
                
                mujoco.mj_step(model, data)

            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
