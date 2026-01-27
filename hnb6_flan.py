import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer
import argparse

from src.mujoco_utils import mujoco_load
from src.grasp_state import StateStruct
from src.grasp_planner import GraspPlanner, get_synergy_mapping, get_tendon_actuator_map

"""
大拇指： lh_A_THJx
    J1:近端弯曲
    J2:远端弯曲
    J3:远端关节侧摆
    J4:虎口大小（值越大，虎口越开）
    J5:掌根关节弯曲（伴有一定旋转）
"""

#给定拇指关节角度范围，实现"C"型手势抓取瓶子
def apply_fixed_joint_angles(model, data, _fixed_cache={}):
    """
    仅在第一次执行时生成随机角度，后续调用将直接应用缓存的值。
    """
    # 1. 检查缓存是否为空（即是否为第一次运行）
    if not _fixed_cache:
        rng = np.random.default_rng(seed=42)
        _fixed_cache['lh_THJ5'] = np.clip(rng.normal(0.5, 0.015), -1.05, 1.05)
        _fixed_cache['lh_THJ4'] = np.clip(rng.normal(1.0, 0.015), 0, 1.22)
        
    # 2. 遍历模型关节并应用
    for jnt_id in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if jnt_name in _fixed_cache:
            joint_addr = model.jnt_qposadr[jnt_id]
            # 强制覆盖 qpos
            data.qpos[joint_addr] = _fixed_cache[jnt_name]



def qpos_to_ctrl(model, planner, target_pose):
    """
 
    """
    # ===== 参数配置 =====
    FLEX_MAX_ANGLE = 1.571          # 四指J3最大弯曲角度 (rad)
    THUMB_MAX_ANGLES = [1.0472, 1.22173, 0.20944, 0.698132, 1.5708]  # 各拇指关节的最大角度
    TENDON_MAX_TENSION = 3.14       # 腱最大张力
    
    # ===== 提取并归一化协同变量 =====
    # target_pose[7] 和 [8] 已经由 StateStruct 限制在有效范围内
    grasp_val = target_pose[7]      # [0, 1]，表示抓取强度
    spread_val = target_pose[8]     # [-0.2, 0.3]，表示手指展开程度
    
    # ===== 初始化控制信号 =====
    ctrl_cmd = np.zeros(model.nu)
    
    # ===== 映射每个执行器 =====
    for i in range(model.nu):
        # 获取执行器对应的关节地址和控制范围
        jnt_id = model.actuator_trnid[i, 0]
        jnt_adr = model.jnt_qposadr[jnt_id]
        ctrl_range = model.actuator_ctrlrange[i]
        
        # 根据关节类型计算目标角度
        if jnt_adr in planner.flex_adrs:
            # 【四指弯曲 J3】：grasp_val [0,1] → 角度 [0, FLEX_MAX_ANGLE]
            val = grasp_val * FLEX_MAX_ANGLE
            
        elif jnt_adr in planner.thumb_adrs:
            # 【大拇指】：grasp_val [0,1] → 各关节按对应系数映射
            thumb_idx = planner.thumb_adrs.index(jnt_adr)
            max_angle = THUMB_MAX_ANGLES[thumb_idx] if thumb_idx < len(THUMB_MAX_ANGLES) else 0.8
            val = grasp_val * max_angle
            
        elif jnt_adr in planner.abd_adrs:
            # 【四指侧摆 J4】：spread_val 已在 [-0.2, 0.3] 范围，直接使用
            val = spread_val
        else:
            # 其他执行器（未分类）设为0
            val = 0.0
        
        # 严格按执行器的控制范围裁剪
        ctrl_cmd[i] = np.clip(val, ctrl_range[0], ctrl_range[1])
    
    # ===== 设置腱执行器的张力 =====
    tendon_actuator_map = get_tendon_actuator_map(model)
    for tendon_name, actuator_idx in tendon_actuator_map.items():
        # 腱张力由 grasp_val 控制：grasp增加时腱张力增加，推动J1、J2联动弯曲
        ctrl_cmd[actuator_idx] = grasp_val * TENDON_MAX_TENSION
    
    return ctrl_cmd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="shadow_hand/scene_left.xml", help='mujoco models path')
    args = parser.parse_args()
    return args
    

rng = np.random.default_rng(seed=42)

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

                # =====特定关节的固定角度（可选覆盖） =====
                # 如果需要固定某些拇指关节的角度，在这里指定
                
                apply_fixed_joint_angles(model, data)

                # 设置关节执行器的控制信号
                joint_ctrl = qpos_to_ctrl(model, planner, exec_state.to_array())
                data.ctrl = joint_ctrl
                
                mujoco.mj_step(model, data)

            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
