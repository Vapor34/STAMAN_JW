import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer
import argparse

from core.simulator import mujoco_load
from core.hand_state import StateStruct
from core.hand_control import HandControl
from planning.position_planner import PositionPlanner
from planning.grasp import ProgressiveGrasp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="configs/scene_left.xml", help='mujoco models path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model, data = mujoco_load(args.model_path)
    controler = HandControl(model, data)

    palm_center_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lh_palm_center")
    initial_position = data.site_xpos[palm_center_id]
    mujoco.mj_forward(model, data)  # Ensure data is updated with initial positions

    # 初始状态 - 使用 StateStruct
    initial_state = StateStruct(
        model,
        data,
        position=initial_position,
        quaternion=[1.0, 0.0, 0.0, 0.0]
    )
    initial_guess = initial_state

    planner = PositionPlanner(initial_guess, model, data, body_name='bottle_body')
    planner.steps = 2000

    # 动画控制变量
    planning_done = False
    target_state = initial_state.copy()  # 规划得到的目标状态

    executor = ProgressiveGrasp(model, data, controler)
    threshold = 1.0  # 接触力阈值，单位根据模型设置（通常是牛顿）

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 规划阶段
            if data.time < 0.002:
                print("\n[状态] 正在规划最佳抓取点...")
                
                # 从初始状态开始扰动
                planner.state = initial_guess.to_array() #state=np.array[x,y,z,qw,qx,qy,qz,syn1,syn2...,syn6]
                planner.state[0:3] += np.random.uniform(-0.05, 0.05, 3)
                best_pose, energy = planner.anneal()

                
                # 将规划结果加载到 StateStruct
                target_state.from_array(best_pose)
                # target_state.grasp = 0.8
                print(f"[完成] 目标 Grasp: {target_state.grasp:.2f}, Spread: {target_state.spread:.2f}")
                print(f"[完成] 目标位置: ({target_state.x:.3f}, {target_state.y:.3f}, {target_state.z:.3f})")
                
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
                    # after 1.0s 
                    # 检测高力接触的 synergy
                    high_force_synergies = set()
                    for i in range(data.ncon):
                        con = data.contact[i]
                        body1id = model.geom_bodyid[con.geom1]
                        body2id = model.geom_bodyid[con.geom2]
                        body1name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1id).lower()
                        body2name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2id).lower() 
                        force = np.linalg.norm(np.array(con.H))
                        bodies_id = []
                        if force > threshold:
                            if body1id in executor.contact_body_ids and planner.obj_body_id in [body1id, body2id]:
                                bodies_id.append(body1id)
                                print(f"Contact {i}: body1={body1name}, body2={body2name}, force={force:.2f}")
                            if body2id in executor.contact_body_ids and planner.obj_body_id in [body1id, body2id]:
                                bodies_id.append(body2id)
                                print(f"Contact {i}: body1={body1name}, body2={body2name}, force={force:.2f}")
                            for body_id in bodies_id:                                 
                                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id).lower()
                                for syn, act_names in controler.get_synergy_map().items():
                                    if body_name in act_names:
                                        high_force_synergies.add(syn)
                    grasp_progress = (anim_time - grasp_start_time) / grasp_duration
                    grasp_progress = np.clip(grasp_progress, 0.0, 1.0)
                    
                    # 根据是否固定决定当前值
                    current_grasp = target_state.grasp if 'grasp' in high_force_synergies else np.clip(target_state.grasp + grasp_progress * 1, 0, 1.5)
                    current_curl = target_state.curl if 'curl' in high_force_synergies else np.clip(target_state.curl + grasp_progress * 1, 0, 1.5)
                    current_thumb_base = target_state.thumb_base if 'thumb_base' in high_force_synergies else target_state.thumb_base
                    current_thumb_flex = target_state.thumb_flex if 'thumb_flex' in high_force_synergies else np.clip(target_state.thumb_flex + grasp_progress * 1, 0, 1.5)


                else:
                    # 0-1s
                    current_grasp = target_state.grasp
                    current_curl = target_state.curl
                    current_thumb_base = target_state.thumb_base
                    current_thumb_flex = target_state.thumb_flex
                    

                # 构造当前执行状态用于计算控制命令
                exec_state = StateStruct(
                    model,
                    data,
                    position=target_state.get_position(),
                    quaternion=target_state.get_quaternion(),
                    grasp=current_grasp,
                    curl=current_curl,
                    spread=target_state.spread,
                    thumb_base=current_thumb_base,
                    thumb_flex=current_thumb_flex,
                )

                # 设置关节执行器的控制信号
                controler.set_hand_state(exec_state)
                # controler.print_all_act_val()

                # =====特定关节的固定角度（可选覆盖） =====
                # 如果需要固定某些拇指关节的角度，在这里指定
                # controler.set_act_val('lh_THJ4', 1.0)  # 拇指近端关节
                # ====================================

                """
                TODO: make tight grasp after setting the best pose (make finger curl slowly until hold the object)
                """
                
                if anim_time < 4.0 and anim_time >3.99:
                    controler.print_all_act_val()
                
                mujoco.mj_step(model, data)
                # # 开始分帧执行抓取，不要一次性调用 execute，否则肉眼看不出动作
                # if anim_time > 1.0 and anim_time < 3.0:
                #     if not executor._running:
                #         executor.start(target_state, target_force=16.0, closure_time=2.0, max_steps=1000)
                #         print("[状态] 已启动渐进式抓取")

                #     result = executor.step()
                #     if result is not None:
                #         print(f"[执行完成] 成功={result['success']} 力量={result['total_force']:.2f} 步数={result['steps']}")
                #         planning_done = False  # 只执行一次

                # if anim_time < 4.0 and anim_time >3.95:
                #     controler.print_all_act_val()
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
