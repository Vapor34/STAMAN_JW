import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse

from core.simulator import mujoco_load
from core.hand_state import StateStruct
from core.hand_control import HandControl
from planning.position_planner import PositionPlanner


DEFAULT_MODEL_PATH = "configs/scene_left.xml"
OBJECT_BODY_NAME = "bottle_body"
HAND_BODY_PREFIX = "lh_"
HAND_BASE_FREEJOINT_NAME = "palm_freejoint"

PLANNER_STEPS = 1000
PLANNING_TIME_EPS = 0.002
INITIAL_Z_OFFSET = np.array([0.0, 0.0, 0.15])
INITIAL_RANDOM_POS_PERTURB = 0.05

FORCE_THRESHOLD_N = 1.0
RELEASE_STEP = 0.03
GRASP_START_TIME = 1.0
GRASP_DURATION = 2.0

DEBUG_PRINT_TIME = 4.0
DEBUG_PRINT_WINDOW = 0.01

FINGER_PREFIX_TO_SYNERGIES = {
    "lh_ff": {"grasp", "curl"},
    "lh_mf": {"grasp", "curl"},
    "lh_rf": {"grasp", "curl"},
    "lh_lf": {"grasp", "curl"},
    "lh_th": {"thumb_base", "thumb_flex"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="mujoco models path")
    return parser.parse_args()


def body_name_to_synergies(body_name):
    for prefix, synergies in FINGER_PREFIX_TO_SYNERGIES.items():
        if body_name.startswith(prefix):
            return synergies
    return set()


def get_hand_base_joint_indices(model):
    hand_base_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HAND_BASE_FREEJOINT_NAME)
    if hand_base_jnt_id == -1:
        hand_base_jnt_id = 0
    return model.jnt_qposadr[hand_base_jnt_id], model.jnt_dofadr[hand_base_jnt_id]


def main():
    args = parse_args()
    model, data = mujoco_load(args.model_path)
    controller = HandControl(model, data)

    mujoco.mj_forward(model, data)

    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    obj_pos = data.xpos[obj_body_id].copy()
    initial_position = obj_pos + INITIAL_Z_OFFSET

    initial_state = StateStruct(
        model,
        data,
        position=initial_position,
        quaternion=[1.0, 0.0, 0.0, 0.0],
    )
    initial_guess = initial_state

    planner = PositionPlanner(initial_guess, model, data, body_name=OBJECT_BODY_NAME)
    planner.steps = PLANNER_STEPS

    hand_base_qpos_adr, hand_base_dof_adr = get_hand_base_joint_indices(model)

    planning_done = False
    target_state = initial_state.copy()
    locked_synergies = {}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if data.time < PLANNING_TIME_EPS:
                print("\n[状态] 正在规划最佳抓取点...")

                planner.state = initial_guess.to_array()
                planner.state[0:3] += np.random.uniform(-INITIAL_RANDOM_POS_PERTURB, INITIAL_RANDOM_POS_PERTURB, 3)
                best_pose, _ = planner.anneal()

                target_state.from_array(best_pose)
                print(f"[完成] 目标 Grasp: {target_state.grasp:.2f}, Spread: {target_state.spread:.2f}")
                print(f"[完成] 目标位置: ({target_state.x:.3f}, {target_state.y:.3f}, {target_state.z:.3f})")

                execution_start_time = time.time()
                planning_done = True

                data.time = PLANNING_TIME_EPS

            if planning_done:
                anim_time = time.time() - execution_start_time

                base_pos = target_state.get_position()
                base_quat = target_state.get_quaternion()
                data.qpos[hand_base_qpos_adr:hand_base_qpos_adr + 3] = base_pos
                data.qpos[hand_base_qpos_adr + 3:hand_base_qpos_adr + 7] = base_quat
                data.qvel[hand_base_dof_adr:hand_base_dof_adr + 6] = 0.0
                mujoco.mj_forward(model, data)

                if anim_time > GRASP_START_TIME:
                    grasp_progress = (anim_time - GRASP_START_TIME) / GRASP_DURATION
                    grasp_progress = np.clip(grasp_progress, 0.0, 1.0)

                    desired_values = {
                        "grasp": np.clip(target_state.grasp + grasp_progress, 0.0, 1.0),
                        "curl": np.clip(target_state.curl + grasp_progress, 0.0, 1.0),
                        "thumb_flex": np.clip(target_state.thumb_flex + grasp_progress, 0.0, 1.0),
                    }

                    force_buf = np.zeros(6)
                    for i in range(data.ncon):
                        con = data.contact[i]
                        body1id = model.geom_bodyid[con.geom1]
                        body2id = model.geom_bodyid[con.geom2]

                        if planner.obj_body_id not in (body1id, body2id):
                            continue

                        mujoco.mj_contactForce(model, data, i, force_buf)
                        contact_force = np.linalg.norm(force_buf[:3])

                        if contact_force > FORCE_THRESHOLD_N:
                            hand_body_id = body1id if body1id != planner.obj_body_id else body2id

                            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, hand_body_id)
                            if body_name is None:
                                continue
                            body_name = body_name.lower()
                            if not body_name.startswith(HAND_BODY_PREFIX):
                                continue

                            syn_groups = body_name_to_synergies(body_name)
                            for syn in syn_groups:
                                if syn in desired_values and syn not in locked_synergies:
                                    locked_value = np.clip(desired_values[syn] - RELEASE_STEP, 0.0, 1.0)
                                    locked_synergies[syn] = locked_value
                                    print(f"[力控] 锁定 {syn} (body={body_name}, force={contact_force:.2f}N, hold={locked_value:.2f})")

                    current_grasp = locked_synergies.get("grasp", desired_values["grasp"])
                    current_curl = locked_synergies.get("curl", desired_values["curl"])
                    current_thumb_base = target_state.thumb_base
                    current_thumb_flex = locked_synergies.get("thumb_flex", desired_values["thumb_flex"])
                else:
                    current_grasp = target_state.grasp
                    current_curl = target_state.curl
                    current_thumb_base = target_state.thumb_base
                    current_thumb_flex = target_state.thumb_flex

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

                controller.set_hand_state(exec_state)

                # if DEBUG_PRINT_TIME - DEBUG_PRINT_WINDOW < anim_time < DEBUG_PRINT_TIME:
                #     controller.print_all_act_val()

                mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
