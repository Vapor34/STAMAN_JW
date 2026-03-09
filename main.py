import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse

from core.simulator import mujoco_load
from core.hand_state import StateStruct
from core.hand_control import HandControl
from planning.grasp import GraspExecutor
from planning.position_planner import PositionPlanner


DEFAULT_MODEL_PATH = "configs/scene_left.xml"
OBJECT_BODY_NAME = "bottle_body"

PLANNER_STEPS = 1000
PLANNING_TIME_EPS = 0.002
INITIAL_Z_OFFSET = np.array([0.0, 0.0, 0.15])
INITIAL_RANDOM_POS_PERTURB = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="mujoco models path")
    return parser.parse_args()


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

    planning_done = False
    target_state = initial_state.copy()
    grasp_executor = None
    execution_start_time = None

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
                grasp_executor = GraspExecutor(model, data, controller, planner, target_state)

                data.time = PLANNING_TIME_EPS

            if planning_done:
                anim_time = time.time() - execution_start_time
                grasp_executor.step(anim_time)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
