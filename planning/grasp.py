"""Post-planning grasp execution utilities."""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from core.hand_state import StateStruct


@dataclass
class GraspConfig:
    """Configuration for force-threshold-based grasp execution."""

    hand_body_prefix: str = "lh_"
    hand_base_freejoint_name: str = "palm_freejoint"
    force_threshold_n: float = 1.0
    release_step: float = 0.03
    grasp_start_time: float = 1.0
    grasp_duration: float = 2.0
    finger_prefix_to_synergies: dict = field(
        default_factory=lambda: {
            "lh_ff": {"grasp", "curl"},
            "lh_mf": {"grasp", "curl"},
            "lh_rf": {"grasp", "curl"},
            "lh_lf": {"grasp", "curl"},
            "lh_th": {"thumb_base", "thumb_flex"},
        }
    )


class GraspExecutor:
    """Execute grasp motion after pre-grasp planning is complete."""

    def __init__(self, model, data, controller, planner, target_state, config=None):
        self.model = model
        self.data = data
        self.controller = controller
        self.planner = planner
        self.target_state = target_state
        self.config = config or GraspConfig()

        self.hand_base_qpos_adr, self.hand_base_dof_adr = self._get_hand_base_joint_indices()
        self.locked_synergies = {}

    def _get_hand_base_joint_indices(self):
        hand_base_jnt_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.config.hand_base_freejoint_name,
        )
        if hand_base_jnt_id == -1:
            hand_base_jnt_id = 0

        return self.model.jnt_qposadr[hand_base_jnt_id], self.model.jnt_dofadr[hand_base_jnt_id]

    def _body_name_to_synergies(self, body_name):
        for prefix, synergies in self.config.finger_prefix_to_synergies.items():
            if body_name.startswith(prefix):
                return synergies
        return set()

    def _lock_hand_base(self):
        base_pos = self.target_state.get_position()
        base_quat = self.target_state.get_quaternion()

        self.data.qpos[self.hand_base_qpos_adr : self.hand_base_qpos_adr + 3] = base_pos
        self.data.qpos[self.hand_base_qpos_adr + 3 : self.hand_base_qpos_adr + 7] = base_quat
        self.data.qvel[self.hand_base_dof_adr : self.hand_base_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _compute_current_synergies(self, anim_time):
        if anim_time <= self.config.grasp_start_time:
            return (
                self.target_state.grasp,
                self.target_state.curl,
                self.target_state.thumb_base,
                self.target_state.thumb_flex,
            )

        grasp_progress = (anim_time - self.config.grasp_start_time) / self.config.grasp_duration
        grasp_progress = np.clip(grasp_progress, 0.0, 1.0)

        desired_values = {
            "grasp": np.clip(self.target_state.grasp + grasp_progress, 0.0, 1.0),
            "curl": np.clip(self.target_state.curl + grasp_progress, 0.0, 1.0),
            "thumb_flex": np.clip(self.target_state.thumb_flex + grasp_progress, 0.0, 1.0),
        }

        force_buf = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1id = self.model.geom_bodyid[con.geom1]
            body2id = self.model.geom_bodyid[con.geom2]

            if self.planner.obj_body_id not in (body1id, body2id):
                continue

            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            contact_force = np.linalg.norm(force_buf[:3])

            if contact_force > self.config.force_threshold_n:
                hand_body_id = body1id if body1id != self.planner.obj_body_id else body2id

                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_body_id)
                if body_name is None:
                    continue

                body_name = body_name.lower()
                if not body_name.startswith(self.config.hand_body_prefix):
                    continue

                syn_groups = self._body_name_to_synergies(body_name)
                for syn in syn_groups:
                    if syn in desired_values and syn not in self.locked_synergies:
                        locked_value = np.clip(desired_values[syn] - self.config.release_step, 0.0, 1.0)
                        self.locked_synergies[syn] = locked_value
                        print(
                            f"[力控] 锁定 {syn} "
                            f"(body={body_name}, force={contact_force:.2f}N, hold={locked_value:.2f})"
                        )

        current_grasp = self.locked_synergies.get("grasp", desired_values["grasp"])
        current_curl = self.locked_synergies.get("curl", desired_values["curl"])
        current_thumb_base = self.target_state.thumb_base
        current_thumb_flex = self.locked_synergies.get("thumb_flex", desired_values["thumb_flex"])

        return current_grasp, current_curl, current_thumb_base, current_thumb_flex

    def step(self, anim_time):
        """Run one simulation step of grasp execution."""
        self._lock_hand_base()

        current_grasp, current_curl, current_thumb_base, current_thumb_flex = self._compute_current_synergies(
            anim_time
        )

        exec_state = StateStruct(
            self.model,
            self.data,
            position=self.target_state.get_position(),
            quaternion=self.target_state.get_quaternion(),
            grasp=current_grasp,
            curl=current_curl,
            spread=self.target_state.spread,
            thumb_base=current_thumb_base,
            thumb_flex=current_thumb_flex,
        )

        self.controller.set_hand_state(exec_state)
        mujoco.mj_step(self.model, self.data)
