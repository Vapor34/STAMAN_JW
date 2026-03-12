"""Post-planning grasp execution with admittance control."""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from core.hand_state import StateStruct


@dataclass
class GraspConfig:
    """Configuration for admittance-controlled grasp execution.

    Admittance law (first-order, per synergy group):

        dx/dt = (1/B) * (F_contact - F_desired)

    where *x* is the synergy value, *B* is the damping coefficient,
    *F_contact* is the max contact force on the finger group, and
    *F_desired* is the target grasping force.

    Before any contact the fingers simply close at ``close_rate``
    (open-loop ramp).  Once contact is detected the admittance loop
    takes over.
    """

    hand_body_prefix: str = "lh_"
    hand_base_freejoint_name: str = "palm_freejoint"

    # --- admittance parameters ---
    desired_force_n: float = 20.0        # target contact force per finger group
    damping_B: float = 5.0             # admittance damping  (higher → slower reaction)
    close_rate: float = 0.5             # synergy/s  open-loop closing speed before contact

    grasp_start_time: float = 1.0       # seconds to hold pre-grasp before closing
    grasp_duration: float = 3.0         # time window for the admittance-controlled phase

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
    """Execute grasp motion with per-synergy admittance control."""

    def __init__(self, model, data, controller, planner, pre_grasp_state, config=None):
        self.model = model
        self.data = data
        self.controller = controller
        self.planner = planner
        self.pre_grasp_state = pre_grasp_state
        self.config = config or GraspConfig()

        self.hand_base_qpos_adr, self.hand_base_dof_adr = (
            self._get_hand_base_joint_indices()
        )

        # Current synergy values – start from pre-grasp targets.
        self.synergy_values = {
            "grasp": self.pre_grasp_state.grasp,
            "curl": self.pre_grasp_state.curl,
            "thumb_base": self.pre_grasp_state.thumb_base,
            "thumb_flex": self.pre_grasp_state.thumb_flex,
        }
        # Track whether each group has ever made contact.
        self.in_contact = {s: False for s in self.synergy_values}
        self._prev_time = None

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
        base_pos = self.pre_grasp_state.get_position()
        base_quat = self.pre_grasp_state.get_quaternion()

        self.data.qpos[self.hand_base_qpos_adr : self.hand_base_qpos_adr + 3] = base_pos
        self.data.qpos[self.hand_base_qpos_adr + 3 : self.hand_base_qpos_adr + 7] = base_quat
        self.data.qvel[self.hand_base_dof_adr : self.hand_base_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---- force sensing ----

    def _get_contact_forces_per_synergy(self):
        """Return the max contact force on each synergy group (N).

        Iterates over all active contacts between the hand and the target
        object.  For each contact the hand body is mapped to its synergy
        group(s) and the maximum normal force across all contacts in that
        group is recorded.
        """
        max_forces = {s: 0.0 for s in self.synergy_values}
        force_buf = np.zeros(6)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1id = self.model.geom_bodyid[con.geom1]
            body2id = self.model.geom_bodyid[con.geom2]

            if self.planner.obj_body_id not in (body1id, body2id):
                continue

            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            contact_force = np.linalg.norm(force_buf[:3])

            hand_body_id = body1id if body1id != self.planner.obj_body_id else body2id
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, hand_body_id
            )
            if body_name is None:
                continue
            body_name = body_name.lower()
            if not body_name.startswith(self.config.hand_body_prefix):
                continue

            syn_groups = self._body_name_to_synergies(body_name)
            for syn in syn_groups:
                if syn in max_forces:
                    max_forces[syn] = max(max_forces[syn], contact_force)

        return max_forces

    # ---- admittance control ----

    def _compute_current_synergies(self, anim_time):
        """Update synergy values using admittance control.

        Phase 1 (anim_time <= grasp_start_time):
            Hold pre-grasp pose.

        Phase 2 (grasp phase):
            For each synergy group:
            - If no contact yet: ramp closed at ``close_rate`` (open-loop).
            - Once contact detected: switch to admittance law
              dx/dt = (1/B) * (F_contact - F_desired)
              so that the finger retreats when force is too high and
              advances when force is too low, converging to F_desired.
        """
        if anim_time <= self.config.grasp_start_time:
            self._prev_time = anim_time
            return (
                self.pre_grasp_state.grasp,
                self.pre_grasp_state.curl,
                self.pre_grasp_state.thumb_base,
                self.pre_grasp_state.thumb_flex,
            )

        # dt since last call
        dt = anim_time - (self._prev_time if self._prev_time is not None else anim_time)
        dt = np.clip(dt, 0.0, 0.05)  # cap to avoid jumps
        self._prev_time = anim_time

        contact_forces = self._get_contact_forces_per_synergy()

        cfg = self.config
        for syn in ("grasp", "curl", "thumb_flex"):
            f_contact = contact_forces[syn]

            if f_contact > 0.1:  # non-trivial contact
                self.in_contact[syn] = True

            if not self.in_contact[syn]:
                # Open-loop: ramp toward closed
                self.synergy_values[syn] += cfg.close_rate * dt
            else:
                # Admittance law:  dx/dt = (1/B)(F_contact - F_desired)
                # Positive error (too much force) → dx > 0 → synergy increases
                # BUT higher synergy = more closed = more force, so we
                # need the sign inverted:  dx/dt = (1/B)(F_desired - F_contact)
                force_error = cfg.desired_force_n - f_contact
                velocity = force_error / cfg.damping_B
                self.synergy_values[syn] += velocity * dt

            self.synergy_values[syn] = np.clip(self.synergy_values[syn], 0.0, 1.0)

        # thumb_base is not force-controlled (orientation only)
        return (
            self.synergy_values["grasp"],
            self.synergy_values["curl"],
            self.synergy_values["thumb_base"],
            self.synergy_values["thumb_flex"],
        )

    def step(self, anim_time):
        """Run one simulation step of grasp execution."""
        self._lock_hand_base()

        current_grasp, current_curl, current_thumb_base, current_thumb_flex = self._compute_current_synergies(
            anim_time
        )

        exec_state = StateStruct(
            self.model,
            self.data,
            position=self.pre_grasp_state.get_position(),
            quaternion=self.pre_grasp_state.get_quaternion(),
            grasp=current_grasp,
            curl=current_curl,
            spread=self.pre_grasp_state.spread,
            thumb_base=current_thumb_base,
            thumb_flex=current_thumb_flex,
        )

        self.controller.set_hand_state(exec_state)
        mujoco.mj_step(self.model, self.data)
