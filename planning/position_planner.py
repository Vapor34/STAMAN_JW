"""
Grasp Planner: Optimize Shadow Hand grasp poses using simulated annealing

Uses the Annealer base class to search for optimal grasp configurations
by minimizing an energy function that balances contact proximity,
orientation, and collision constraints.
"""

import mujoco
import numpy as np
import trimesh
from simanneal import Annealer
from trimesh.proximity import signed_distance

from core.hand_control import HandControl
from core.hand_state import StateStruct


class PositionPlanner(Annealer):
    """
    Simulated annealing planner for Shadow Hand grasp optimization

    Finds pre-pose(position, orientation, synergy values) to prepare for grasping
    while minimizing contact distance, ensuring proper orientation,
    and avoiding collisions.
    """

    def __init__(self, state, model, data, body_name, hand_body_prefix="lh_"):
        """
        Initialize grasp planner

        Args:
            state: Initial state as 13D array or StateStruct
            model: MuJoCo model
            data: MuJoCo data
            body_name: Name of the object body to grasp
            hand_body_prefix: Prefix for hand body names
        """
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix
        self.act_ctrl = HandControl(model, data)

        # Get body and geom IDs for obj
        self.obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.obj_geom_ids = self._get_geoms_id_of_body(self.obj_body_id)

        # Load and simplify obj mesh
        raw_mesh = trimesh.load("./configs/assets/bottle.obj")
        # Handle Scene objects by merging meshes
        if isinstance(raw_mesh, trimesh.Scene):
            self.obj_mesh = raw_mesh.dump(concatenate=True)
        else:
            self.obj_mesh = raw_mesh

        # Apply the same scale as in MuJoCo scene XML (scale="0.15 0.15 0.15")
        self.obj_mesh.apply_scale(0.15)

        # NOTE: Do NOT simplify the mesh with fast_simplification!
        # Simplification breaks trimesh.signed_distance (flips normals/winding),
        # making penetration detection completely unreliable.
        # The original mesh (2378 faces) is fast enough (~5ms per signed_distance call).

        # Get floor geom ID
        self.floor_geom_id = model.geom("floor").id

        # Get palm body ID (for orientation calculation)
        self.palm_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{self.hand_prefix}palm",
        )

        # Collect hand geometry IDs (for collision detection - include ALL hand bodies)
        self.hand_geom_ids = []
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                self.hand_geom_ids.extend(self._get_geoms_id_of_body(i))

        # Contact bodies for proximity optimization: only distal + middle bodies
        # Exclude knuckle, proximal, metacarpal, thbase, thhub - these are structural
        # bodies near the palm that would bias the optimizer to move the palm into the object
        # rather than curling fingers around it.
        # Weight: distal (fingertip) bodies get higher weight than middle bodies.
        self.contact_body_ids = []
        self.contact_body_weights = []
        contact_keywords_high = ["distal"]  # fingertips: weight 3.0
        contact_keywords_low = ["middle"]  # mid-phalanx: weight 1.0
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                name_lower = name.lower()
                if any(k in name_lower for k in contact_keywords_high):
                    self.contact_body_ids.append(i)
                    self.contact_body_weights.append(3.0)
                elif any(k in name_lower for k in contact_keywords_low):
                    self.contact_body_ids.append(i)
                    self.contact_body_weights.append(1.0)
        self.contact_body_weights = np.array(self.contact_body_weights)

        super(PositionPlanner, self).__init__(state)

    def set_hand_pose(self, state_struct):
        """
        Apply hand pose configuration to MuJoCo simulation

        Maps 13D synergy state to hand configuration by setting:
        - Base position and orientation (7D)
        - Joint angles via synergy variables (6D)

        Then performs forward kinematics to update all body positions.

        Args:
            state_struct: StateStruct with desired pose
        """
        # Set base position and quaternion
        palm_pos = state_struct.get_position()
        quat = state_struct.get_quaternion()
        # Rotate offset by current orientation to get correct world-frame offset
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        world_offset = rot_mat @ state_struct.offset
        self.data.qpos[0:3] = palm_pos - world_offset
        self.data.qpos[3:7] = quat

        # Apply synergy values to actuators
        self.act_ctrl.set_hand_state(state_struct)

        # Perform forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def move(self):
        """
        Simulated annealing perturbation step with temperature-adaptive step sizes

        Applies small random changes to hand pose while respecting constraints.
        Step sizes scale with temperature for better exploration at high T and exploitation at low T.
        """
        # Convert to StateStruct for convenient manipulation
        current = StateStruct(self.model, self.data)
        current.from_array(self.state)

        # Temperature-adaptive scaling (higher T = larger steps)
        if hasattr(self, "T") and hasattr(self, "Tmax") and self.Tmax > 0:
            temp_scale = max(0.1, self.T / self.Tmax)
        else:
            temp_scale = 1.0

        # Position perturbation with bounds
        pos_step = 0.01 * temp_scale
        current.position += np.random.normal(0, pos_step, 3)
        obj_pos = self.data.xpos[self.obj_body_id]
        current.position = np.clip(current.position, obj_pos - 1, obj_pos + 1)

        # Quaternion perturbation with normalization
        quat_step = 0.05 * temp_scale
        current.quaternion += np.random.normal(0, quat_step, 4)
        current._normalize_quaternion()

        # Synergy variable perturbations with value clipping
        grasp_step = 0.08 * temp_scale
        current.grasp += np.random.normal(0, grasp_step)
        current.grasp = np.clip(current.grasp, 0.0, 1.0)

        curl_step = 0.05 * temp_scale
        current.curl += np.random.normal(0, curl_step)
        current.curl = np.clip(current.curl, 0.0, 1.0)

        spread_step = 0.05 * temp_scale
        current.spread += np.random.normal(0, spread_step)
        current.spread = np.clip(current.spread, -0.2, 0.3)

        thumb_base_step = 0.5 * temp_scale
        current.thumb_base += np.random.normal(0, thumb_base_step)
        current.thumb_base = np.clip(current.thumb_base, 0.0, 1.0)

        thumb_flex_step = 0.5 * temp_scale
        current.thumb_flex += np.random.normal(0, thumb_flex_step)
        current.thumb_flex = np.clip(current.thumb_flex, 0.0, 1.0)

        # Update state array
        self.state = current.to_array()

    def energy(self):
        """
        Energy function to minimize

        Calculates grasp quality metric combining:
        1. Proximity: Finger distal/middle bodies close to object surface
        2. Penetration: Hard penalty for bodies inside the object
        3. Finger direction: Fingertips should point toward the object
        4. Collision: Penalty for self-collision and floor contact

        Returns:
            float: Total energy (lower is better)
        """
        # Update physics state
        state_struct = StateStruct(self.model, self.data)
        state_struct.from_array(self.state)
        self.set_hand_pose(state_struct)

        # Get object geometry in world frame
        b_pos = self.data.xpos[self.obj_body_id]
        b_mat = self.data.xmat[self.obj_body_id].reshape(3, 3)
        points_world = self.data.xpos[self.contact_body_ids]

        # Transform contact body positions to object local frame
        points_local = (points_world - b_pos) @ b_mat.T

        # Signed distance: positive = INSIDE (penetrating), negative = OUTSIDE
        s_dists = signed_distance(self.obj_mesh, points_local)

        # ---- Finger proximity + penetration (weighted by body importance) ----
        target_dist = 0.005  # ideal ~5mm from surface

        proximity_energy = 0.0
        penetration_energy = 0.0

        for idx, sd in enumerate(s_dists):
            w = self.contact_body_weights[idx]
            if sd > 0:
                # INSIDE the object: very heavy penalty
                penetration_energy += (10000.0 + sd * 50000.0) * w
            else:
                # OUTSIDE: penalize distance from target
                # Linear + quadratic: strong pull far away, precision near target
                deviation = max(0.0, abs(sd) - target_dist)
                proximity_energy += w * (deviation + deviation ** 2)

        # ---- Finger direction: Z axis should point toward object ----
        palm_rot = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_pos = self.data.xpos[self.palm_body_id]
        finger_dir = palm_rot[:, 2]  # Z axis = fingertip direction
        vec_to_obj = b_pos - palm_pos
        vec_to_obj_norm = vec_to_obj / (np.linalg.norm(vec_to_obj) + 1e-6)
        finger_dir_energy = 1.0 - np.dot(finger_dir, vec_to_obj_norm)

        # ---- MuJoCo collision penalty ----
        # Penalize ALL interpenetration: hand-object, self-collision, and floor
        collision_penalty = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.dist < 0:
                g1, g2 = con.geom1, con.geom2
                is_hand1 = g1 in self.hand_geom_ids
                is_hand2 = g2 in self.hand_geom_ids
                is_obj = (g1 in self.obj_geom_ids or g2 in self.obj_geom_ids)
                is_floor = (g1 == self.floor_geom_id or g2 == self.floor_geom_id)

                if (is_hand1 or is_hand2) and is_obj:
                    # Hand-object penetration: heavy penalty
                    collision_penalty += 500.0 + abs(con.dist) * 50000
                elif is_hand1 and is_hand2:
                    # Self-collision
                    collision_penalty += 500.0 + abs(con.dist) * 50000
                elif (is_hand1 or is_hand2) and is_floor:
                    # Floor collision
                    collision_penalty += 500.0 + abs(con.dist) * 50000

        # Weighted sum
        total_energy = (
            proximity_energy * 100.0 +
            penetration_energy +
            finger_dir_energy * 50.0 +
            collision_penalty
        )

        return total_energy

    def _get_geoms_id_of_body(self, body_id):
        """
        Get all geometry IDs for a body

        Args:
            body_id: Index of the body

        Returns:
            list: Geometry IDs associated with this body
        """
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]

