"""
Grasp Planner: Optimize Shadow Hand grasp poses using simulated annealing

Uses the Annealer base class to search for optimal grasp configurations
by minimizing an energy function that balances contact proximity,
orientation, and collision constraints.
"""

from simanneal import Annealer
import mujoco
import numpy as np
from core.hand_state import StateStruct
from core.hand_control import HandControl
import trimesh
from trimesh.proximity import closest_point
import trimesh.proximity as proximity
import os
import fast_simplification


class PositionPlanner(Annealer):
    """
    Simulated annealing planner for Shadow Hand grasp optimization
    
    Finds pre-pose(position, orientation, synergy values) to prepare for grasping
    while minimizing contact distance, ensuring proper orientation,
    and avoiding collisions.
    """
    
    def __init__(self, state, model, data, body_name, hand_body_prefix='lh_'):
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
        raw_mesh = trimesh.load('./configs/assets/bottle.obj')
        # Handle Scene objects by merging meshes
        if isinstance(raw_mesh, trimesh.Scene):
            self.obj_mesh = raw_mesh.dump(concatenate=True)
        else:
            self.obj_mesh = raw_mesh
        
        vertices = self.obj_mesh.vertices
        faces = self.obj_mesh.faces
        new_vertices, new_faces = fast_simplification.simplify(
            vertices, faces, target_count=500
        )
        self.obj_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

        # Get floor geom ID
        self.floor_geom_id = model.geom("floor").id

        # Get palm body ID
        self.palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.hand_prefix}palm")
        self.palm_center_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lh_palm_center")
        if self.palm_center_site_id == -1:
            print("Warning: Site 'lh_palm_center' not found!")

        # Collect hand geometry and contact points
        self.hand_geom_ids = []
        self.contact_body_ids = []
        excluded_keywords = ['wrist', 'forearm', 'palm']
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                self.hand_geom_ids.extend(self._get_geoms_id_of_body(i))
                name_lower = name.lower()
                is_excluded = any(k in name_lower for k in excluded_keywords)
                if not is_excluded:
                    self.contact_body_ids.append(i)

        # Build synergy variable mappings
        syn_map = self.act_ctrl.get_synergy_map()
        self.syn_grasp = syn_map["grasp"]
        self.syn_curl = syn_map["curl"]
        self.syn_spread = syn_map["spread"]
        self.syn_thumb_base = syn_map["thumb_base"]
        self.syn_thumb_flex = syn_map["thumb_flex"]
        self.syn_wrist = syn_map["wrist"]

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
        self.data.qpos[0:3] = [palm_pos[0], palm_pos[1], palm_pos[2]-state_struct.offset[2]]
        self.data.qpos[3:7] = state_struct.get_quaternion()
        
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
        if hasattr(self, 'T') and hasattr(self, 'Tmax') and self.Tmax > 0:
            temp_scale = max(0.1, self.T / self.Tmax)
        else:
            temp_scale = 1.0
        
        # Position perturbation with bounds
        pos_step = 0.00015 * temp_scale
        current.position +=  + np.random.normal(0, pos_step, 3)
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
        1. Proximity: Average distance from fingers to object surface
        2. Palm alignment: Hand orientation alignment with object
        3. Collision avoidance: Penalty for penetration
        
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
        palm_center_pos = self.data.site_xpos[self.palm_center_site_id]
        
        # Stack all query points
        all_query_points_world = np.vstack([points_world, palm_center_pos])
        
        # Transform to object local coordinate system
        all_points_local = (all_query_points_world - b_pos) @ b_mat
        
        # Batch signed distance queries on mesh (positive outside, negative inside)
        dists = proximity.signed_distance(self.obj_mesh, all_points_local)
        
        # For closest points, still need them for palm alignment
        closest_points_local, _, _ = closest_point(self.obj_mesh, all_points_local)
        
        # Extract finger and palm distances
        finger_dists = dists[:-1]
        palm_closest_point_local = closest_points_local[-1]
        # #---------
        # 各距离的平方和（对穿透更敏感）
        proximity_energy = np.sum(finger_dists**2)

        # # 或者取最大值（关注最远的那根）
        # proximity_energy = np.max(np.abs(finger_dists))

        # # 也可以组合：sum of squares + max 等
        # #--------------
        
        # Palm distance term
        palm_dist = np.linalg.norm(palm_center_pos - b_pos)
        
        # Orientation alignment: compute angle between palm normal and surface normal
        target_point_world = palm_closest_point_local @ b_mat.T + b_pos
        vec_to_surface = target_point_world - palm_center_pos
        
        dist_to_surface = np.linalg.norm(vec_to_surface)
        vec_to_surface_norm = vec_to_surface / (dist_to_surface + 1e-6)
        
        palm_rot = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_normal = -palm_rot[:, 1]
        
        orientation_energy = 1.0 - np.dot(palm_normal, vec_to_surface_norm)

        # Collision penalty
        collision_penalty = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.dist < 0: 
                collision_penalty += 10.0 + abs(con.dist) * 5000

        # Weighted sum of all energy terms
        total_energy = (
            proximity_energy * 100.0 + 
            palm_dist * 100.0 +
            orientation_energy * 10.0 +
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
    
    def get_min_dist(self, body_pos):
        """
        Calculate minimum distance from a point to obj surface
        
        Args:
            body_pos: Query point position
            
        Returns:
            float: Minimum distance to mesh surface
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "..", "assets", "bottle.obj")
        mesh = trimesh.load(file_path)

        closest_point_val, distance, triangle_id = mesh.proximity.closest_point([body_pos])
        return distance[0]
