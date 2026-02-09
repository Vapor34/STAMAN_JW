"""
抓取规划器：使用模拟退火优化Shadow Hand的抓取姿态
"""
from simanneal import Annealer
import mujoco
import numpy as np
from simanneal import Annealer
from src.grasp_state import StateStruct
from src.grasp_control import GraspControl
import trimesh
from trimesh.proximity import closest_point
import os
import fast_simplification

class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name, hand_body_prefix='lh_'):
        """
        :param state: np.array type
        """
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix
        self.act_ctrl = GraspControl(model, data)

        # get body and geom ids for bottle
        self.bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body_name)
        self.bottle_geom_ids = self._get_geoms_id_of_body(self.bottle_body_id)

        raw_mesh = trimesh.load('./shadow_hand/assets/bottle.obj')
        # 关键修正：判断是否为 Scene，如果是则合并
        if isinstance(raw_mesh, trimesh.Scene):
            # 将场景中所有的 mesh 合并成一个巨大的 Trimesh
            self.bottle_mesh = raw_mesh.dump(concatenate=True)
        else:
            self.bottle_mesh = raw_mesh
        vertices = self.bottle_mesh.vertices
        faces = self.bottle_mesh.faces
        new_vertices, new_faces = fast_simplification.simplify(
            vertices, faces, target_count=500
        )
        # 重新构建回 trimesh 对象
        self.bottle_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

        # get floor geom id
        self.floor_geom_id = model.geom("floor").id

        #get palm body id
        self.palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.hand_prefix}palm")
        self.palm_center_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lh_palm_center")
        if self.palm_center_site_id == -1:
            print("Warning: Site 'lh_palm_center' not found!")

        self.hand_geom_ids = [] #all hand geoms (for collision checking
        self.contact_body_ids = [] #body ids for contact distance checking
        excluded_keywords = ['wrist', 'forearm', 'palm']
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                self.hand_geom_ids.extend(self._get_geoms_id_of_body(i))
                name_lower = name.lower()
                is_excluded = any(k in name_lower for k in excluded_keywords)
                if not is_excluded:
                    self.contact_body_ids.append(i)


        syn_map = self.act_ctrl.get_synergy_map()
        self.syn_grasp = syn_map["grasp"] #stores ids of actuators for each synergy group
        self.syn_curl = syn_map["curl"]
        self.syn_spread = syn_map["spread"]
        self.syn_thumb_base = syn_map["thumb_base"]
        self.syn_thumb_flex = syn_map["thumb_flex"]
        self.syn_wrist = syn_map["wrist"]

        
        super(GraspPlanner, self).__init__(state)


    def set_hand_pose(self, state_struct):
        """将规划状态转换为MuJoCo的运动学配置
        map 13D synergy state to 24D joint configuration.
        """


        # 重置数据以获得干净的初始状态
        # mujoco.mj_resetData(self.model, self.data)
        
        # ===== 1. 设置手掌基本位置和姿态 =====
        self.data.qpos[0:3] = state_struct.get_position()
        self.data.qpos[3:7] = state_struct.get_quaternion()
        
        # ===== 2. 根据协同变量设置手指关节 =====
        self.act_ctrl.set_hand_state(state_struct)
        
        
        # ===== 3. 执行前向运动学计算 =====
        # 这计算所有位置、速度、加速度和接触点
        mujoco.mj_forward(self.model, self.data)

    def move(self):
        """
        Annealer 要求 self.state 是数组，但内部使用 StateStruct 进行清晰的状态操作
        仅负责修改数值状态，不进行物理更新
        """
        # 转换为 StateStruct 便于操作
        current = StateStruct(self.model, self.data)
        current.from_array(self.state)
        
        # 位置扰动
        # 随机扰动
        # 假设瓶子在 [0.3, 0, 0]，限制手部只能在附近 0.5m 范围内活动
        current.position += np.random.normal(0, 0.015, 3)
        bottle_pos = self.data.xpos[self.bottle_body_id]
        current.position = np.clip(current.position, bottle_pos - 0.5, bottle_pos + 0.5)

        # 姿态扰动 (四元数)
        current.quaternion += np.random.normal(0, 0.05, 4)
        current._normalize_quaternion()
        
        current.grasp += np.random.normal(0, 0.08)
        current.grasp = np.clip(current.grasp, 0.0, 1.0)  # 改为允许完整 [0, 1] 范围

        current.curl += np.random.normal(0, 0.05)
        current.curl = np.clip(current.curl, 0.0, 1.0)

        current.spread += np.random.normal(0, 0.05)
        current.spread = np.clip(current.spread, -0.2, 0.3)

        current.thumb_base += np.random.normal(0, 0.5)
        current.thumb_base = np.clip(current.thumb_base, 0.0, 1.0)

        current.thumb_flex += np.random.normal(0, 0.5)
        current.thumb_flex = np.clip(current.thumb_flex, 0.0, 1.0)

        # 转换回数组供 Annealer 使用
        self.state = current.to_array()

    # def energy(self):
    #     """抓取能量函数（优化目标）
        
    #     目标：
    #     1. 最小化手指到物体的距离 → 手靠近物体
    #     2. 手掌对准物体 → 便于抓取
    #     3. 避免碰撞和超出关节限位
    #     4. 倾向于中等抓取力度 → 既能接触又不过度闭合
    #     """
    #     # ===== 权重配置 =====
    #     W_PROXIMITY = 25.0        # 接近性权重
    #     W_ORIENTATION = 1    # 朝向性权重
    #     W_COLLISION_HAND = 2  # 手-物碰撞惩罚
    #     W_JOINT_LIMIT = 100    # 关节超限惩罚
        
    #     # 将13D数组转换为StateStruct
    #     state_struct = StateStruct(self.model, self.data)
    #     state_struct.from_array(self.state)
    #     self.set_hand_pose(state_struct)



    #     # ===== 1. 接近性能量：手指到物体的平均距离 =====
    #     total_dist = 0.0
    #     bottle_pos = self.data.xpos[self.bottle_body_id]
    #     palm_center_pos = self.data.site_xpos[self.palm_center_site_id]

    #     for b_id in self.contact_body_ids:
    #         total_dist += np.linalg.norm(self.data.xpos[b_id] - bottle_pos)
    #     total_dist += np.linalg.norm(palm_center_pos - bottle_pos)
    #     proximity_energy = total_dist / (len(self.contact_body_ids)+1)
        
    #     # ===== 2. 朝向性能量：手掌法向量与指向物体方向的夹角 =====
    #     # 手掌法向量 = 手掌坐标系的-Y轴（指向掌心侧）
    #     rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
    #     palm_normal = -rot_mat[:, 1]  # 指向掌心方向
        
    #     # 从手掌指向物体的方向
    #     vec_to_bottle = bottle_pos - palm_center_pos
    #     vec_to_bottle_norm = vec_to_bottle / (np.linalg.norm(vec_to_bottle) + 1e-6)
        
    #     # cos_angle = dot(palm_normal, vec_to_bottle_norm)
    #     # 当朝向一致时 = 1.0，反向时 = -1.0
    #     # energy = (1.0 - cos_angle) → 朝向一致时最小
    #     orientation_energy = (1.0 - np.dot(palm_normal, vec_to_bottle_norm)) * 1.5
        
    #     # ===== 3. 碰撞惩罚 =====
    #     collision_penalty = 0.0
    #     for i in range(self.data.ncon):
    #         con = self.data.contact[i]
    #         is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
    #         is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
    #         if is_bottle and is_hand and con.dist < 0:
    #             # 穿透越深，惩罚越大，而不是一刀切
    #             collision_penalty += abs(con.dist) * W_COLLISION_HAND * 200


        
    #     # ===== 4. 关节限位惩罚 =====
    #     joint_limit_penalty = 0.0
    #     for i in range(1, self.model.njnt):
    #         jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
    #         if jnt_name and self.hand_prefix in jnt_name:
    #             q_addr = self.model.jnt_qposadr[i]
    #             q_val = self.data.qpos[q_addr]
    #             low, high = self.model.jnt_range[i]
                
    #             # 超出下限
    #             if q_val < low:
    #                 joint_limit_penalty += np.square(low - q_val)
    #             # 超出上限
    #             elif q_val > high:
    #                 joint_limit_penalty += np.square(q_val - high)
        

    #     # ===== 总能量 =====
    #     total_energy = (
    #         proximity_energy * W_PROXIMITY +
    #         orientation_energy * W_ORIENTATION +
    #         collision_penalty +
    #         joint_limit_penalty * W_JOINT_LIMIT
    #     )
    #     return total_energy
    
    def energy(self):
        # 1. 更新物理状态
        state_struct = StateStruct(self.model, self.data)
        state_struct.from_array(self.state)
        self.set_hand_pose(state_struct) # 内部建议改用 mj_fwdPosition

        # 2. 获取瓶子和手的关键数据
        b_pos = self.data.xpos[self.bottle_body_id]
        b_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        points_world = self.data.xpos[self.contact_body_ids]
        palm_center_pos = self.data.site_xpos[self.palm_center_site_id]
        # 将手掌中心拼接到数组最后，构成 (N+1, 3) 的数组
        all_query_points_world = np.vstack([points_world, palm_center_pos])
        
        # 3. 【核心修正】坐标系转换：将手部点转到瓶子局部坐标系
        all_points_local = (all_query_points_world - b_pos) @ b_mat
        
        # 4. 【性能优化】批量计算距离

        closest_points_local, dists, _ = closest_point(self.bottle_mesh, all_points_local)
        
        # 5. 【逻辑修正】计算平均表面距离
        finger_dists = dists[:-1]
        palm_closest_point_local = closest_points_local[-1]
        proximity_energy = np.mean(finger_dists)
        
        # 6. 【功能增强】手掌中心也要靠近（给它更高权重）

        palm_dist = np.linalg.norm(palm_center_pos - b_pos)
        
        # 7. -----------------------------------------------------------
        # 计算能量项 2：朝向性 (Orientation) - 你的修改重点在这里
        # -----------------------------------------------------------
        
        # 将刚才查到的“瓶子表面最近点”转回世界坐标系
        # 公式：P_world = P_local @ R.T + T
        # (注意：因为 b_mat 是正交矩阵，转置等于逆)
        target_point_world = palm_closest_point_local @ b_mat.T + b_pos
        
        # 计算从“掌心”指向“瓶子表面最近点”的向量
        vec_to_surface = target_point_world - palm_center_pos
        
        # 归一化 (防止除以0)
        dist_to_surface = np.linalg.norm(vec_to_surface)
        vec_to_surface_norm = vec_to_surface / (dist_to_surface + 1e-6)
        
        # 获取手掌法线
        palm_rot = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_normal = -palm_rot[:, 1] # 假设 -Y 是掌心法线
        
        # 计算点积能量 (1.0 - cos_angle)
        # 如果手掌正对表面，点积为 1，能量为 0
        orientation_energy = 1.0 - np.dot(palm_normal, vec_to_surface_norm)

        # 8. 简单的碰撞惩罚（如果没有这个，手会飞进瓶子中心）
        collision_penalty = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.dist < -0.0002: # 穿透超过2mm
                collision_penalty += 10.0 + abs(con.dist) * 50000

        # 总能量（权重你可以微调）
        total_energy = (
            proximity_energy * 100.0 + 
            palm_dist * 50.0 +
            orientation_energy * 20.0 +
            collision_penalty
        )

        
        return total_energy



    def _get_geoms_id_of_body(self, body_id):
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]
    
    def get_min_dist(self, body_pos):
        #get mesh of bottle
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "..", "assets", "bottle.obj")
        mesh = trimesh.load(file_path)

        closest_point, distance, triangle_id = mesh.proximity.closest_point([body_pos])
    
        # distance[0] 即为最短距离
        return distance[0]