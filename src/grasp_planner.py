"""
抓取规划器：使用模拟退火优化Shadow Hand的抓取姿态
"""
from simanneal import Annealer
import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer
import argparse
from src.grasp_state import StateStruct
from src.grasp_control import GraspControl


class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name, hand_body_prefix='lh_'):
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix
        self.act_ctrl = GraspControl(model, data)

        # get body and geom ids for bottle
        self.bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body_name)
        self.bottle_geom_ids = self._get_geoms_id_of_body(self.bottle_body_id)
        
        # get floor geom id
        geom_name = "floor"
        self.floor_geom_id = model.geom(geom_name).id

        self.palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.hand_prefix}palm")
    
        self.hand_geom_ids = [] #all hand geoms (for collision checking
        self.contact_body_ids = [] #body ids for contact distance checking
        excluded_keywords = ['wrist', 'forearm']
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

    def _get_geoms_id_of_body(self, body_id):
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]

    def set_hand_pose(self, state_struct):
        """将规划状态转换为MuJoCo的运动学配置
        map 13D synergy state to 24D joint configuration.
        """


        # 重置数据以获得干净的初始状态
        mujoco.mj_resetData(self.model, self.data)
        
        # ===== 1. 设置手掌基本位置和姿态 =====
        self.data.qpos[0:3] = state_struct.get_position()
        self.data.qpos[3:7] = state_struct.get_quaternion()
        
        # ===== 2. 根据协同变量设置手指关节 =====
        self.act_ctrl.set_hand_state(state_struct)
        
        
        # ===== 3. 执行前向运动学计算 =====
        # 这计算所有位置、速度、加速度和接触点
        mujoco.mj_forward(self.model, self.data)

    def move(self):
        """模拟退火的扰动函数
        
        Annealer 要求 self.state 是数组，但内部使用 StateStruct 进行清晰的状态操作
        """
        # 转换为 StateStruct 便于操作
        current = StateStruct()
        current.from_array(self.state)
        
        # 位置扰动
        if np.random.random() > 0.7:
            # 随机扰动
            current.position += np.random.normal(0, 0.015, 3)
        else:
            # 朝向物体移动
            bottle_pos = self.data.xpos[self.bottle_body_id]
            palm_pos = self.data.xpos[self.palm_body_id]+ np.array([0.0, 0.0, -0.03])  
            direction = bottle_pos - palm_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                current.position += (direction / dist) * 0.002

        # 姿态扰动 (四元数)
        current.quaternion += np.random.normal(0, 0.05, 4)
        current._normalize_quaternion()
        
        # 抓取强度扰动
        current.grasp += np.random.normal(0, 0.08)
        current.grasp = np.clip(current.grasp, 0.0, 1.0)  # 改为允许完整 [0, 1] 范围

        current.curl += np.random.normal(0, 0.05)
        current.curl = np.clip(current.curl, 0.0, 1.0)

        # 展开程度扰动
        current.spread += np.random.normal(0, 0.05)
        current.spread = np.clip(current.spread, -0.2, 0.3)

        current.thumb_base += np.random.normal(0, 0.05)
        current.thumb_base = np.clip(current.thumb_base, 0.0, 1.0)

        current.thumb_flex += np.random.normal(0, 0.05)
        current.thumb_flex = np.clip(current.thumb_flex, 0.0, 1.0)

        self.set_hand_pose(current)
        
        # 转换回数组供 Annealer 使用
        self.state = current.to_array()

    def energy(self):
        """抓取能量函数（优化目标）
        
        目标：
        1. 最小化手指到物体的距离 → 手靠近物体
        2. 手掌对准物体 → 便于抓取
        3. 避免碰撞和超出关节限位
        4. 倾向于中等抓取力度 → 既能接触又不过度闭合
        """
        # ===== 权重配置 =====
        W_PROXIMITY = 25.0        # 接近性权重
        W_ORIENTATION = 10.0      # 朝向性权重
        W_COLLISION_HAND = 200.0  # 手-物碰撞惩罚
        W_COLLISION_FLOOR = 500.0 # 手-地面碰撞惩罚
        W_JOINT_LIMIT = 1000.0    # 关节超限惩罚
        W_GRASP_PRIOR = 1.0       # 抓取力度先验权重
        GRASP_TARGET = 1        # 目标抓取力度（偏好值）
        COLLISION_THRESHOLD = -0.005  # 碰撞判定阈值(m)，<0为穿透
        
        # 将13D数组转换为StateStruct
        state_struct = StateStruct()
        state_struct.from_array(self.state)
        # self.set_hand_pose(state_struct)
        
        # ===== 1. 接近性能量：手指到物体的平均距离 =====
        bottle_pos = self.data.xpos[self.bottle_body_id]
        total_dist = 0.0
        for b_id in self.contact_body_ids:
            if b_id == self.palm_body_id:
                dist = np.linalg.norm(self.data.xpos[b_id] + np.array([0.0, 0.0, -0.03]) - bottle_pos)
            else:
                dist = np.linalg.norm(self.data.xpos[b_id] - bottle_pos)
            total_dist += dist
        proximity_energy = total_dist / len(self.contact_body_ids)
        
        # ===== 2. 朝向性能量：手掌法向量与指向物体方向的夹角 =====
        # 手掌法向量 = 手掌坐标系的-Y轴（指向掌心侧）
        rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_normal = -rot_mat[:, 1]  # 指向掌心方向
        
        # 从手掌指向物体的方向
        vec_to_bottle = bottle_pos - self.data.xpos[self.palm_body_id]
        vec_to_bottle_norm = vec_to_bottle / (np.linalg.norm(vec_to_bottle) + 1e-6)
        
        # cos_angle = dot(palm_normal, vec_to_bottle_norm)
        # 当朝向一致时 = 1.0，反向时 = -1.0
        # energy = (1.0 - cos_angle) → 朝向一致时最小
        orientation_energy = (1.0 - np.dot(palm_normal, vec_to_bottle_norm)) * 1.5
        
        # ===== 3. 碰撞惩罚 =====
        collision_penalty = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
            is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
            
            # 手-物穿透（con.dist < 0表示穿透深度）
            if is_bottle and is_hand and con.dist < COLLISION_THRESHOLD:
                collision_penalty += W_COLLISION_HAND
            
            # 手-地面碰撞（避免手碰地）
            is_floor = (con.geom1 == self.floor_geom_id or con.geom2 == self.floor_geom_id)
            if is_floor and is_hand:
                collision_penalty += W_COLLISION_FLOOR
        
        # ===== 4. 关节限位惩罚 =====
        joint_limit_penalty = 0.0
        for i in range(1, self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name and self.hand_prefix in jnt_name:
                q_addr = self.model.jnt_qposadr[i]
                q_val = self.data.qpos[q_addr]
                low, high = self.model.jnt_range[i]
                
                # 超出下限
                if q_val < low:
                    joint_limit_penalty += np.square(low - q_val)
                # 超出上限
                elif q_val > high:
                    joint_limit_penalty += np.square(q_val - high)
        
        # ===== 5. 抓取力度先验（倾向于中等力度） =====
        # 目标grasp=GRASP_TARGET：既能有效接触，又避免过度闭合
        grasp_prior_energy = np.square(state_struct.grasp - GRASP_TARGET)
        
        # ===== 总能量 =====
        total_energy = (
            proximity_energy * W_PROXIMITY +
            orientation_energy * W_ORIENTATION +
            collision_penalty +
            joint_limit_penalty * W_JOINT_LIMIT +
            grasp_prior_energy * W_GRASP_PRIOR
        )
        
        return total_energy