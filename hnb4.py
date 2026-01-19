"""
尝试修改：
    1. 手瞬移到指定位置时，五指会抽动，有可能弹飞水瓶。
    2. 指关节弯曲时部分关节不弯曲（推测get_synergy_mapping或energy有问题），似乎只有近端关节弯曲。 
改动部分：
    1. get_synergy_mapping
    2. qpos_to_ctrl
    3. 新增 quaternion_slerp 
    4. main

"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer

# --- 辅助函数 ---



def get_synergy_mapping(model, hand_prefix):
    """
    根据关节名称自动分类关节索引。
    返回: flex_indices (四指弯曲), abd_indices (四指侧摆), thumb_indices (大拇指)
    四指弯曲是指关节名称中包含 'j1', 'j2', 'j3' 的关节，负责手指的弯曲动作。
    j1 是近端指节，j2 是中间指节，j3 是远端指节。
    侧摆是指关节名称中包含 'j4' 或 'abd' 的关节，能够实现手指张开/合拢动作。
    大拇指是指关节名称中包含 'th' 的关节，负责大拇指的各种动作。
    """
    flex_indices = [] 
    abd_indices = [] 
    thumb_indices = []

    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name and hand_prefix in jnt_name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE and "lh_WR" not in jnt_name:
            qpos_adr = model.jnt_qposadr[i]
            jnt_name_lower = jnt_name.lower()
            
            # Shadow Hand 命名惯例处理
            # th = 大拇指
            # j4 或 abd = 侧摆/张开
            # j1, j2, j3 = 弯曲 (无论顺序如何，都归类为 flex)
            
            if 'th' in jnt_name_lower: 
                thumb_indices.append(qpos_adr)
            elif 'j4' in jnt_name_lower or 'abd' in jnt_name_lower:
                abd_indices.append(qpos_adr)
            elif any(x in jnt_name_lower for x in ['j1', 'j2', 'j3', 'flex']):
                flex_indices.append(qpos_adr)
            else:
                # 兜底：如果有关节没被上述规则捕获，默认归为弯曲
                flex_indices.append(qpos_adr)
                
    return flex_indices, abd_indices, thumb_indices

def qpos_to_ctrl(model, planner, target_pose):
    ctrl_cmd = np.zeros(model.nu)
    
    # 归一化的控制信号 (0.0 ~ 1.0)
    grasp_val = np.clip(target_pose[7], 0.0, 1.0)
    # 侧摆信号，通常需要映射到 -1 ~ 1 或特定角度，这里假设输入是归一化意图
    spread_val = np.clip(target_pose[8], -1.0, 1.0) 

    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_adr = model.jnt_qposadr[jnt_id]
        
        # 获取该执行器的物理控制范围 (例如: [-0.3, 0.3] 或 [0, 1.57])
        min_ctrl, max_ctrl = model.actuator_ctrlrange[i]
        ctrl_span = max_ctrl - min_ctrl
        
        val = 0.0
        
        if jnt_adr in planner.flex_adrs:
            # 弯曲：将 0~1 映射到 min~max
            # 注意：grasp_val ** 1.5 让小数值弯曲更慢（非线性），看起来更自然
            ratio = grasp_val ** 1.2 
            val = min_ctrl + ratio * ctrl_span
            
        elif jnt_adr in planner.thumb_adrs:
            # 大拇指：同理
            ratio = grasp_val 
            val = min_ctrl + ratio * ctrl_span
            
        elif jnt_adr in planner.abd_adrs:
            # 侧摆：假设 spread_val 在 -1(闭合) 到 1(张开) 之间
            # 我们将其映射到 ctrl 范围的中心向两边扩展
            mid = (max_ctrl + min_ctrl) / 2
            half_span = ctrl_span / 2
            val = mid + spread_val * half_span
            
        else:
            # 其他关节（如手腕）保持原位或归零
            val = 0.0 # 或者保持当前状态
            
        ctrl_cmd[i] = np.clip(val, min_ctrl, max_ctrl)
        
    return ctrl_cmd

def quaternion_slerp(q0, q1, t):
    """简单的四元数球面插值"""
    # 归一化
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return (1.0 - t) * q0 + t * q1

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1









# --- 规划器类 ---

class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name, hand_body_prefix='lh_'):
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix

        # 1. 获取对象 ID
        self.bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body_name)
        self.bottle_geom_ids = self._get_geoms_id_of_body(self.bottle_body_id)

        self.palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.hand_prefix}palm")
        self.hand_geom_ids = []
        self.contact_body_ids = [] # <--- 新增：用于存储计算距离的所有Body ID

        # 定义需要排除的关键词
        excluded_keywords = ['wrist', 'forearm']

        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                # 记录所有手部 geom 用于碰撞检测
                self.hand_geom_ids.extend(self._get_geoms_id_of_body(i))
                
                # --- 修改逻辑开始 ---
                # 筛选用于计算距离的 Body：必须包含手前缀，且不包含 'wrist' 和 'forearm'
                name_lower = name.lower()
                is_excluded = any(k in name_lower for k in excluded_keywords)
                
                if not is_excluded:
                    self.contact_body_ids.append(i)
                    # 打印一下确认是否正确获取 (调试用)
                    print(f"Distance calculation includes: {name}")
                # --- 修改逻辑结束 ---

        # 3. 获取关节映射 (保持不变)
        self.flex_adrs, self.abd_adrs, self.thumb_adrs = get_synergy_mapping(model, hand_body_prefix)
        
        super(GraspPlanner, self).__init__(state)

    def _get_geoms_id_of_body(self, body_id):
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]

    def set_hand_pose(self, state):
        """仅用于能量计算的运动学设置 (无物理，时间静止)"""
        # 注意：这里我们不重置整个 simulation，只重置数据流以进行纯几何计算
        # 如果频繁调用 mj_resetData 可能会比较慢，但在退火中通常是可以接受的
        # 为了更快的速度，可以只覆盖 qpos 并调用 mj_kinematics (如果不需要接触检测)
        # 但这里需要 check_collision，所以必须 mj_forward
        
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. 设置基座
        self.data.qpos[0:7] = state[0:7]
        
        # 2. 解析协同变量并设置关节
        grasp_synergy = np.clip(state[7], 0.0, 1.0)
        spread_synergy = np.clip(state[8], -0.2, 0.3)

        # 弯曲
        flex_angle = grasp_synergy * 1.5
        for adr in self.flex_adrs:
            self.data.qpos[adr] = flex_angle

        # 侧摆
        for adr in self.abd_adrs:
            self.data.qpos[adr] = spread_synergy * 0.5

        # 大拇指
        thumb_flex = grasp_synergy * 1.2
        for i, adr in enumerate(self.thumb_adrs):
            self.data.qpos[adr] = thumb_flex * (0.5 + i * 0.2)

        # 3. 更新几何状态
        mujoco.mj_forward(self.model, self.data)

    def move(self):
        """模拟退火的扰动函数"""
        # A. 随机位移 vs 启发式位移
        if np.random.random() > 0.7:
            # 纯随机微调
            self.state[0:3] += np.random.normal(0, 0.015, 3)
        else:
            # 启发式：移向瓶子
            bottle_pos = self.data.xpos[self.bottle_body_id] # 获取当前瓶子位置
            palm_pos = self.data.xpos[self.palm_body_id]
            direction = bottle_pos - palm_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                self.state[0:3] += (direction / dist) * 0.002 # 步长 2mm

        # B. 旋转扰动
        self.state[3:7] += np.random.normal(0, 0.05, 4)
        self.state[3:7] /= np.linalg.norm(self.state[3:7]) # 归一化四元数
        
        # C. 协同变量扰动
        self.state[7] += np.random.normal(0, 0.08) # Grasp
        self.state[7] = np.clip(self.state[7], 0.1, 0.9) # 保持在有效范围内

        self.state[8] += np.random.normal(0, 0.05) # Spread
        self.state[8] = np.clip(self.state[8], -0.1, 0.2)

    def energy(self):
        """ 修改后的能量函数 """
        self.set_hand_pose(self.state)
        
        # --- 1. 修改：全身距离项 (Distance Energy) ---
        bottle_pos = self.data.xpos[self.bottle_body_id]
        total_dist = 0.0
        
        # 遍历所有筛选出的 Body (手掌 + 各个指节)
        for b_id in self.contact_body_ids:
            # 计算欧氏距离
            if b_id == self.palm_body_id: # palm body id
                actual_pos = self.data.xpos[b_id] - [0, 0, 0.3]  # 调整 palm 位置以更准确反映接触点
                dist = np.linalg.norm(actual_pos - bottle_pos)
            else:
                dist = np.linalg.norm(self.data.xpos[b_id] - bottle_pos)
            total_dist += dist
            
        # 计算平均距离
        avg_dist = total_dist / len(self.contact_body_ids)

        # --- 2. 朝向项 (Orientation Energy) ---
        rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
        # 假设 -Y 轴是手心朝向 (根据你的XML结构推断)
        palm_normal = -rot_mat[:, 1] 
        
        palm_pos = self.data.xpos[self.palm_body_id]
        vec_to_bottle = bottle_pos - palm_pos
        vec_to_bottle_norm = vec_to_bottle / (np.linalg.norm(vec_to_bottle) + 1e-6)
        
        alignment = np.dot(palm_normal, vec_to_bottle_norm)
        orient_energy = (1.0 - alignment) * 1.5 

        # --- 3. 姿态先验 (Pose Energy) ---
        pose_energy = np.square(self.state[7] - 0.35) 

        # --- 4. 碰撞惩罚 (Collision Penalty) ---
        collision_penalty = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
            is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
            
            if is_bottle and is_hand:
                # 稍微允许一点负距离 (穿模) 以确保紧密接触，但超过一定阈值则重罚
                if con.dist < -0.00: 
                    collision_penalty += 100.0
                else:
                    # 奖励轻微的接触 (可选，如果希望手主动去贴合)
                    collision_penalty -= 0.01 

        # 权重系数调整：由于现在计算的点多了，avg_dist 可能会比只算指尖大一些或者小一些，
        # 建议适当增加距离项的权重，让手更贴近物体。
        return avg_dist * 20.0 + orient_energy + pose_energy + collision_penalty


# --- 主程序 ---

def main():
    model_path = 'shadow_hand/scene_left.xml' 
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    data = mujoco.MjData(model)

    # 初始状态
    initial_guess = np.zeros(9)
    initial_guess[0:3] = [0.4, 0.4, 0.3] 
    initial_guess[3] = 1.0 
    initial_guess[7] = 0.0 # 初始完全张开

    planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')
    planner.steps = 2500

    # 状态标志
    planning_done = False
    
    # 记录目标参数
    target_palm_pos = np.zeros(3)
    target_palm_quat = np.zeros(4)
    final_grasp_synergy = 0.0 
    final_spread_synergy = 0.0
    
    execution_start_time = 0.0

    print(">>> 启动 MuJoCo 查看器...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- 阶段 1: 规划与瞬移 ---
            # 使用 planning_done 标志位来防止无限循环，因为 set_hand_pose 会重置 time
            if not planning_done:
                print("\n[状态] 正在规划最佳抓取点...")
                
                # 1. 运行退火算法寻找最佳点
                planner.state = initial_guess.copy()
                best_pose, energy = planner.anneal()
                
                print(f"[完成] 瞬移目标 Grasp: {best_pose[7]:.2f}")

                # 2. 准备瞬移数据
                # 构造一个“张开手”的姿态，但位置在目标点
                teleport_pose = best_pose.copy()
                teleport_pose[7] = 0.0  # 强制手指张开 (Grasp = 0)
                teleport_pose[8] = 0.0  # 强制手指居中 (Spread = 0)
                
                # 3. 执行瞬移 (使用 planner 的辅助函数)
                # 这会调用 mj_resetData，清除所有速度(qvel)和加速度，
                # 确保手是“静止”出现在目标点的，没有任何惯性。
                planner.set_hand_pose(teleport_pose)
                
                # 保存后续抓取需要的目标参数
                target_palm_pos = best_pose[0:3]
                target_palm_quat = best_pose[3:7]
                final_grasp_synergy = best_pose[7]  # 这是规划出的最终握力
                final_spread_synergy = best_pose[8]
                
                # 标记规划完成，重置时间
                planning_done = True
                execution_start_time = time.time()
                
                # 由于 set_hand_pose 重置了仿真时间，我们需要确保 viewer 同步
                viewer.sync()
                continue # 跳过这一帧，让物理引擎在新位置初始化

            # --- 阶段 2: 锁定基座 & 执行抓取 ---
            if planning_done:
                # [关键操作 1]：每帧强制锁定基座位置（钉在空中）
                # 即使有碰撞反弹，这一步也会强制把手腕拉回目标点
                data.qpos[0:3] = target_palm_pos
                data.qpos[3:7] = target_palm_quat
                
                # [关键操作 2]：每帧消除基座速度，防止震荡
                data.qvel[0:6] = 0.0

                # 动画逻辑：
                # 0.5秒缓冲 -> 2.0秒抓取
                anim_time = time.time() - execution_start_time
                wait_time = 0.5    # 瞬移后停顿 0.5 秒，让人看清
                grasp_duration = 2.0 # 抓取过程持续 2 秒
                
                current_grasp_val = 0.0
                
                if anim_time < wait_time:
                    # 缓冲期：保持张开，不动
                    current_grasp_val = 0.0
                else:
                    # 抓取期：计算进度
                    grasp_t = (anim_time - wait_time) / grasp_duration
                    grasp_t = np.clip(grasp_t, 0.0, 1.0)
                    current_grasp_val = final_grasp_synergy * grasp_t

                # 构造控制指令
                temp_pose = np.zeros(9)
                temp_pose[7] = current_grasp_val      # 动态弯曲
                temp_pose[8] = final_spread_synergy   # 保持规划的侧摆角度
                
                # 计算并应用力矩
                # 注意：这里调用的是上一轮修复过的 qpos_to_ctrl
                ctrl_cmd = qpos_to_ctrl(model, planner, temp_pose)
                data.ctrl[:] = ctrl_cmd

                # 物理步进
                mujoco.mj_step(model, data)

            viewer.sync()
            
            # 帧率控制
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

