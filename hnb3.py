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
        # 排除 free joint 和不包含前缀的关节
        if jnt_name and hand_prefix in jnt_name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE and "lh_WR" not in jnt_name:
            qpos_adr = model.jnt_qposadr[i]
            jnt_name_lower = jnt_name.lower()
            
            if 'th' in jnt_name_lower: 
                thumb_indices.append(qpos_adr)
                print(f"Thumb joint found: {jnt_name} at qpos adr {qpos_adr}")
            elif 'j4' in jnt_name_lower or 'abd' in jnt_name_lower:
                abd_indices.append(qpos_adr)
                print(f"Abduction joint found: {jnt_name} at qpos adr {qpos_adr}")
            else:
                flex_indices.append(qpos_adr)
                print(f"Flexion joint found: {jnt_name} at qpos adr {qpos_adr}")

                
    return flex_indices, abd_indices, thumb_indices

def qpos_to_ctrl(model, planner, target_pose):
    ctrl_cmd = np.zeros(model.nu)
    grasp_val = np.clip(target_pose[7], 0.0, 1.0)
    spread_val = np.clip(target_pose[8], -0.2, 0.3)
    
    # 增加过压系数
    grasp_force_factor = 1.15 

    for i in range(model.nu):
        # 使用更快的内置 ID 获取方式
        jnt_id = model.actuator_trnid[i, 0]
        jnt_adr = model.jnt_qposadr[jnt_id]
        
        # 获取该执行器的物理限制
        ctrl_range = model.actuator_ctrlrange[i]
        
        if jnt_adr in planner.flex_adrs:
            # 加入非线性：握得越紧，指尖闭合速率越高
            val = (grasp_val ** 1.1) * 1.7 * grasp_force_factor
        elif jnt_adr in planner.thumb_adrs:
            val = grasp_val * 1.3
        elif jnt_adr in planner.abd_adrs:
            val = spread_val
        else:
            val = 0.0
            
        # 确保不超出执行器限位
        ctrl_cmd[i] = np.clip(val, ctrl_range[0], ctrl_range[1])
        
    return ctrl_cmd

# --- 规划器类 ---

class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name, hand_body_prefix='lh_'):
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix

        # 获取对象 ID
        self.bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body_name)
        self.bottle_geom_ids = self._get_geoms_id_of_body(self.bottle_body_id)
        
        # 获取地面几何 ID
        geom_name = "floor"
        self.floor_geom_id = model.geom(geom_name).id


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
        """ 带有环境约束和关节限位惩罚的能量函数 """
        self.set_hand_pose(self.state)
        
        # --- 1. 距离项 (Distance Energy) ---
        bottle_pos = self.data.xpos[self.bottle_body_id]
        total_dist = 0.0
        for b_id in self.contact_body_ids:
            dist = np.linalg.norm(self.data.xpos[b_id] - bottle_pos)
            total_dist += dist
        avg_dist = total_dist / len(self.contact_body_ids)

        # --- 2. 朝向项 (Orientation Energy) ---
        rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_normal = -rot_mat[:, 1] # 假设-Y是手心
        vec_to_bottle = bottle_pos - self.data.xpos[self.palm_body_id]
        vec_to_bottle_norm = vec_to_bottle / (np.linalg.norm(vec_to_bottle) + 1e-6)
        orient_energy = (1.0 - np.dot(palm_normal, vec_to_bottle_norm)) * 1.5 

        # --- 3. 碰撞惩罚 (Collision Penalty) ---
        collision_penalty = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
            is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
            if is_bottle and is_hand:
                if con.dist < -0.005: # 允许轻微穿模，重度穿模惩罚
                    collision_penalty += 200.0
            if (con.geom1 == self.floor_geom_id or con.geom2 == self.floor_geom_id) and is_hand:
                collision_penalty += 500.0

        # --- 4. 新增：关节限位惩罚 (Joint Limit Penalty) ---
        # 这一步至关重要，防止 set_hand_pose 计算出的 qpos 超出 XML 范围
        limit_penalty = 0.0
        # 我们只检查手部的关节 (跳过前7位 freejoint)
        for i in range(1, self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name and self.hand_prefix in jnt_name:
                q_addr = self.model.jnt_qposadr[i]
                q_val = self.data.qpos[q_addr]
                low, high = self.model.jnt_range[i]
                
                # 如果超出范围，计算差值的平方
                if q_val < low:
                    limit_penalty += np.square(low - q_val)
                elif q_val > high:
                    limit_penalty += np.square(q_val - high)
        
        # 权重设置：关节限制是硬约束，权重应设得非常高
        limit_weight = 1000.0 

        # --- 5. 姿态先验 (Pose Energy) ---
        pose_energy = np.square(self.state[7] - 0.4) 

        # 汇总所有能量
        return (avg_dist * 25.0 + 
                orient_energy + 
                collision_penalty + 
                pose_energy + 
                limit_penalty * limit_weight)


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

    # --- 动画控制变量 ---
    planning_done = False
    
    # 目标变量
    target_palm_pos = initial_guess[0:3].copy()
    target_palm_quat = initial_guess[3:7].copy()
    final_grasp_synergy = 0.0 # 规划算出的最终握力
    final_spread_synergy = 0.0
    
    # 动态过程变量
    current_grasp_val = 0.0   # 当前手指弯曲度 (动态增加)
    execution_start_time = 0.0 # 记录开始执行的时间

    print(">>> 启动 MuJoCo 查看器...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- 1. 规划阶段 (Reset 时触发) ---
            if data.time < 0.002:
                print("\n[状态] 正在规划最佳抓取点...")
                
                # 随机化初始点并运行退火
                planner.state = initial_guess.copy()
                planner.state[0:3] += np.random.uniform(-0.05, 0.05, 3)
                best_pose, energy = planner.anneal()
                
                print(f"[完成] 目标 Grasp: {best_pose[7]:.2f}")

                # 记录目标，但不立即应用
                target_palm_pos = best_pose[0:3]
                target_palm_quat = best_pose[3:7]
                final_grasp_synergy = best_pose[7]
                final_spread_synergy = best_pose[8]
                
                # 重置执行状态
                current_grasp_val = 0.0 # 手指重置为张开
                execution_start_time = time.time() # 记录当前真实时间作为动画起点
                planning_done = True
                
                # 将仿真时间稍微推前一点避免重复触发
                data.time = 0.002

            # --- 2. 执行阶段 (动画逻辑) ---
            if planning_done:
                # 计算动画经过的时间
                anim_time = time.time() - execution_start_time
                
                # A. 手掌移动 (0.0s ~ 1.0s): 使用插值平滑移动过去
                # 这是一个简单的线性插值 (Lerp)
                move_duration = 1.0
                move_progress = np.clip(anim_time / move_duration, 0.0, 1.0)
                
                # 当前手掌位置 = 初始位置 * (1-t) + 目标位置 * t
                # 为了简单，这里假设从当前位置平滑过渡不太容易(因为物理步进在变)，
                # 我们直接把手掌“吸”到目标位置，或者如果你想要移动动画，可以使用 move_progress
                # 这里为了稳定，我们让手掌直接到位，主要展示手指慢动作：
                data.qpos[0:3] = target_palm_pos
                data.qpos[3:7] = target_palm_quat
                data.qpos[7] = 0.0  # 手指张开
                data.qpos[8] = final_spread_synergy

                # B. 手指抓取 (1.0s ~ 3.0s): 慢动作弯曲
                grasp_start_time = 1.0
                grasp_duration = 2.0 # 抓取动作持续 2 秒
                
                if anim_time > grasp_start_time:
                    # 计算抓取进度 0.0 -> 1.0
                    grasp_progress = (anim_time - grasp_start_time) / grasp_duration
                    grasp_progress = np.clip(grasp_progress, 0.0, 1.0)
                    
                    # 动态更新当前的协同值
                    current_grasp_val = final_grasp_synergy * grasp_progress
                else:
                    current_grasp_val = 0.0 # 移动期间保持张开

                # C. 计算并应用控制信号
                # 构造一个临时的 pose 向量用于计算 ctrl
                temp_pose = np.zeros(9)
                temp_pose[7] = current_grasp_val     # 动态变化的握力
                temp_pose[8] = final_spread_synergy  # 侧摆保持不变
                
                # 每一帧都重新计算 ctrl
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

