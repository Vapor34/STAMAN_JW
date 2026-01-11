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
    """
    flex_indices = [] 
    abd_indices = [] 
    thumb_indices = []

    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        # 排除 free joint 和不包含前缀的关节
        if jnt_name and hand_prefix in jnt_name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = model.jnt_qposadr[i]
            jnt_name_lower = jnt_name.lower()
            
            if 'th' in jnt_name_lower: 
                thumb_indices.append(qpos_adr)
            elif 'j4' in jnt_name_lower or 'abd' in jnt_name_lower:
                abd_indices.append(qpos_adr)
            else:
                flex_indices.append(qpos_adr)
                
    return flex_indices, abd_indices, thumb_indices

def qpos_to_ctrl(model, planner, target_pose):
    """
    将规划出的状态 (synergy值) 转换为电机控制信号 (ctrl)。
    包含 Force Closure 策略：让控制目标比规划目标稍大，以产生握力。
    """
    ctrl_cmd = np.zeros(model.nu)
    
    # 解析 target_pose (来自规划器的输出)
    grasp_synergy = np.clip(target_pose[7], 0.0, 1.0)
    spread_synergy = np.clip(target_pose[8], -0.2, 0.3)
    
    # 策略：Over-closing (过关闭)。
    # 规划时假设抓到 1.0 的位置刚好接触，控制时命令电机去 1.1 的位置，
    # 这样物理引擎会计算出接触力。
    grasp_synergy_cmd = min(grasp_synergy * 1.1, 1.0) 

    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not act_name: continue
        
        # 通过 actuator 找到对应的 joint
        jnt_id = model.actuator_trnid[i, 0] 
        jnt_qpos_adr = model.jnt_qposadr[jnt_id]
        
        target_angle = 0.0
        
        # 根据关节类型分配角度
        if jnt_qpos_adr in planner.thumb_adrs:
            # 大拇指：基础弯曲 + 指尖增强
            target_angle = grasp_synergy_cmd * 1.2
            if "thdistal" in act_name.lower() or "thmiddle" in act_name.lower():
                 target_angle *= 1.2 
            
        elif jnt_qpos_adr in planner.abd_adrs:
            # 侧摆
            target_angle = spread_synergy
            
        elif jnt_qpos_adr in planner.flex_adrs:
            # 四指弯曲：设定较大的最大弯曲角以确保握紧
            target_angle = grasp_synergy_cmd * 1.6 
            
        ctrl_cmd[i] = target_angle
        
    return ctrl_cmd

# --- 规划器类 ---

class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name, hand_body_prefix='lh_'):
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix

        # 1. 获取对象 ID
        self.bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body_name)
        self.bottle_geom_ids = self._get_geoms_of_body(self.bottle_body_id)

        self.palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.hand_prefix}palm")
        self.hand_geom_ids = []
        self.contact_body_ids = [] # <--- 新增：用于存储计算距离的所有Body ID

        # 定义需要排除的关键词
        excluded_keywords = ['wrist', 'forearm']

        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                # 记录所有手部 geom 用于碰撞检测
                self.hand_geom_ids.extend(self._get_geoms_of_body(i))
                
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

    def _get_geoms_of_body(self, body_id):
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
            palm_pos = self.state[0:3]
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
        pose_energy = np.square(self.state[7] - 0.35) * 2.0

        # --- 4. 碰撞惩罚 (Collision Penalty) ---
        collision_penalty = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
            is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
            
            if is_bottle and is_hand:
                # 稍微允许一点负距离 (穿模) 以确保紧密接触，但超过一定阈值则重罚
                if con.dist < -0.005: 
                    collision_penalty += 100.0
                else:
                    # 奖励轻微的接触 (可选，如果希望手主动去贴合)
                    collision_penalty -= 0.1 

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

    # 1. 初始猜测状态 [x, y, z, qw, qx, qy, qz, grasp_syn, spread_syn]
    initial_guess = np.zeros(9)
    # 假设瓶子在 0.4, 0.4，手初始在上方
    initial_guess[0:3] = [0.4, 0.4, 0.3] 
    initial_guess[3] = 1.0 # w=1 (无旋转)
    initial_guess[7] = 0.5 # 初始协同值

    planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')
    
    # 2. 退火参数设置
    planner.steps = 2500     # 迭代次数
    planner.Tmax = 20.0      # 初始温度
    planner.Tmin = 0.001     # 结束温度

    # 3. 运行时变量
    current_ctrl_target = np.zeros(model.nu)
    target_palm_pos = initial_guess[0:3].copy()
    target_palm_quat = initial_guess[3:7].copy()
    
    planning_done = False # 标志位：是否已完成规划

    print(">>> 启动 MuJoCo 查看器...")
    print(">>> 点击 Viewer 的 'Reset' 按钮将触发新的抓取规划。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- 逻辑控制 ---
            
            # A. 检测 Reset 信号 (time 归零) -> 触发规划
            if data.time < 0.002:
                print("\n[状态] 检测到 Reset，开始规划...")
                
                # 1. 稍微随机化初始点，避免每次结果完全一样
                random_start = initial_guess.copy()
                random_start[0:3] += np.random.uniform(-0.05, 0.05, 3)
                planner.state = random_start
                
                # 2. 运行模拟退火 (阻塞式计算，此时 Viewer 会卡顿一下)
                t0 = time.time()
                best_pose, energy = planner.anneal()
                print(f"[结果] 规划耗时 {time.time()-t0:.2f}s, 能量: {energy:.4f}")
                print(f"[目标] Grasp Synergy: {best_pose[7]:.2f}")

                # 3. 将规划结果转换为控制指令
                # 此时我们只更新"目标变量"，不直接改写 data.qpos (除非为了瞬移调试)
                current_ctrl_target = qpos_to_ctrl(model, planner, best_pose)
                target_palm_pos = best_pose[0:3]
                target_palm_quat = best_pose[3:7]

                # 4. 可选：规划完成后，立刻将手“瞬移”到目标位置附近，方便观察
                # 这样可以避免手从很远的地方飞过来撞翻瓶子
                data.qpos[0:3] = target_palm_pos
                data.qpos[3:7] = target_palm_quat
                # 手指也先瞬移到半张开状态 (避免鬼畜)
                mujoco.mj_forward(model, data) # 刷新几何

                planning_done = True
                
                # 防止循环重复触发，手动推进一点时间
                if data.time == 0:
                    data.time = 0.0001

            # B. 物理模拟循环
            if planning_done:
                # 1. 手指控制: 使用计算好的 Force Closure 角度
                # 确保每个 actuator 都被赋值
                if len(current_ctrl_target) == model.nu:
                    data.ctrl[:] = current_ctrl_target
                
                # 2. 手腕控制 (Floating Base Fixed)
                # 因为没有机械臂，我们需要把手掌的 6DoF 基座“钉”在规划好的空中位置
                # 这种方法虽然消除了基座速度，但对于测试抓取是有效的
                data.qpos[0:3] = target_palm_pos
                data.qpos[3:7] = target_palm_quat
                
                # 3. 物理步进 (计算接触力、摩擦力、手指运动)
                mujoco.mj_step(model, data)

            # C. 渲染同步
            viewer.sync()

            # 帧率控制
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()