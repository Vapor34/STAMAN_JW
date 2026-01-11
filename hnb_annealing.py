import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer


def get_synergy_mapping(model, hand_prefix):
    """
    创建一个简化的映射,将2个协同变量映射到所有手指关节。
    返回两个列表: flex_indices (负责弯曲的关节索引), abd_indices (负责侧摆的关节索引)
    """
    flex_indices = [] # 弯曲关节 (Flexion/Curl)
    abd_indices = []  # 侧摆关节 (Abduction/Spread)
    thumb_indices = [] # 大拇指关节

    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
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

def qpos_to_ctrl(model, data, planner, target_pose):
    """
    将规划出的 9维 状态向量转换为具体的电机控制信号 (ctrl)。
    假设 XML 中的 Actuator 是 Position 类型 (servo)。
    """
    ctrl_cmd = np.zeros(model.nu)
    
    # 解析 target_pose
    grasp_synergy = np.clip(target_pose[7], 0.0, 1.0)
    spread_synergy = np.clip(target_pose[8], -0.2, 0.3)
    
    # 稍微增加一点抓握力度 (Over-closing)，确保抓紧
    # 比如规划出 0.8，我们命令电机去 0.9，利用物理接触停在表面
    grasp_synergy_cmd = min(grasp_synergy * 1.1, 1.0) 

    # 遍历所有 Actuator
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not act_name: continue
        
        # 找到该 Actuator 对应的 Joint
        # (MuJoCo 中 actuator 对应的关节 ID 存储在 trnid 中)
        jnt_id = model.actuator_trnid[i, 0] 
        jnt_qpos_adr = model.jnt_qposadr[jnt_id]
        
        # 判断这个关节属于哪类
        target_angle = 0.0
        
        if jnt_qpos_adr in planner.thumb_adrs:
            # 大拇指
            # 这里需要根据你的 synergy 逻辑反推角度
            # 简单起见，假设大拇指最大弯曲 1.5 rad
            target_angle = grasp_synergy_cmd * 1.5
            if "thdistal" in act_name.lower() or "thmiddle" in act_name.lower():
                 target_angle *= 1.2 # 指尖弯多点
            
        elif jnt_qpos_adr in planner.abd_adrs:
            # 侧摆
            target_angle = spread_synergy
            
        elif jnt_qpos_adr in planner.flex_adrs:
            # 四指弯曲
            # 同样假设最大弯曲 1.8 rad
            target_angle = grasp_synergy_cmd * 1.8
            
        # 赋值给 ctrl
        ctrl_cmd[i] = target_angle
        
    return ctrl_cmd

class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name, hand_body_prefix='lh_'):
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix

        # 1. 瓶子信息与固定位置
        self.bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body_name)
        bottle_jnt_id = model.body_jntadr[self.bottle_body_id]
        self.bottle_qpos_adr = model.jnt_qposadr[bottle_jnt_id]
        self.target_bottle_pos = np.array([0.4, 0.4, 0.1, 1, 0, 0, 0]) 
        self.bottle_geom_ids = self._get_geoms_of_body(self.bottle_body_id)

        # 2. 手部信息
        self.palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.hand_prefix}palm")
        self.hand_geom_ids = []
        self.finger_tip_body_ids = []
        self.hand_joint_qpos_adrs = []
        
        tip_suffixes = ['ffdistal', 'mfdistal', 'rfdistal', 'lfdistal', 'thdistal']
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                self.hand_geom_ids.extend(self._get_geoms_of_body(i))
                if any(suffix in name for suffix in tip_suffixes):
                    self.finger_tip_body_ids.append(i)

        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and self.hand_prefix in name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.hand_joint_qpos_adrs.append(model.jnt_qposadr[i])

        self.flex_adrs, self.abd_adrs, self.thumb_adrs = get_synergy_mapping(model, hand_body_prefix)
        print(f"协同映射: 弯曲关节数={len(self.flex_adrs)}, 侧摆关节数={len(self.abd_adrs)}, 大拇指关节数={len(self.thumb_adrs)}")


        super(GraspPlanner, self).__init__(state)

    def _get_geoms_of_body(self, body_id):
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]
    

    def set_hand_pose(self, state):
        mujoco.mj_resetData(self.model, self.data)
        # 1. 设置手掌
        self.data.qpos[0:7] = state[0:7]
        # # 2. 固定瓶子
        # self.data.qpos[self.bottle_qpos_adr : self.bottle_qpos_adr + 7] = self.target_bottle_pos
        # --- 修改部分：应用协同控制 ---
        # state[7] 控制抓握力度 (0~1)
        grasp_synergy = np.clip(state[7], 0.0, 1.0)
        # state[8] 控制张开程度 (-0.2 ~ 0.2)
        spread_synergy = np.clip(state[8], -0.2, 0.3)

        # A. 设置四指弯曲 (J3弯曲多一点，J1/J2少一点，这里简化为统一比例)
        # 假设最大弯曲角度为 1.5 弧度
        flex_angle = grasp_synergy * 1.5
        for adr in self.flex_adrs:
             # 简单的比例控制，远端关节弯曲得比近端少一些会更自然，这里先统一设置
            self.data.qpos[adr] = flex_angle

        # B. 设置四指侧摆 (Spread)
        # 食指往负方向摆，小指往正方向摆
        # 这里需要根据关节具体名称来精细控制，简化的做法是给一个微小的随机扰动或统一值
        # 为简单起见，暂且让侧摆关节保持在0附近，主要靠手掌调整
        for adr in self.abd_adrs:
            self.data.qpos[adr] = spread_synergy * 0.5 # 侧摆角度小一点

        # C. 设置大拇指 (大拇指需要特殊的对掌动作)
        # 简化处理：让大拇指随着抓握协同一起向手心弯曲
        thumb_flex = grasp_synergy * 1.2
        # 大拇指的第一个关节通常是侧摆/对掌，第二个是弯曲
        for i, adr in enumerate(self.thumb_adrs):
            # 这是一个非常粗糙的映射，让大拇指关节按不同比例弯曲
            self.data.qpos[adr] = thumb_flex * (0.5 + i * 0.2) 
            
        # --- 修改结束 ---

        mujoco.mj_forward(self.model, self.data)

    #指尖到瓶子距离之和
    def calculate_distance_to_bottle(self):
        bottle_pos = self.data.xpos[self.bottle_body_id]
        total_dist = 0
        for b_id in self.finger_tip_body_ids:
            tip_pos = self.data.xpos[b_id]
            total_dist += np.linalg.norm(tip_pos - bottle_pos)
        return total_dist / len(self.finger_tip_body_ids)

    def check_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            is_bottle = (contact.geom1 in self.bottle_geom_ids or contact.geom2 in self.bottle_geom_ids)
            is_hand = (contact.geom1 in self.hand_geom_ids or contact.geom2 in self.hand_geom_ids)
            if is_bottle and is_hand:
                # 稍微允许一点穿模，利于寻找紧密接触点
                if contact.dist < -0.005: 
                    return True
        return False

    def move(self):
        """ 启发式扰动：加入向瓶子靠近的‘引力’ """
        # 50% 概率进行随机探索，50% 概率进行启发式靠近
        if np.random.random() > 0.6:
            # 随机探索 (维持现状)
            self.state[0:3] += np.random.normal(0, 0.02, 3)
        else:
            # 启发式：向瓶子方向挪动
            bottle_pos = self.target_bottle_pos[0:3]
            palm_pos = self.state[0:3]
            direction = bottle_pos - palm_pos
            # 挪动步长 1cm
            self.state[0:3] += (direction / (np.linalg.norm(direction) + 1e-6)) * 0.001

        # 旋转扰动 (微调)
        self.state[3:7] += np.random.normal(0, 0.1, 4)
        self.state[3:7] /= np.linalg.norm(self.state[3:7])
        
        # --- 修改部分：手指扰动 ---
        # 现在 state[7:] 只有两个元素了
        
        # 扰动抓握协同 (state[7])
        self.state[7] += np.random.normal(0, 0.1)
        # 限制在 [0, 1] 范围内，0是张开，1是握紧
        self.state[7] = np.clip(self.state[7], 0.0, 0.8) # 0.8 不要握太死

        # 扰动张开协同 (state[8])
        self.state[8] += np.random.normal(0, 0.05)
        self.state[8] = np.clip(self.state[8], 0, 0.2)

    def energy(self):
        self.set_hand_pose(self.state)
        
        # 1. 距离项 (指尖到瓶子)
        dist = 0
        bottle_pos = self.data.xpos[self.bottle_body_id]
        for b_id in self.finger_tip_body_ids:
            dist += np.linalg.norm(self.data.xpos[b_id] - bottle_pos)
        dist_score = dist / len(self.finger_tip_body_ids)

        # 2. 朝向项 (手心对准瓶子)
        rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_forward = -rot_mat[:, 1] # -Y轴是手心
        #lh_palm实际上处于掌根位置，palm_central记录实际掌心位置用于计算
        palm_central = self.data.xpos[self.palm_body_id]
        palm_central[1] = palm_central[1] - 0.03
        vec_to_bottle = bottle_pos - palm_central
        vec_to_bottle /= np.linalg.norm(vec_to_bottle)
        orient_score = (1.0 - np.dot(palm_forward, vec_to_bottle)) * 2.0

        # 3. 姿态惩罚 (奖励张开的手掌，惩罚握得太紧或太乱)
        # 假设关节在 0.4 弧度左右是比较好的预抓取姿态
        # pose_penalty = np.mean(np.square(self.state[7:] - 0.4)) * 0.5

        # 新增：奖励 "半张开" 的预抓取姿态 (grasp_synergy 在 0.3 左右)
        # 这样手既不是全张开也不是全握紧，最适合去包络物体
        grasp_synergy_val = self.state[7]
        # 如果协同值偏离 0.3，能量升高
        pose_energy = np.square(grasp_synergy_val - 0.3) * 1.0

        # 4. 碰撞惩罚
        collision_penalty = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids) and \
               (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids):
                if con.dist < 0.002: #  2mm Safety margin
                    collision_penalty = 50.0

        return dist_score + orient_score + pose_energy + collision_penalty

# def generate_new_grasp(model, data, planner, default_guess):
#     """
#     重采样逻辑：
#     1. 给初始猜测加随机偏移（确保多样性）
#     2. 运行退火算法
#     3. 将结果写入 data.qpos
#     """
#     print("\n 正在检测到重置，生成新的随机位姿...")
    
#     # 随机化起点：手掌位置在瓶子上方 10cm 范围内随机漂移
#     random_guess = default_guess.copy()
#     random_guess[0:3] += np.random.uniform(-0.08, 0.08, 3) 
#     # 随机化初始协同张开度
#     random_guess[7] = np.random.uniform(0.1, 0.5) 
    
#     planner.state = random_guess
#     best_pose, energy = planner.anneal()
    
#     # 将规划好的位姿写入 MuJoCo 内存
#     # 1. 设置 qpos (位置)
#     planner.set_hand_pose(best_pose)
    
#     apply_control_to_all(model, data, planner, best_pose)

#     print(f"[完成] 新位姿能量值: {energy:.4f}")

#     data.qpos[0:7] = best_pose[0:7]

# def apply_control_to_all(model, data, planner, best_pose):
#     # 提取协同变量
#     grasp_val = np.clip(best_pose[7], 0.0, 1.0)
#     spread_val = np.clip(best_pose[8], -0.2, 0.3)
    
#     n_act = model.nu # 执行器总数
#     for i in range(n_act):
#         act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#         if not act_name: continue
        
#         name_lower = act_name.lower()
#         # 1. 处理手腕 (Wrist) - 保持固定或设定特定角度
#         if 'wr' in name_lower or 'wrist' in name_lower:
#             # 这里的控制值应对应 best_pose 中规划的手腕角度
#             # 如果没规划，通常设为 0
#             data.ctrl[i] = 0.0 
            
#         # 2. 处理大拇指 (Thumb)
#         elif 'th' in name_lower:
#             data.ctrl[i] = grasp_val * 1.2
            
#         # 3. 处理手指侧摆 (Spread/Abduction)
#         elif 'j4' in name_lower or 'abd' in name_lower:
#             data.ctrl[i] = spread_val
            
#         # 4. 处理手指弯曲 (Flexion)
#         else:
#             data.ctrl[i] = grasp_val * 1.5

# --- 主程序 ---

def main():
    model_path = 'shadow_hand/scene_left.xml' # 请确保路径正确
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except:
        print("未找到模型文件，请检查路径。")
        return
        
    data = mujoco.MjData(model)

    # 1. 初始化规划器
    initial_guess = np.zeros(9)
    initial_guess[0:3] = [0.4, 0.4, 0.3] # 起始高一点
    initial_guess[3] = 1.0 
    initial_guess[7] = 0.5 # 初始半握
    
    planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')
    planner.steps = 2000

    # 用于存储当前的控制目标
    current_ctrl_target = np.zeros(model.nu)
    # 用于存储目标手掌位置（浮动基座）
    target_palm_pos = initial_guess[0:3].copy()
    target_palm_quat = initial_guess[3:7].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(">>> 准备就绪。请按空格键开始(如果viewer支持) 或 等待自动开始...")
        
        # 标志位
        planning_done = False
        
        while viewer.is_running():
            step_start = time.time()

            # --- 状态机逻辑 ---
            
            # 阶段 A: 如果刚重置 (time close to 0)，执行规划
            if data.time < 0.002:
                print(">>> [规划阶段] 正在计算最佳抓取位姿 (模拟退火)...")
                # 恢复初始猜测
                planner.state = initial_guess.copy()
                # 运行退火 (纯计算，不渲染)
                best_pose, energy = planner.anneal()
                print(f">>> [规划完成] 能量: {energy:.4f}")
                print(f"    目标协同值: {best_pose[7]:.2f}")
                
                # 计算出这个 pose 对应的电机指令
                current_ctrl_target = qpos_to_ctrl(model, data, planner, best_pose)
                target_palm_pos = best_pose[0:3]
                target_palm_quat = best_pose[3:7]
                
                # 策略选择：
                # 选项1 (瞬移)：直接把手设置过去 (用于调试规划结果)
                # planner.set_hand_pose(best_pose) 
                # data.qpos[0:7] = best_pose[0:7] # 浮动基座瞬移
                
                # 选项2 (控制)：把手瞬移到目标位置，但手指张开，然后慢慢合拢 (更真实)
                # 这里我们简化：先瞬移基座到目标位置，然后通过 ctrl 驱动手指
                data.qpos[0:3] = target_palm_pos
                data.qpos[3:7] = target_palm_quat
                
                # 必须调用 forward 更新瞬移后的几何
                mujoco.mj_forward(model, data) 
                
                planning_done = True

            # 阶段 B: 物理执行循环
            if planning_done:
                # 1. 应用手指控制信号
                # Shadow Hand 的 XML 通常是 Position Control
                # data.ctrl[:] = current_ctrl_target 
                # 为了防止某个电机报错，逐个赋值更安全
                for i in range(len(current_ctrl_target)):
                    data.ctrl[i] = current_ctrl_target[i]

                # 2. 浮动基座控制 (如果不固定手腕)
                # 如果你的 scene_left.xml 里手腕是 free joint，它会掉下来。
                # 你需要把 data.qpos[0:7] 锁死在规划的位置，或者添加 mocap body 绑定。
                # 这里最简单的做法是：每一帧都强制重置基座位置 (即“钉在空中”)
                data.qpos[0:3] = target_palm_pos
                data.qpos[3:7] = target_palm_quat
                # 注意：强行修改 qpos 会消除基座的速度，这虽然不物理，但能让手悬空。
                # 更好的做法是在 XML 里把手腕设为 mocap body。

                # 3. 物理步进
                mujoco.mj_step(model, data)

            viewer.sync()

            # 帧率控制
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()