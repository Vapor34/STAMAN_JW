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
            
            # 根据 Shadow Hand 标准命名规则分类 (根据你的实际XML修改)
            # J1, J2 是远端弯曲关节
            # J3 是掌指关节弯曲 (Knuckle Flexion) - 主要抓握动力
            # J4 是掌指关节侧摆 (Knuckle Abduction/Spread)
            # TH (大拇指) 通常有5个关节
            
            # 简化的启发式分类：
            jnt_name_lower = jnt_name.lower()
            if 'th' in jnt_name_lower: # 大拇指特殊处理
                thumb_indices.append(qpos_adr)
            elif 'j4' in jnt_name_lower or 'abd' in jnt_name_lower: # 侧摆
                abd_indices.append(qpos_adr)
            else: # 其他都算弯曲 (J1, J2, J3)
                flex_indices.append(qpos_adr)
                
    return flex_indices, abd_indices, thumb_indices

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
        # 2. 固定瓶子
        self.data.qpos[self.bottle_qpos_adr : self.bottle_qpos_adr + 7] = self.target_bottle_pos
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
        if np.random.random() > 0.5:
            # 随机探索 (维持现状)
            self.state[0:3] += np.random.normal(0, 0.02, 3)
        else:
            # 启发式：向瓶子方向挪动
            bottle_pos = self.target_bottle_pos[0:3]
            palm_pos = self.state[0:3]
            direction = bottle_pos - palm_pos
            # 挪动步长 1cm
            self.state[0:3] += (direction / (np.linalg.norm(direction) + 1e-6)) * 0.01

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
        self.state[8] = np.clip(self.state[8], -0.2, 0.2)

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
        palm_forward = -rot_mat[:, 1] # 假设 Z 轴是手心
        vec_to_bottle = bottle_pos - self.data.xpos[self.palm_body_id]
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
                if con.dist < -0.002: # 允许 2mm 接触
                    collision_penalty = 50.0

        return dist_score + orient_score + pose_energy + collision_penalty

def generate_new_grasp(planner, default_guess):
    """
    重采样逻辑：
    1. 给初始猜测加随机偏移（确保多样性）
    2. 运行退火算法
    3. 将结果写入 data.qpos
    """
    print("\n 正在检测到重置，生成新的随机位姿...")
    
    # 随机化起点：手掌位置在瓶子上方 10cm 范围内随机漂移
    random_guess = default_guess.copy()
    random_guess[0:3] += np.random.uniform(-0.08, 0.08, 3) 
    # 随机化初始协同张开度
    random_guess[7] = np.random.uniform(0.1, 0.5) 
    
    planner.state = random_guess
    best_pose, energy = planner.anneal()
    
    # 将规划好的位姿写入 MuJoCo 内存
    # 1. 设置 qpos (位置)
    planner.set_hand_pose(best_pose)
    
    apply_control_to_all(planner, best_pose)

    print(f"[完成] 新位姿能量值: {energy:.4f}")

    data.qpos[0:7] = best_pose[0:7]

def apply_control_to_all(planner, best_pose):
    # 提取协同变量
    grasp_val = np.clip(best_pose[7], 0.0, 1.0)
    spread_val = np.clip(best_pose[8], -0.2, 0.3)
    
    n_act = model.nu # 执行器总数
    for i in range(n_act):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not act_name: continue
        
        name_lower = act_name.lower()
        # 1. 处理手腕 (Wrist) - 保持固定或设定特定角度
        if 'wr' in name_lower or 'wrist' in name_lower:
            # 这里的控制值应对应 best_pose 中规划的手腕角度
            # 如果没规划，通常设为 0
            data.ctrl[i] = 0.0 
            
        # 2. 处理大拇指 (Thumb)
        elif 'th' in name_lower:
            data.ctrl[i] = grasp_val * 1.2
            
        # 3. 处理手指侧摆 (Spread/Abduction)
        elif 'j4' in name_lower or 'abd' in name_lower:
            data.ctrl[i] = spread_val
            
        # 4. 处理手指弯曲 (Flexion)
        else:
            data.ctrl[i] = grasp_val * 1.5

# --- 主程序 ---
try:
    model_path = 'shadow_hand/scene_left.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 计算 fingers joints 数量
    actual_finger_joints = model.nq - 14 
    

    # 现在的状态向量：7位手掌位姿 + 2位协同变量 = 9位
    initial_guess = np.zeros(9)

    # 手掌初始位置 (在瓶子上方)
    initial_guess[0:3] = [0.4, 0.4, 0.25] 
    initial_guess[3] = 1.0 # 初始旋转

    # 手指初始姿态：半张开 (协同值为 0.3)
    initial_guess[7] = 0.3 # Grasp synergy
    initial_guess[8] = 0.0 # Spread synergy

    planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')
    
    # 关键：足够的步数让退火算法寻优
    print("开始模拟退火规划，请耐心等待...")
    planner.steps = 2000 # 建议至少 2000-5000
    planner.Tmax = 10.0
    planner.Tmin = 0.01
    
    # t_start = time.time()
    # best_pose, energy = planner.anneal()
    # print(f"规划完成，耗时 {time.time()-t_start:.2f}s，最低能量: {energy:.4f}")

    # # 应用最佳位姿进行演示
    # planner.set_hand_pose(best_pose)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 记录一个标志位，防止在一瞬间（time=0时）重复触发多次规划
        has_planned_this_reset = False

        print("展示最佳采样位姿 (静态)。")
        while viewer.is_running():
            step_start = time.time()

            # 检测 Reset 信号
            if data.time < 1e-4: # 当点击 Reset 时，time 会变成 0
                print("检测到重置，正在重新生成位姿...")
                if not has_planned_this_reset:
                    generate_new_grasp(planner, initial_guess)
                    has_planned_this_reset = True
                    data.time = 0.0001
            else:
                # 当时间开始流动（哪怕只有一点点），重置标志位
                has_planned_this_reset = False


            # 保持静态显示。如果你想看“抓取动作”，这里可以运行 mj_step
            mujoco.mj_step(model, data) #已在scene_left.xml中将gravity设置为0
            
            viewer.sync()
            time.sleep(0.1)
            
            # 维持循环频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"发生错误：{e}")