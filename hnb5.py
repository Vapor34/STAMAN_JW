import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer
import argparse


from src.mujoco_utils import mujoco_load
from src.grasp_state import StateStruct

def get_tendon_actuator_map(model):
    """
    建立 腱名称 到 执行器控制索引 的映射表
    tendon_to_actuator={腱名称: 执行器控制索引}
    """
    tendon_to_actuator = {}
    
    for act_id in range(model.nu):
        # 判断执行器是否作用于腱 (mjTRN_TENDON = 3)
        if model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_TENDON:
            # 获取该执行器关联的 Tendon ID
            t_id = model.actuator_trnid[act_id, 0]
            # 获取腱的名字
            t_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, t_id)
            # 记录映射
            tendon_to_actuator[t_name] = act_id
            
    return tendon_to_actuator



def control_tendon_actuators(model, data, ctrl_value):
    tendon_to_actuator = get_tendon_actuator_map(model)
    
    # 将腱执行器的控制值设置到 data.ctrl
    for tendon_name, actuator_idx in tendon_to_actuator.items():
        data.ctrl[actuator_idx] = ctrl_value





def get_synergy_mapping(model, hand_prefix):
    """
    【手动分类】根据Shadow Hand实际配置手动列出所有关节并分类。
    
    执行器配置（来自 hand_joint_test.py）：
    ✓ 可控关节（有执行器）:
      - 食指 J3 (lh_FFJ3)、中指 J3 (lh_MFJ3)、无名指 J3 (lh_RFJ3)、小指 J3 (lh_LFJ3)
      - 食指 J4 (lh_FFJ4)、中指 J4 (lh_MFJ4)、无名指 J4 (lh_RFJ4)、小指 J4 (lh_LFJ4)
      - 大拇指所有关节 (THJ1-5)
      - 手腕 (WRJ1-2)
    
    ✗ 被动关节（无执行器）:
      - 四指 J1、J2 (被忽略)
      - 小指 J5 (lh_LFJ5) (被忽略)
    """
    flex_indices = [] 
    abd_indices = [] 
    thumb_indices = []
    tendon_indices = []

    jnt_map = {} # 关节名称 → qpos地址
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name:
            jnt_map[jnt_name] = model.jnt_qposadr[i]



    # 【四指弯曲】仅 J3（有执行器的远端关节）
    flex_j3_names = ['lh_FFJ3', 'lh_MFJ3', 'lh_RFJ3', 'lh_LFJ3']
    for jnt_name in flex_j3_names:
        if jnt_name in jnt_map:
            qpos_adr = jnt_map[jnt_name]
            flex_indices.append(qpos_adr)
    
    # 【侧摆】J4（四指的侧摆关节）
    abd_j4_names = ['lh_FFJ4', 'lh_MFJ4', 'lh_RFJ4', 'lh_LFJ4']
    for jnt_name in abd_j4_names:
        if jnt_name in jnt_map:
            qpos_adr = jnt_map[jnt_name]
            abd_indices.append(qpos_adr)
    
    # 【大拇指】所有关节
    thumb_names = ['lh_THJ1', 'lh_THJ2', 'lh_THJ3', 'lh_THJ4', 'lh_THJ5']
    for jnt_name in thumb_names:
        if jnt_name in jnt_map:
            qpos_adr = jnt_map[jnt_name]
            thumb_indices.append(qpos_adr)
    
    # 【被忽略的关节】被动关节（无执行器）
    ignored_names = [
        'lh_FFJ1', 'lh_FFJ2',  # 食指
        'lh_MFJ1', 'lh_MFJ2',  # 中指
        'lh_RFJ1', 'lh_RFJ2',  # 无名指
        'lh_LFJ1', 'lh_LFJ2', 'lh_LFJ5',  # 小指
        'lh_WRJ1', 'lh_WRJ2'   # 手腕（可以添加如需要）
    ]
    for jnt_name in ignored_names:
        if jnt_name in jnt_map:
            qpos_adr = jnt_map[jnt_name]
    
    # tendon_names = ["lh_FFJ0", "lh_MFJ0", "lh_RFJ0", "lh_LFJ0"]
    # for jnt_name in tendon_names:
    #     if jnt_name in jnt_map:
    #         qpos_adr = jnt_map[jnt_name]
    #         tendon_indices.append(qpos_adr)
    #         print(f"tendon inds: {qpos_adr}")

    
    
    return flex_indices, abd_indices, thumb_indices

def qpos_to_ctrl_improved(model, data, planner, target_pose):
    """
    【正确版本】协同变量 → 目标关节角度（位置伺服）
    
    【关键认识】Shadow Hand使用 <position> 执行器（位置伺服），不是 <motor>
    • 控制输入语义：目标关节角度（rad），在 ctrlrange 范围内
    • 执行器内部：自动使用PID将目标角度转换为力矩
    • 错误做法：把 grasp_val 当做力度乘数（这是 <motor> 的方式）
    
    【正确做法】
    • 定义每个关节类型的最大摆动角度
    • grasp_val [0,1] 线性映射到 [0, max_angle]
    • 位置伺服自动处理动力学

    target_pose: 9D数组 [x, y, z, qw, qx, qy, qz, grasp, spread]
    """
    #### grasping encoding: (x y z qw qx qy qz alpha beta) 
    ctrl_cmd = np.zeros(model.nu)
    grasp_val = np.clip(target_pose[7], 0.0, 1.0) # tanh sigmoid 归一化函数
    spread_val = np.clip(target_pose[8], -0.2, 0.3)

    # print(f"======= grasp value: {grasp_val}")
    
    # 【参数】每类关节的最大控制角度
    # 这些值应该在各自的 ctrlrange 范围内
    flex_max_angle = 1.571        # J3 弯曲最大角度 (rad)，ctrlrange=[-0.262, 1.571]
    abd_max_angle = 0.349         # J4 侧摆最大角度 (rad)，ctrlrange=[-0.349, 0.349]
    thumb_max_angles = [1.0472, 1.22173, 0.20944, 0.698132, 1.5708] # 各拇指关节的最大角度
    tendon_max_angle = 3.14

    tendon_actuator_map = get_tendon_actuator_map(model)

    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_adr = model.jnt_qposadr[jnt_id]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        
        # 获取该执行器的物理限制
        ctrl_range = model.actuator_ctrlrange[i]
        
        if jnt_adr in planner.flex_adrs:
            # 【四指弯曲】grasp_val [0,1] → 目标角度 [0, flex_max_angle]
            val = grasp_val * flex_max_angle
                
        elif jnt_adr in planner.thumb_adrs:
            # 【大拇指】根据拇指内索引选择对应的最大角度
            try:
                thumb_idx = planner.thumb_adrs.index(jnt_adr)
                max_angle = thumb_max_angles[thumb_idx] if thumb_idx < len(thumb_max_angles) else 0.8
            except (ValueError, IndexError):
                max_angle = 0.8
            val = grasp_val * max_angle
            
        elif jnt_adr in planner.abd_adrs:
            # 【四指侧摆】spread_val 已经是角度偏移
            val = spread_val
        # elif jnt_adr in planner.tendon_adrs:
        #     val = 1.0
        #     pass
            
        else:
            val = 0.0

        ctrl_cmd[i] = np.clip(val, ctrl_range[0], ctrl_range[1])

    for td_name, actuator_idx in tendon_actuator_map.items():
        ctrl_cmd[actuator_idx] = grasp_val * tendon_max_angle               

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
        self.contact_body_ids = []

        excluded_keywords = ['wrist', 'forearm']

        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                self.hand_geom_ids.extend(self._get_geoms_id_of_body(i))
                
                name_lower = name.lower()
                is_excluded = any(k in name_lower for k in excluded_keywords)
                
                if not is_excluded:
                    self.contact_body_ids.append(i)
                    print(f"Distance calculation includes: {name}")

        # 获取关节映射
        self.flex_adrs, self.abd_adrs, self.thumb_adrs = get_synergy_mapping(model, hand_body_prefix)
        
        super(GraspPlanner, self).__init__(state)

    def _get_geoms_id_of_body(self, body_id):
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]

    def set_hand_pose(self, state_struct):
        """将规划状态转换为MuJoCo的运动学配置
        
        这个函数用于能量计算中的前向运动学评估。
        它将9D规划状态（位置、姿态、协同变量）映射到手部的24个关节配置。
        
        映射关系：
        ├─ 状态位置 (3D)  → qpos[0:3]    手掌XYZ坐标
        ├─ 状态姿态 (4D)  → qpos[3:7]    手掌四元数
        ├─ grasp_synergy  → 四指J3+大拇指  (弯曲关节)
        ├─ spread_synergy → 四指J4        (侧摆关节)
        └─ 被动关节       → 自动耦合       (J1、J2通过腱系统耦联)
        
        Args:
            state_struct (StateStruct): 9D规划状态
                - position: 手掌位置 [x, y, z]
                - quaternion: 手掌姿态 [qw, qx, qy, qz]
                - grasp: 抓取强度 [0, 1]（0=完全张开，1=完全闭合）
                - spread: 展开程度 [-0.2, 0.3]（-0.2=最收紧，0.3=最展开）
        """
        # ===== 参数定义 =====
        # 这些系数将协同变量映射到关节角度
        FLEX_MAX_ANGLE = 1.5       # 四指J3最大弯曲角度 (rad)
        ABD_SCALE = 0.5            # 四指J4侧摆缩放因子
        THUMB_FLEX_ANGLE = 1.2     # 大拇指弯曲最大角度 (rad)
        THUMB_PROGRESSION = 0.2    # 各拇指关节的递进比例
        
        # 重置数据以获得干净的初始状态
        mujoco.mj_resetData(self.model, self.data)
        
        # ===== 1. 设置手掌基本位置和姿态 =====
        self.data.qpos[0:3] = state_struct.get_position()
        self.data.qpos[3:7] = state_struct.get_quaternion()
        
        # ===== 2. 根据协同变量设置手指关节 =====
        grasp = state_struct.grasp      # [0, 1]
        spread = state_struct.spread    # [-0.2, 0.3]
        
        # 【四指弯曲 J3】由 grasp_synergy 控制
        # grasp=0 → 0 rad (完全张开)
        # grasp=1 → FLEX_MAX_ANGLE rad (完全闭合)
        flex_angle = grasp * FLEX_MAX_ANGLE
        for joint_addr in self.flex_adrs:
            self.data.qpos[joint_addr] = flex_angle
        
        # 【四指侧摆 J4】由 spread_synergy 控制
        # spread=-0.2 → 最收紧
        # spread=+0.3 → 最展开
        # 乘以缩放因子将范围映射到物理范围
        spread_angle = spread * ABD_SCALE
        for joint_addr in self.abd_adrs:
            self.data.qpos[joint_addr] = spread_angle
        
        # 【大拇指】由 grasp_synergy 控制，但各关节角度递进
        # 这使得大拇指弯曲时能自然地形成对立姿态
        thumb_flex = grasp * THUMB_FLEX_ANGLE
        for i, joint_addr in enumerate(self.thumb_adrs):
            # 使用递进系数：关节0=0.5倍，关节1=0.7倍，关节2=0.9倍，等等
            progression = 0.5 + i * THUMB_PROGRESSION
            self.data.qpos[joint_addr] = thumb_flex * progression
        
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
            palm_pos = self.data.xpos[self.palm_body_id]
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

        # 展开程度扰动
        current.spread += np.random.normal(0, 0.05)
        current.spread = np.clip(current.spread, -0.2, 0.3)
        
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
        GRASP_TARGET = 0.4        # 目标抓取力度（偏好值）
        COLLISION_THRESHOLD = -0.005  # 碰撞判定阈值(m)，<0为穿透
        
        # 将9D数组转换为StateStruct
        state_struct = StateStruct()
        state_struct.from_array(self.state)
        self.set_hand_pose(state_struct)
        
        # ===== 1. 接近性能量：手指到物体的平均距离 =====
        bottle_pos = self.data.xpos[self.bottle_body_id]
        total_dist = 0.0
        for b_id in self.contact_body_ids:
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
        # 目标grasp=0.4：既能有效接触，又避免过度闭合
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="shadow_hand/scene_left.xml", help='mujoco models path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model, data = mujoco_load(args.model_path)

    # 初始状态 - 使用 StateStruct
    initial_state = StateStruct(
        position=[0.4, 0.4, 0.3],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        grasp=0.0,
        spread=0.0
    )
    initial_guess = initial_state.to_array()

    planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')
    planner.steps = 2500

    # 动画控制变量
    planning_done = False
    target_state = initial_state.copy()  # 规划得到的目标状态
    current_grasp_val = 0.0
    execution_start_time = 0.0

    # for i in range(model.njnt):
    #     name = model.joint(i).name
    #     adr = model.joint(i).qposadr
    #     dof = model.joint(i).dofadr
    #     print(f"joint {i:2d}: {name:20s} qpos[{adr}]  dof[{dof}]")


    # print(">>> 启动 MuJoCo 查看器...")
    # for i in range(model.nu):
    #     name = model.actuator(i).name
    #     print(f"ctrl[{i:2d}] -> actuator: {name}")
    # exit()

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 规划阶段
            if data.time < 0.002:
                print("\n[状态] 正在规划最佳抓取点...")
                
                # 从初始状态开始扰动
                planner.state = initial_guess.copy()
                planner.state[0:3] += np.random.uniform(-0.05, 0.05, 3)
                best_pose, energy = planner.anneal()
                # best_pose[7] = 1.0
                
                # 将规划结果加载到 StateStruct
                target_state.from_array(best_pose)
                target_state.grasp = 0.8
                print(f"[完成] 目标 Grasp: {target_state.grasp:.2f}, Spread: {target_state.spread:.2f}")
                print(f"[完成] 目标位置: ({target_state.x:.3f}, {target_state.y:.3f}, {target_state.z:.3f})")
                
                current_grasp_val = 0.0
                execution_start_time = time.time()
                planning_done = True
                
                data.time = 0.002

            # 执行阶段
            if planning_done:
                anim_time = time.time() - execution_start_time
                
                # 0.0s ~ 1.0s：手掌瞬移
                data.qpos[0:3] = target_state.get_position()
                data.qpos[3:7] = target_state.get_quaternion()
                
                # 1.0s ~ 3.0s：手指闭合
                grasp_start_time = 1.0
                grasp_duration = 2.0
                
                if anim_time > grasp_start_time:
                    grasp_progress = (anim_time - grasp_start_time) / grasp_duration
                    grasp_progress = np.clip(grasp_progress, 0.0, 1.0)
                    current_grasp_val = target_state.grasp * grasp_progress
                else:
                    current_grasp_val = 0.0

                # 构造当前执行状态用于计算控制命令
                exec_state = StateStruct(
                    position=target_state.get_position(),
                    quaternion=target_state.get_quaternion(),
                    grasp=current_grasp_val,
                    spread=target_state.spread
                )
                
                # 设置关节执行器的控制信号
                joint_ctrl = qpos_to_ctrl_improved(model, data, planner, exec_state.to_array())
                data.ctrl = joint_ctrl
                
                # 设置腱执行器的控制信号（覆盖腱对应的 ctrl 索引，实现 J1、J2 的被动耦联）
                # control_tendon_actuators(model, data, 2.0)
                
                mujoco.mj_step(model, data)

            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
