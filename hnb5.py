import mujoco
import mujoco.viewer
import numpy as np
import time
from simanneal import Annealer

# class StateStruct:
#     def __init__(self):
#         self.data[9]
#         xxx
        #   self.quat = [xxx]
    

# --- 辅助函数 ---

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
    
    # 构建关节名称 → qpos_adr 的映射
    jnt_map = {}
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name:
            jnt_map[jnt_name] = model.jnt_qposadr[i]
    
    print("\n=== 手部关节分类 ===")
    
    # 【四指弯曲】仅 J3（有执行器的远端关节）
    print("\n【四指弯曲 - J3 远端关节】:")
    flex_j3_names = ['lh_FFJ3', 'lh_MFJ3', 'lh_RFJ3', 'lh_LFJ3']
    for jnt_name in flex_j3_names:
        if jnt_name in jnt_map:
            qpos_adr = jnt_map[jnt_name]
            flex_indices.append(qpos_adr)
            print(f"  ✓ {jnt_name:15s} → qpos_adr {qpos_adr}")
    
    # 【侧摆】J4（四指的侧摆关节）
    print("\n【四指侧摆 - J4 关节】:")
    abd_j4_names = ['lh_FFJ4', 'lh_MFJ4', 'lh_RFJ4', 'lh_LFJ4']
    for jnt_name in abd_j4_names:
        if jnt_name in jnt_map:
            qpos_adr = jnt_map[jnt_name]
            abd_indices.append(qpos_adr)
            print(f"  ✓ {jnt_name:15s} → qpos_adr {qpos_adr}")
    
    # 【大拇指】所有关节
    print("\n【大拇指 - 所有关节】:")
    thumb_names = ['lh_THJ1', 'lh_THJ2', 'lh_THJ3', 'lh_THJ4', 'lh_THJ5']
    for jnt_name in thumb_names:
        if jnt_name in jnt_map:
            qpos_adr = jnt_map[jnt_name]
            thumb_indices.append(qpos_adr)
            print(f"  ✓ {jnt_name:15s} → qpos_adr {qpos_adr}")
    
    # 【被忽略的关节】被动关节（无执行器）
    print("\n【被忽略的关节 - 被动关节（无执行器）】:")
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
            print(f"  ✗ {jnt_name:15s} → qpos_adr {qpos_adr} (被动/不使用)")
    
    print(f"\n总结: flex={len(flex_indices)} J3, abd={len(abd_indices)} J4, thumb={len(thumb_indices)}")
    print("=" * 50)
    
    return flex_indices, abd_indices, thumb_indices

def qpos_to_ctrl_improved(model, planner, target_pose):
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
    """
    ctrl_cmd = np.zeros(model.nu)
    grasp_val = np.clip(target_pose[7], 0.0, 1.0)
    spread_val = np.clip(target_pose[8], -0.2, 0.3)
    
    # 【参数】每类关节的最大控制角度
    # 这些值应该在各自的 ctrlrange 范围内
    flex_max_angle = 1.571        # J3 弯曲最大角度 (rad)，ctrlrange=[-0.262, 1.571]
    abd_max_angle = 0.349         # J4 侧摆最大角度 (rad)，ctrlrange=[-0.349, 0.349]
    thumb_max_angles = [1.0472, 1.22173, 0.20944, 0.698132, 1.5708] # 各拇指关节的最大角度

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
            
        else:
            val = 0.0
            
        # 严格按执行器的 ctrlrange 进行裁剪
        # 位置伺服会将这个值作为目标角度
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

    def set_hand_pose(self, state):
        """仅用于能量计算的运动学设置"""
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[0:7] = state[0:7]
        
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

        mujoco.mj_forward(self.model, self.data)

    def move(self):
        """模拟退火的扰动函数"""
        if np.random.random() > 0.7:
            self.state[0:3] += np.random.normal(0, 0.015, 3)#高斯，均值0,标准差0.015，size=3
        else:
            bottle_pos = self.data.xpos[self.bottle_body_id]
            palm_pos = self.data.xpos[self.palm_body_id]
            direction = bottle_pos - palm_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                self.state[0:3] += (direction / dist) * 0.002

        self.state[3:7] += np.random.normal(0, 0.05, 4)
        self.state[3:7] /= np.linalg.norm(self.state[3:7])
        
        self.state[7] += np.random.normal(0, 0.08)
        self.state[7] = np.clip(self.state[7], 0.1, 0.9)

        self.state[8] += np.random.normal(0, 0.05)
        self.state[8] = np.clip(self.state[8], -0.1, 0.2)

    def energy(self):
        """ 带有环境约束和关节限位惩罚的能量函数 """
        self.set_hand_pose(self.state)
        
        # 距离项
        bottle_pos = self.data.xpos[self.bottle_body_id]
        total_dist = 0.0
        for b_id in self.contact_body_ids:
            dist = np.linalg.norm(self.data.xpos[b_id] - bottle_pos)
            total_dist += dist
        avg_dist = total_dist / len(self.contact_body_ids)

        # 朝向项
        rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_normal = -rot_mat[:, 1]
        vec_to_bottle = bottle_pos - self.data.xpos[self.palm_body_id]
        vec_to_bottle_norm = vec_to_bottle / (np.linalg.norm(vec_to_bottle) + 1e-6)
        orient_energy = (1.0 - np.dot(palm_normal, vec_to_bottle_norm)) * 1.5 

        # 碰撞惩罚
        collision_penalty = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
            is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
            if is_bottle and is_hand:
                if con.dist < -0.005:
                    collision_penalty += 200.0
            if (con.geom1 == self.floor_geom_id or con.geom2 == self.floor_geom_id) and is_hand:
                collision_penalty += 500.0

        # 关节限位惩罚
        limit_penalty = 0.0
        for i in range(1, self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name and self.hand_prefix in jnt_name:
                q_addr = self.model.jnt_qposadr[i]
                q_val = self.data.qpos[q_addr]
                low, high = self.model.jnt_range[i]
                
                if q_val < low:
                    limit_penalty += np.square(low - q_val)
                elif q_val > high:
                    limit_penalty += np.square(q_val - high)
        
        limit_weight = 1000.0 

        # 姿态先验
        pose_energy = np.square(self.state[7] - 0.4) 

        return (avg_dist * 25.0 + 
                orient_energy * 10 + 
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
    initial_guess[7] = 0.0

    planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')
    planner.steps = 2500

    # 动画控制变量
    planning_done = False
    
    target_palm_pos = initial_guess[0:3].copy()
    target_palm_quat = initial_guess[3:7].copy()
    final_grasp_synergy = 0.0
    final_spread_synergy = 0.0
    
    current_grasp_val = 0.0
    execution_start_time = 0.0

    for i in range(model.njnt):
        name = model.joint(i).name
        adr = model.joint(i).qposadr
        dof = model.joint(i).dofadr
        print(f"joint {i:2d}: {name:20s} qpos[{adr}]  dof[{dof}]")


    print(">>> 启动 MuJoCo 查看器...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 规划阶段
            if data.time < 0.002:
                print("\n[状态] 正在规划最佳抓取点...")
                
                planner.state = initial_guess.copy()
                planner.state[0:3] += np.random.uniform(-0.05, 0.05, 3)
                best_pose, energy = planner.anneal()
                
                print(f"[完成] 目标 Grasp: {best_pose[7]:.2f}")

                target_palm_pos = best_pose[0:3]
                target_palm_quat = best_pose[3:7]
                final_grasp_synergy = best_pose[7]
                final_spread_synergy = best_pose[8]
                
                current_grasp_val = 0.0
                execution_start_time = time.time()
                planning_done = True
                
                data.time = 0.002

            # 执行阶段
            if planning_done:
                anim_time = time.time() - execution_start_time
                
                # 0.0s ~ 1.0s：手掌瞬移
                data.qpos[0:3] = target_palm_pos
                data.qpos[3:7] = target_palm_quat
                data.qpos[7] = 0.0  # 手指张开
                data.qpos[8] = final_spread_synergy

                # TODO 打印qpos对应的index含义
                
                # 【重要】不要直接设置手指 qpos，让执行器去控制！
                # 移除了对 data.qpos 中手指关节的直接设置

                # 1.0s ~ 3.0s：手指闭合
                grasp_start_time = 1.0
                grasp_duration = 2.0
                
                if anim_time > grasp_start_time:
                    grasp_progress = (anim_time - grasp_start_time) / grasp_duration
                    grasp_progress = np.clip(grasp_progress, 0.0, 1.0)
                    current_grasp_val = final_grasp_synergy * grasp_progress
                else:
                    current_grasp_val = 0.0

                # 构造临时 pose 用于计算 ctrl
                temp_pose = np.zeros(9)
                temp_pose[7] = current_grasp_val
                temp_pose[8] = final_spread_synergy
                
                # 【改进】使用新的 ctrl 计算函数
                ctrl_cmd = qpos_to_ctrl_improved(model, planner, temp_pose)
                data.ctrl[:] = ctrl_cmd
                
                mujoco.mj_step(model, data)

            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
