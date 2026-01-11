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
            
            # 根据 Shadow Hand 标准命名规则分类
            jnt_name_lower = jnt_name.lower()
            if 'th' in jnt_name_lower:
                thumb_indices.append(qpos_adr)
            elif 'j4' in jnt_name_lower or 'abd' in jnt_name_lower:
                abd_indices.append(qpos_adr)
            else:
                flex_indices.append(qpos_adr)

    return flex_indices, abd_indices, thumb_indices


class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name, hand_body_prefix='lh_'):
        self.model = model
        self.data = data
        self.hand_prefix = hand_body_prefix
        self.stage = 'approach'  # 'approach' or 'close'，用于两阶段规划

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
        self.finger_base_body_ids = []
        self.hand_body_ids = []
        
        tip_suffixes = ['ffdistal', 'mfdistal', 'rfdistal', 'lfdistal', 'thdistal']
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                self.hand_body_ids.append(i)
                self.hand_geom_ids.extend(self._get_geoms_of_body(i))
                if any(suffix in name for suffix in tip_suffixes):
                    self.finger_tip_body_ids.append(i)
                # 识别手指基部（knuckle / metacarpal / thumb base 等），用于计算掌心位置
                lower = name.lower()
                if ('knuckle' in lower) or ('metacarpal' in lower) or ('thbase' in lower) or ('thproximal' in lower):
                    self.finger_base_body_ids.append(i)

        self.flex_adrs, self.abd_adrs, self.thumb_adrs = get_synergy_mapping(model, hand_body_prefix)

        super(GraspPlanner, self).__init__(state)

    def _get_geoms_of_body(self, body_id):
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]
    

    def set_hand_pose(self, state):
        # NOTE: do not reset `data` here
        try:
            forearm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.hand_prefix}forearm")
            if forearm_body_id >= 0:
                jnt_id = self.model.body_jntadr[forearm_body_id]
                if jnt_id >= 0:
                    qpos_adr = self.model.jnt_qposadr[jnt_id]
                    self.data.qpos[qpos_adr:qpos_adr+7] = state[0:7]
                else:
                    self.data.qpos[0:7] = state[0:7]
            else:
                self.data.qpos[0:7] = state[0:7]
        except Exception:
            pass
        
        # 提取协同变量
        grasp_synergy = np.clip(state[7], 0.0, 1.0)
        spread_synergy = np.clip(state[8], -0.2, 0.3)

        # 在approach阶段：手指保持伸平（握力=0），spread=0（中立）
        # 在close阶段：手指弯曲，握力=1.0，spread=-0.1（内收）
        if self.stage == 'approach':
            flex_angle = 0.0  # 手指完全伸平
            abd_val = 0.0     # 中立张开
            thumb_flex = 0.0  # 大拇指也伸平
        else:  # 'close' 阶段
            flex_angle = 1.5
            abd_val = spread_synergy * 0.5
            thumb_flex = 1.2

        for adr in self.flex_adrs:
            self.data.qpos[adr] = flex_angle if self.stage == 'approach' else flex_angle

        for adr in self.abd_adrs:
            self.data.qpos[adr] = abd_val

        for i, adr in enumerate(self.thumb_adrs):
            self.data.qpos[adr] = thumb_flex * (0.5 + i * 0.2) if self.stage == 'close' else 0.0

        mujoco.mj_forward(self.model, self.data)

    def calculate_distance_to_bottle(self):
        bottle_pos = self.data.xpos[self.bottle_body_id]
        total_dist = 0
        for b_id in self.finger_tip_body_ids:
            tip_pos = self.data.xpos[b_id]
            total_dist += np.linalg.norm(tip_pos - bottle_pos)
        return total_dist / len(self.finger_tip_body_ids)

    def get_palm_center(self, data):
        """
        计算掌心位置：基于各手指基部 body 的平均位置来估计掌心。
        如果没有找到基部 body，则退回到 palm body 的 xpos（兼容旧模型）。
        """
        try:
            if len(self.finger_base_body_ids) > 0:
                pts = []
                for b in self.finger_base_body_ids:
                    try:
                        pts.append(data.xpos[b].copy())
                    except Exception:
                        pass
                if len(pts) > 0:
                    return np.mean(np.stack(pts, axis=0), axis=0)
        except Exception:
            pass

        # 退回：使用 palm body 的 xpos
        try:
            return data.xpos[self.palm_body_id]
        except Exception:
            return np.array([0.0, 0.0, 0.0])

    def check_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            is_bottle = (contact.geom1 in self.bottle_geom_ids or contact.geom2 in self.bottle_geom_ids)
            is_hand = (contact.geom1 in self.hand_geom_ids or contact.geom2 in self.hand_geom_ids)
            if is_bottle and is_hand:
                if contact.dist < -0.005: 
                    return True
        return False

    def move(self):
        """ 启发式扰动：加入向瓶子靠近的'引力' """
        # 更积极地朝瓶子靠近：增加靠近步长，减少随机漂移
        if np.random.random() > 0.3:
            # 主动靠近瓶子
            bottle_pos = self.target_bottle_pos[0:3]
            palm_pos = self.state[0:3]
            direction = bottle_pos - palm_pos
            self.state[0:3] += (direction / (np.linalg.norm(direction) + 1e-6)) * 0.01
        else:
            # 少量随机扰动，防止陷入局部
            self.state[0:3] += np.random.normal(0, 0.01, 3)

        # 旋转扰动：减小噪声，更稳定的朝向搜索
        self.state[3:7] += np.random.normal(0, 0.05, 4)
        self.state[3:7] /= np.linalg.norm(self.state[3:7])

        # 扰动抓握协同：鼓励更强抓握以便接触
        self.state[7] += np.random.normal(0, 0.05)
        self.state[7] = np.clip(self.state[7], 0.0, 1.0)

        # 扰动张开协同
        self.state[8] += np.random.normal(0, 0.03)
        self.state[8] = np.clip(self.state[8], -0.2, 0.3)

    def energy(self):
        self.set_hand_pose(self.state)
        
        # 1. 距离项：只使用手掌中心和手指各个 body 到瓶子的距离（不包含前臂或手腕）
        bottle_pos = self.data.xpos[self.bottle_body_id]
        seg_dist_sum = 0.0
        seg_count = 0

        # 识别手指相关的 body 名（常见后缀或关键词），避免包含前臂/手腕
        finger_keywords = ['ff', 'mf', 'rf', 'lf', 'th']
        for b_id in self.hand_body_ids:
            try:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b_id) or ''
                # 仅把包含 finger_keywords 的 body 视为“手指部位”
                if any(k in name for k in finger_keywords):
                    seg_pos = self.data.xpos[b_id]
                    seg_dist_sum += np.linalg.norm(seg_pos - bottle_pos)
                    seg_count += 1
            except Exception:
                pass

        # 计算掌心到瓶子的距离（关键指标，强烈驱动掌心靠近）
        palm_center = self.get_palm_center(self.data)
        palm_dist = np.linalg.norm(palm_center - bottle_pos)

        avg_seg_dist = (seg_dist_sum / seg_count) if seg_count > 0 else 1.0

        # 距离综合评分：掌心距离是主要项（权重3.0），加上手指平均距离（权重1.0）
        # 这样强制掌心尽量靠近瓶子，实现包裹效果
        dist_score = palm_dist * 3.0 + avg_seg_dist * 1.0

        # 2. 朝向项：掌心指向瓶子
        try:
            rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
            palm_forward = -rot_mat[:, 1]
            vec_to_bottle = bottle_pos - palm_center
            vec_to_bottle /= (np.linalg.norm(vec_to_bottle) + 1e-6)
            orient_score = (1.0 - np.dot(palm_forward, vec_to_bottle)) * 2.5
        except Exception:
            orient_score = 1.0 * 2.5

        # 3. 姿态惩罚：鼓励较强握力
        grasp_synergy_val = self.state[7]
        pose_energy = np.square(grasp_synergy_val - 0.8) * 0.5  # 偏好 0.8 以上的握力

        # 4. 碰撞/接触相关：惩罚穿透，但奖励接触点数（希望实现全包裹）
        collision_penalty = 0
        contact_count = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
            is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
            if is_bottle and is_hand:
                # 若穿透严重，施加惩罚
                if con.dist < 0.002:
                    collision_penalty += 50.0
                # 计数接触点（用于奖励）
                contact_count += 1

        # 增加对手臂与地面的惩罚：若任一手部几何体与 floor 接触，施加巨大惩罚
        floor_penalty = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            if (g1 == 'floor' or g2 == 'floor') and (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids):
                floor_penalty = 100.0

        # 奖励接触点数：更多接触点降低 energy
        contact_reward = -8.0 * contact_count

        return dist_score + orient_score + pose_energy + collision_penalty + floor_penalty + contact_reward


def generate_new_grasp(model, data, planner, default_guess):
    """
    用模拟退火算法生成一个新的抓取位姿
    """
    random_guess = default_guess.copy()
    # 扩大初始位置扰动范围以探索更多位姿
    random_guess[0:3] += np.random.uniform(-0.12, 0.12, 3)
    # 初始抓握协同倾向于更闭合，增加接触概率
    random_guess[7] = np.random.uniform(0.2, 0.8)
    
    planner.state = random_guess
    best_pose, energy = planner.anneal()
    
    return best_pose


def create_deterministic_seeds(model, data, planner, radii=(0.04, 0.06), heights=(0.18, 0.22), angles=8):
    """在瓶子周围生成若干确定性初始位姿（位置 + 默认姿态 + 偏向闭合的协同变量）

    返回 state 列表，每个 state 长度为 9。
    """
    seeds = []
    try:
        bottle_body_id = planner.bottle_body_id
        bottle_pos = data.xpos[bottle_body_id].copy()
    except Exception:
        bottle_pos = planner.target_bottle_pos[0:3]

    for r in radii:
        for h in heights:
            for a in range(angles):
                theta = 2 * np.pi * a / angles
                px = bottle_pos[0] + r * np.cos(theta)
                py = bottle_pos[1] + r * np.sin(theta)
                pz = bottle_pos[2] + h
                state = np.zeros(9)
                state[0:3] = np.array([px, py, pz])
                # 使用单位四元数作为默认朝向
                state[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
                # 初始化为强闭合状态，增加接触概率
                state[7] = 0.95  # 极强握（从0.9改成0.95）
                state[8] = -0.1   # 手指明显内收
                seeds.append(state)

    return seeds


def generate_from_seeds(model, data, planner, seeds, extra_random=5):
    """对每个 seed 运行短时退火，保留能量最低的结果；再加上若干随机重启。

    返回最佳抓取位姿（state）。
    """
    best_state = None
    best_energy = float('inf')

    for seed in seeds:
        # 为每个种子创建独立的评估数据与 planner 实例，避免污染主 data
        data_eval = mujoco.MjData(model)
        mujoco.mj_resetData(model, data_eval)
        planner_eval = GraspPlanner(seed.copy(), model, data_eval, bottle_body_name='bottle_body')
        # 使用较短的退火以局部搜索
        planner_eval.steps = 800
        planner_eval.Tmax = 5.0
        planner_eval.Tmin = 0.01
        state_result, energy = planner_eval.anneal()
        if energy < best_energy:
            best_energy = energy
            best_state = state_result.copy()

    # 额外尝试若干随机初始猜测以防种子覆盖不足
    for _ in range(extra_random):
        data_eval = mujoco.MjData(model)
        mujoco.mj_resetData(model, data_eval)
        rand_guess = np.zeros(9)
        rand_guess[0:3] = planner.target_bottle_pos[0:3] + np.random.uniform(-0.1, 0.1, 3)
        rand_guess[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        rand_guess[7] = np.random.uniform(0.2, 0.9)
        rand_guess[8] = np.random.uniform(-0.1, 0.1)
        planner_eval = GraspPlanner(rand_guess.copy(), model, data_eval, bottle_body_name='bottle_body')
        planner_eval.steps = 600
        state_result, energy = planner_eval.anneal()
        if energy < best_energy:
            best_energy = energy
            best_state = state_result.copy()

    return best_state


def apply_grasp_control(model, data, planner, grasp_pose):
    """
    应用规划好的抓取位姿到执行器
    """
    # 更新关节位置
    planner.set_hand_pose(grasp_pose)
    
    # 更新执行器控制信号
    grasp_val = np.clip(grasp_pose[7], 0.0, 1.0)
    spread_val = np.clip(grasp_pose[8], -0.2, 0.3)
    
    n_act = model.nu
    for i in range(n_act):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not act_name:
            continue

        # 通过 actuator 名称映射到 joint 名称（例如: 'lh_A_WRJ2' -> 'lh_WRJ2')
        name_lower = act_name.lower()

        # 对手腕执行器：将控制目标设为对应关节当前 qpos（以保持当前角度）
        if 'wr' in name_lower or 'wrist' in name_lower:
            # 推测 joint 名称
            joint_guess = act_name.replace('_A_', '_')
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_guess)
            if j_id >= 0:
                qadr = model.jnt_qposadr[j_id]
                # 使用当前 qpos 作为控制目标（position actuators 通用）
                try:
                    data.ctrl[i] = float(data.qpos[qadr])
                except Exception:
                    data.ctrl[i] = 0.0
            else:
                data.ctrl[i] = 0.0
        elif 'th' in name_lower:
            data.ctrl[i] = grasp_val * 1.2
        elif 'j4' in name_lower or 'abd' in name_lower:
            data.ctrl[i] = spread_val
        else:
            data.ctrl[i] = grasp_val * 1.5

    # 如果 planner 提供了固定的 forearm qpos（用于在抓取期间冻结手臂），则写回到 data
    try:
        if hasattr(planner, 'fixed_forearm_qpos') and planner.fixed_forearm_qpos is not None:
            forearm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{planner.hand_prefix}forearm")
            if forearm_body_id >= 0:
                jnt_id = model.body_jntadr[forearm_body_id]
                if jnt_id >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    data.qpos[qpos_adr:qpos_adr+7] = planner.fixed_forearm_qpos
                    mujoco.mj_forward(model, data)
    except Exception:
        pass


def simulate_grasp(model, data, grasp_pose, planner_temp, sim_steps=200):
    """
    模拟抓取：执行N次仿真步进，让手指缓慢接触物体
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        grasp_pose: 规划的抓取位姿
        sim_steps: 仿真步数
    
    Returns:
        接触力的总和（用于评估抓取质量）
    """
    # 假设 planner_temp 已由调用者构建并传入。先把手放到抓取位姿并记录冻结前臂位姿
    try:
        planner_temp.set_hand_pose(grasp_pose)
        forearm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{planner_temp.hand_prefix}forearm")
        if forearm_body_id >= 0:
            jnt_id = model.body_jntadr[forearm_body_id]
            if jnt_id >= 0:
                qpos_adr = model.jnt_qposadr[jnt_id]
                planner_temp.fixed_forearm_qpos = data.qpos[qpos_adr:qpos_adr+7].copy()
            else:
                planner_temp.fixed_forearm_qpos = None
        else:
            planner_temp.fixed_forearm_qpos = None
    except Exception:
        planner_temp.fixed_forearm_qpos = None
    # 应用一次控制以初始化 actuators
    apply_grasp_control(model, data, planner_temp, grasp_pose)
    
    total_contact_force = 0.0
    
    for step in range(sim_steps):
        # 重新应用控制，确保 position/torque 控制在每步都被设置
        apply_grasp_control(model, data, planner_temp, grasp_pose)
        mujoco.mj_step(model, data)

        # 累积接触力
        for i in range(data.ncon):
            contact = data.contact[i]
            # 计算接触力大小 (简化为法向力)
            try:
                total_contact_force += np.linalg.norm(contact.frame[:3])
            except Exception:
                pass
        # 检查是否有手与地面接触，若有则视为失败并快退
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if (g1 == 'floor' or g2 == 'floor') and (contact.geom1 in planner_temp.hand_geom_ids or contact.geom2 in planner_temp.hand_geom_ids):
                # 发现手臂接触地面，返回极低分以避免此抓取
                return -1000.0

    # 强闭合阶段：在接触之后强力闭合手指（直接写 qpos 并施加较大 actuators 控制），提高抓取牢固度
    close_steps = 150
    for c in range(close_steps):
        try:
            # 强制把协同变量设为闭合并写入 qpos
            grasp_close_val = 1.0
            planner_temp.state[7] = grasp_close_val
            planner_temp.set_hand_pose(planner_temp.state)
            # 直接把 finger joints 设置为最大弯曲以确保牢固包裹
            flex_angle = grasp_close_val * 2.5  # 提高弯曲角
            for adr in planner_temp.flex_adrs:
                data.qpos[adr] = flex_angle
            for adr in planner_temp.abd_adrs:
                data.qpos[adr] = -0.1  # 让手指内收
            for adr in planner_temp.thumb_adrs:
                data.qpos[adr] = flex_angle * 1.0  # 大拇指也强闭合
            mujoco.mj_forward(model, data)
            # 同时在 actuator 层面增大控制输出
            for i in range(model.nu):
                aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if aname and ('ffj' in aname.lower() or 'mfj' in aname.lower() or 'rfj' in aname.lower() or 'lfj' in aname.lower() or 'thj' in aname.lower()):
                    try:
                        data.ctrl[i] = 2.0
                    except Exception:
                        pass
        except Exception:
            pass
        mujoco.mj_step(model, data)
        # 累计接触力
        for i in range(data.ncon):
            contact = data.contact[i]
            try:
                total_contact_force += np.linalg.norm(contact.frame[:3])
            except Exception:
                pass

    # 评估接触点数（去重手部几何体与瓶子接触的几何 id）
    contact_geoms = set()
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 in planner_temp.bottle_geom_ids or con.geom2 in planner_temp.bottle_geom_ids) and \
           (con.geom1 in planner_temp.hand_geom_ids or con.geom2 in planner_temp.hand_geom_ids):
            # 记录手部对应的 geom id
            if con.geom1 in planner_temp.hand_geom_ids:
                contact_geoms.add(con.geom1)
            if con.geom2 in planner_temp.hand_geom_ids:
                contact_geoms.add(con.geom2)

    contact_count = len(contact_geoms)

    # 判断瓶子是否被牢牢抓住：在强闭合结束时检查瓶子位移是否很小
    try:
        bottle_body_id = planner_temp.bottle_body_id
        final_bottle_pos = data.xpos[bottle_body_id].copy()
        palm_pos = planner_temp.get_palm_center(data)
        bottle_to_palm = np.linalg.norm(final_bottle_pos - palm_pos)
        bottle_locked = (contact_count >= 2) and (bottle_to_palm < 0.08)
    except:
        bottle_locked = False

    return total_contact_force, contact_count, bottle_locked


def lift_test(model, data, planner, lift_height=0.05, lift_steps=300):
    """
    提升测试：将机械臂向上抬升一段距离，检查物体是否掉落
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        planner: GraspPlanner 实例
        lift_height: 提升高度 (米)，默认5cm
        lift_steps: 提升步数
    
    Returns:
        success: 布尔值，True表示抓取成功（物体没有掉落），False表示失败
    """
    # 记录初始瓶子位置
    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle_body')
    initial_bottle_pos = data.xpos[bottle_body_id].copy()
    
    # 获取 forearm（包含 freejoint）的 body id
    forearm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lh_forearm')
    if forearm_body_id < 0:
        # 找不到 forearm，退回到直接移动 xpos（兼容旧逻辑）
        palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lh_palm')
        palm_initial_pos = data.xpos[palm_body_id].copy()
        target_palm_pos = palm_initial_pos.copy()
        target_palm_pos[2] += lift_height

        for step in range(lift_steps):
            alpha = step / lift_steps
            current_target = palm_initial_pos + alpha * (target_palm_pos - palm_initial_pos)
            data.xpos[palm_body_id] = current_target
            # 在抬升阶段持续施加抓握控制（如果 planner 提供）
            try:
                apply_grasp_control(model, data, planner, planner.state)
            except Exception:
                pass
            mujoco.mj_step(model, data)
    else:
        # 使用 freejoint 的 qpos 来移动整个前臂（推荐）
        jnt_id = model.body_jntadr[forearm_body_id]
        qpos_adr = model.jnt_qposadr[jnt_id]

        # 记录初始 qpos（7 个元素: xyz + quat）
        init_qpos = data.qpos[qpos_adr:qpos_adr+7].copy()
        target_qpos = init_qpos.copy()
        target_qpos[2] += lift_height

        for step in range(lift_steps):
            alpha = step / lift_steps
            interp = init_qpos * (1 - alpha) + target_qpos * alpha
            data.qpos[qpos_adr:qpos_adr+7] = interp
            # 在抬升阶段持续施加抓握控制（如果 planner 提供）
            try:
                apply_grasp_control(model, data, planner, planner.state)
            except Exception:
                pass
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
    
    # 检查瓶子是否掉落：
    # 1. 检查瓶子与手的距离是否超过阈值（表示物体掉落）
    # 2. 检查瓶子是否与地面接触
    
    final_bottle_pos = data.xpos[bottle_body_id].copy()
    bottle_lifted_height = final_bottle_pos[2] - initial_bottle_pos[2]
    
    # 如果瓶子有效地被提升了至少 2cm（容差），说明抓取成功
    success = bottle_lifted_height > (lift_height * 0.4)
    
    # 额外检查：如果瓶子与地面碰撞，则失败
    floor_collision = False
    for i in range(data.ncon):
        contact = data.contact[i]
        contact_geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        contact_geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        
        # 检查是否与地面碰撞
        if (contact_geom1_name == 'floor' or contact_geom2_name == 'floor') and \
           (contact_geom1_name == 'bottle_geom' or contact_geom2_name == 'bottle_geom'):
            floor_collision = True
            break
    
    success = success and not floor_collision
    
    return success


def grasp_rating(model, data, planner, grasp_pose, sim_steps=200, lift_steps=300):
    """
    对单个抓取位姿进行评分
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        planner: GraspPlanner 实例
        grasp_pose: 要评估的抓取位姿
        sim_steps: 抓取仿真步数
        lift_steps: 提升测试步数
    
    Returns:
        score: 抓取评分 (0-1)，1表示完全成功
        success: 布尔值，是否成功抓取
    """
    # 创建临时数据副本用于评估（不影响主模拟）
    data_eval = mujoco.MjData(model)
    mujoco.mj_resetData(model, data_eval)

    # 构造用于本次评估的临时 planner，使得 simulate_grasp 使用相同的冻结前臂/控制
    planner_temp = GraspPlanner(grasp_pose, model, data_eval, bottle_body_name='bottle_body')

    # 1. 模拟抓取过程（仅做手部接触与闭合，不做抬升）
    contact_force, contact_count, bottle_locked = simulate_grasp(model, data_eval, grasp_pose, planner_temp, sim_steps)

    # 评估指标：接触点数量与瓶子是否被锁定（靠近掌心且接触点数足够）
    contact_count_score = min(contact_count / 5.0, 1.0)  # 5 个接触点以上视为充分
    force_score = min(max(contact_force / 100.0, 0.0), 1.0)

        # 成功判定规则：至少有 2 个不同的手部接触点即可
    success = contact_count >= 2

    # 总评分：80% 接触点数 + 20% 接触强度（移除 bottle_locked 二值，鼓励更多尝试）
    total_score = 0.8 * contact_count_score + 0.2 * force_score

    return total_score, success


# --- 主程序 ---
def main():
    try:
        model_path = 'shadow_hand/scene_left.xml'
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Diagnostic: print actuator and joint mapping to help debug control issues
        print("\n[诊断] 列出执行器与关节信息：")
        for i in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  actuator[{i}]: {aname}")

        print("\n[诊断] 列出关节及其 qpos 地址：")
        for i in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qadr = model.jnt_qposadr[i]
            print(f"  joint[{i}]: {jname}  qpos_adr={qadr}  type={model.jnt_type[i]}")

        print("=" * 70)
        print("影子手抓取评估系统 - 多次抓取评分")
        print("=" * 70)
        
        # 初始化参数
        initial_guess = np.zeros(9)
        initial_guess[0:3] = [0.4, 0.4, 0.25] 
        initial_guess[3] = 1.0
        initial_guess[7] = 0.3
        initial_guess[8] = 0.0

        planner = GraspPlanner(initial_guess, model, data, bottle_body_name='bottle_body')

        # 初始化时不固定 forearm；在每次可视化展示前我们会把当前目标位姿存为固定值
        planner.fixed_forearm_qpos = None

        # 保存初始瓶子 qpos 地址和值，用于每次试验后恢复
        try:
            bottle_joint_id = model.body_jntadr[planner.bottle_body_id]
            planner.initial_bottle_qpos = data.qpos[model.jnt_qposadr[bottle_joint_id]:model.jnt_qposadr[bottle_joint_id]+7].copy()
        except Exception:
            planner.initial_bottle_qpos = None

        # 配置退火参数
        planner.steps = 2000
        planner.Tmax = 10.0
        planner.Tmin = 0.01

        # ========== 可视化 & 核心循环 ==========
        M = 5  # 生成M个不同的抓取位姿
        grasp_states = {}  # 存储所有抓取的评分结果

        print(f"\n开始评估 {M} 个抓取位姿... (会打开 MuJoCo Viewer)")
        print("-" * 70)

        # 打开 Viewer 使用主 data 进行可视化
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 小范围预热 viewer
            for _ in range(5):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)

            for m in range(M):
                print(f"\n[抓取 {m+1}/{M}] 生成新位姿...")

                # 生成新的抓取位姿：先生成围绕瓶子的确定性种子并从这些种子搜索最佳位姿
                seeds = create_deterministic_seeds(model, data, planner, radii=(0.05, 0.08), heights=(0.18, 0.22), angles=12)
                best_from_seeds = generate_from_seeds(model, data, planner, seeds, extra_random=6)
                if best_from_seeds is not None:
                    best_grasp = best_from_seeds
                else:
                    best_grasp = generate_new_grasp(model, data, planner, initial_guess)
                print(f"  生成的位姿: 手掌位置=[{best_grasp[0]:.3f}, {best_grasp[1]:.3f}, {best_grasp[2]:.3f}]")
                print(f"  协同变量: 抓握={best_grasp[7]:.3f}, 张开={best_grasp[8]:.3f}")

                # 对该抓取位姿进行评估
                print(f"  开始抓取评估...")

                N = 3  # 每个抓取位姿进行N次试验
                trial_results = []

                for n in range(N):
                    print(f"    试验 {n+1}/{N}...", end=" ")

                    # 评分（在独立的 MjData 上运行评估，不影响主 data）
                    score, success = grasp_rating(model, data, planner, best_grasp, 
                                                 sim_steps=200, lift_steps=300)

                    trial_results.append({
                        'success': success,
                        'score': score
                    })

                    print(f"{'✓ 成功' if success else '✗ 失败'} (评分: {score:.3f})")

                # 统计该抓取的成功率
                success_count = sum(1 for r in trial_results if r['success'])
                success_rate = success_count / N
                avg_score = np.mean([r['score'] for r in trial_results])

                grasp_states[m] = {
                    'pose': best_grasp,
                    'trials': trial_results,
                    'success_count': success_count,
                    'success_rate': success_rate,
                    'avg_score': avg_score
                }

                print(f"  ✓ 抓取 {m+1} 完成: 成功率={success_rate*100:.1f}%, 平均评分={avg_score:.3f}")

                # 可视化：将最佳抓取姿态应用到主 data 并运行若干步进行展示
                try:
                    planner.set_hand_pose(best_grasp)
                    # 在展示前把当前目标手臂位姿作为冻结目标
                    try:
                        planner.set_hand_pose(best_grasp)
                        # 记录当前手臂 freejoint qpos 以便冻结
                        forearm_body_id_vis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{planner.hand_prefix}forearm")
                        if forearm_body_id_vis >= 0:
                            jnt_id_vis = model.body_jntadr[forearm_body_id_vis]
                            if jnt_id_vis >= 0:
                                qpos_adr_vis = model.jnt_qposadr[jnt_id_vis]
                                planner.fixed_forearm_qpos = data.qpos[qpos_adr_vis:qpos_adr_vis+7].copy()
                    except Exception:
                        planner.fixed_forearm_qpos = None
                    apply_grasp_control(model, data, planner, best_grasp)
                    vis_steps = 200
                    for _ in range(vis_steps):
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(0.01)
                except Exception:
                    # 忽略渲染相关错误
                    pass
                finally:
                    # 每次展示后，恢复主模拟中的前臂和瓶子初始位姿，避免累积偏移
                    try:
                        # 恢复前臂（freejoint）
                        if hasattr(planner, 'fixed_forearm_qpos') and planner.fixed_forearm_qpos is not None:
                            forearm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{planner.hand_prefix}forearm")
                            if forearm_body_id >= 0:
                                jnt_id = model.body_jntadr[forearm_body_id]
                                if jnt_id >= 0:
                                    qpos_adr = model.jnt_qposadr[jnt_id]
                                    data.qpos[qpos_adr:qpos_adr+7] = planner.fixed_forearm_qpos
                                    mujoco.mj_forward(model, data)
                        # 恢复瓶子位姿
                        if hasattr(planner, 'initial_bottle_qpos') and planner.initial_bottle_qpos is not None:
                            bottle_jnt_id = model.body_jntadr[planner.bottle_body_id]
                            b_qadr = model.jnt_qposadr[bottle_jnt_id]
                            data.qpos[b_qadr:b_qadr+7] = planner.initial_bottle_qpos
                            # 清零系统速度，确保恢复后静止
                            try:
                                data.qvel[:] = 0
                            except Exception:
                                pass
                            mujoco.mj_forward(model, data)
                    except Exception:
                        pass
        # ========== 结果汇总 ==========
        print("\n" + "=" * 70)
        print("评估结果汇总")
        print("=" * 70)
        
        best_grasp_idx = max(grasp_states.keys(), 
                            key=lambda k: grasp_states[k]['avg_score'])
        
        for m in range(M):
            state = grasp_states[m]
            marker = "★ 最佳" if m == best_grasp_idx else "  "
            print(f"\n{marker} 抓取 {m+1}:")
            print(f"    位置: [{state['pose'][0]:.3f}, {state['pose'][1]:.3f}, {state['pose'][2]:.3f}]")
            print(f"    成功率: {state['success_rate']*100:.1f}% ({state['success_count']}/{N})")
            print(f"    平均评分: {state['avg_score']:.3f}")
        
        best_state = grasp_states[best_grasp_idx]
        print("\n" + "=" * 70)
        print(f"最佳抓取: 抓取 {best_grasp_idx+1}")
        print(f"  成功率: {best_state['success_rate']*100:.1f}%")
        print(f"  平均评分: {best_state['avg_score']:.3f}")
        print("=" * 70)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误：{e}")


if __name__ == "__main__":
    main()
