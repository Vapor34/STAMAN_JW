# hnb3.py 代码逻辑整理

## 📋 整体架构

这个项目是一个 **影子手 (Shadow Hand) + 瓶子抓取仿真系统**，核心逻辑分为三层：

```
┌─────────────────────────────────────────┐
│  1️⃣  规划层 (GraspPlanner - 模拟退火)   │ ← 使用 Annealer 优化手掌位置
│     输出：最优的手掌姿态 (位置+旋转)     │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  2️⃣  协同变量映射 (Synergy Mapping)     │
│     手指协同 + 侧摆 → 关节角度           │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  3️⃣  执行层 (MuJoCo 动画和控制)         │
│     手掌移动 → 手指依次闭合 → 实时渲染   │
└─────────────────────────────────────────┘
```

---

## 🔍 详细逻辑流程

### 第一步：初始化

```python
def main():
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path('shadow_hand/scene_left.xml')
    data = mujoco.MjData(model)
    
    # 2. 设置初始状态
    initial_guess = np.zeros(9)
    initial_guess[0:3] = [0.4, 0.4, 0.3]  # 手掌初始位置
    initial_guess[3:7] = [1, 0, 0, 0]     # 手掌四元数 (恒等旋转)
    initial_guess[7] = 0.0                 # 握力协同变量 (0=张开)
    initial_guess[8] = ?                   # 侧摆协同变量
    
    # 3. 创建规划器
    planner = GraspPlanner(initial_guess, model, data, 'bottle_body')
```

**状态向量 `state[9]`：**
| 索引 | 内容 | 范围 |
|------|------|------|
| 0-2 | 手掌位置 (x, y, z) | 无约束 |
| 3-6 | 手掌四元数 (qx, qy, qz, qw) | 单位四元数 |
| 7 | 握力协同 (grasp_synergy) | [0, 1] |
| 8 | 侧摆协同 (spread_synergy) | [-0.2, 0.3] |

---

### 第二步：规划阶段（模拟退火优化）

**触发条件：** `data.time < 0.002` (仿真刚开始)

#### 2.1 GraspPlanner 初始化

```python
class GraspPlanner(Annealer):
    def __init__(self, state, model, data, bottle_body_name):
        # 1. 获取对象 ID
        self.bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body_name)
        self.bottle_geom_ids = self._get_geoms_id_of_body(self.bottle_body_id)
        
        # 2. 获取手部信息
        self.palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lh_palm')
        self.hand_geom_ids = [所有手部几何 ID]
        self.contact_body_ids = [排除腕部/前臂后的所有手指 body ID]
        
        # 3. 获取关节映射（关键！）
        self.flex_adrs, self.abd_adrs, self.thumb_adrs = get_synergy_mapping(model, 'lh_')
```

**关节分类：**
```
┌─────────────────────────────────────────┐
│      影子手关节映射 (get_synergy_mapping) │
├─────────────────────────────────────────┤
│ • flex_adrs    → j1, j2, j3 (手指弯曲)   │
│ • abd_adrs     → j4, abd (手指侧摆)      │
│ • thumb_adrs   → th* (大拇指关节)        │
└─────────────────────────────────────────┘
```

#### 2.2 退火主循环（每次迭代）

```
每一步迭代：

1️⃣  move()
    ├─ 手掌位移：随机微调 或 启发式移向瓶子
    ├─ 旋转扰动：四元数随机扰动 + 重新归一化
    └─ 协同变量扰动：grasp/spread 随机扰动 + 裁剪

2️⃣  energy()
    ├─ set_hand_pose(self.state)
    │  ├─ 重置物理数据
    │  ├─ 根据 state[0:7] 设置手掌位置和旋转
    │  ├─ 根据 state[7] (grasp_synergy) 计算手指弯曲角度
    │  └─ 根据 state[8] (spread_synergy) 计算手指侧摆角度
    │
    └─ 计算多项能量：
       ├─ 距离项：手部关键点到瓶子的平均距离 × 25.0
       ├─ 朝向项：手心法向与指向瓶子方向的夹角 × 10
       ├─ 碰撞惩罚：手与瓶子/地面穿模 × 200~500
       ├─ 关节限位惩罚：超出 XML 范围的关节 × 1000
       └─ 姿态先验：握力 != 0.4 的平方差

3️⃣  接受/拒绝决策
    ├─ 如果 ΔE < 0（能量降低）→ 接受
    └─ 如果 ΔE > 0（能量升高）→ 以概率 exp(-ΔE/T) 接受（模拟退火）

⏱️  温度下降：T = T_max × exp(-log(T_max/T_min) × step/steps)
```

**关键问题的答案：**

❓ **问题：energy 计算时，是否基于五指伸直？**

✅ **答案：不是。** energy 会根据 `move()` 中随机扰动后的 `state[7]` 来设置手指角度。
- 如果你想让 energy 基于五指伸直来计算距离，应该在 `energy()` 中临时设置 `state[7]=0`

❓ **问题：move() 和 energy() 的执行顺序？**

✅ **答案：**
```
初始化：E = energy()

循环 step=1 to steps:
  1. move()           ← 修改 self.state
  2. energy()         ← 根据新的 self.state 计算能量
  3. 比较 ΔE 决定是否接受
```

---

### 第三步：执行阶段（动画和控制）

**触发条件：** `planning_done == True`

#### 3.1 获取规划结果

```python
best_pose, energy = planner.anneal()  # 2500 步退火

target_palm_pos = best_pose[0:3]      # 优化后的手掌位置
target_palm_quat = best_pose[3:7]     # 优化后的手掌旋转
final_grasp_synergy = best_pose[7]    # 优化后的握力
final_spread_synergy = best_pose[8]   # 优化后的侧摆
```

#### 3.2 动画序列

```
时间轴：

┌────────────────────────────────────────────────┐
│ 0.0s ~ 1.0s：手掌移动                          │
│ • 直接设置手掌位置/旋转到目标位置               │
│ • 手指保持张开 (qpos[7] = 0.0)                 │
│ • 侧摆保持最终值                               │
│                                                 │
│ 1.0s ~ 5.0s：手指闭合                          │
│ • 手掌保持不动                                 │
│ • current_grasp_val 从 0 → final_grasp_synergy │
│ • 每帧重新计算关节角度并应用控制信号            │
└────────────────────────────────────────────────┘
```

#### 3.3 协同变量 → 关节角度的映射

```python
def set_hand_pose(state):
    grasp_synergy = state[7]        # [0, 1]
    spread_synergy = state[8]       # [-0.2, 0.3]
    
    # 弯曲关节角度 = grasp_synergy × 1.5
    flex_angle = grasp_synergy * 1.5
    for adr in self.flex_adrs:
        qpos[adr] = flex_angle
    
    # 侧摆关节角度 = spread_synergy × 0.5
    for adr in self.abd_adrs:
        qpos[adr] = spread_synergy * 0.5
    
    # 大拇指角度 = grasp_synergy × (1.2 * (0.5~0.9))
    for i, adr in enumerate(self.thumb_adrs):
        qpos[adr] = grasp_synergy * 1.2 * (0.5 + i * 0.2)
```

#### 3.4 协同变量 → 执行器命令的映射

```python
def qpos_to_ctrl(model, planner, target_pose):
    grasp_val = target_pose[7]      # [0, 1]
    spread_val = target_pose[8]     # [-0.2, 0.3]
    grasp_force_factor = 1.15       # 过压系数
    
    for each actuator i:
        if actuator 对应 flex_adr:
            ctrl[i] = (grasp_val ^ 1.1) × 1.7 × 1.15  ← 非线性，握得越紧越快
        elif actuator 对应 thumb_adr:
            ctrl[i] = grasp_val × 1.3
        elif actuator 对应 abd_adr:
            ctrl[i] = spread_val
        
        # 确保在执行器限位内
        ctrl[i] = clip(ctrl[i], ctrl_range[0], ctrl_range[1])
```

---

## 📊 能量函数详解

```
总能量 = Σ 各项能量 × 权重

┌──────────────────────────────┐
│ 1. 距离项 (Distance Energy)   │
│    avg_dist × 25.0            │
│  → 鼓励手指靠近瓶子            │
└──────────────────────────────┘

┌──────────────────────────────┐
│ 2. 朝向项 (Orientation)       │
│    (1 - cos(angle)) × 10      │
│  → 鼓励手心朝向瓶子            │
└──────────────────────────────┘

┌──────────────────────────────┐
│ 3. 碰撞惩罚 (Collision)       │
│  • 手-瓶穿模 (dist<-5mm) 200  │
│  • 手-地面接触 500             │
│  → 避免穿模和碰撞地面          │
└──────────────────────────────┘

┌──────────────────────────────┐
│ 4. 关节限位惩罚 (Joint Limits)│
│    Σ (超出范围的差值)² × 1000 │
│  → 强制关节在 XML 范围内      │
└──────────────────────────────┘

┌──────────────────────────────┐
│ 5. 姿态先验 (Pose Prior)      │
│    (grasp - 0.4)²             │
│  → 倾向于中等握力而非极端值    │
└──────────────────────────────┘
```

---

## 🎯 关键设计点

### 1. 为什么需要两层映射？

```
状态层 (Synergy)
state[7], state[8]
     ↓
关节层 (Joint Angles)
qpos[adr1], qpos[adr2], ...
     ↓
执行器层 (Actuator Commands)
ctrl[0], ctrl[1], ...

好处：
• 降维：从 20+ 个自由度 → 2 个协同变量 → 易优化
• 物理感知：考虑执行器非线性 (grasp_val^1.1)
• 过压控制：grasp_force_factor = 1.15 增加握力
```

### 2. 为什么在 energy() 中调用 mj_resetData？

```python
def set_hand_pose(state):
    mujoco.mj_resetData(...)  # 重置物理状态
    
    # 设置几何位置和关节角度
    # 调用 mj_forward(...) 更新碰撞检测
    
原因：
• 每次 energy() 计算都需要独立的物理状态快照
• move() 产生新的 state 后，需要重新计算碰撞和几何
• 否则会累积上次计算的结果
```

### 3. 为什么 move() 中的启发式方向是针对上一次的 bottle_pos？

```python
def move(self):
    # 获取上一次 energy() 中计算的 bottle 位置
    bottle_pos = self.data.xpos[self.bottle_body_id]
    
⚠️ 这里有个细微问题：
self.data 是共享的全局数据，每次 set_hand_pose() 
都会重置它，所以 bottle_pos 其实是当前规划状态下的瓶子位置
（假设瓶子在仿真中不动）
```

---

## 🚀 执行流程总结

```
┌─────────────────────────────────────────────────────┐
│ main()                                              │
│ • 加载模型和数据                                    │
│ • 创建 GraspPlanner 实例                            │
│ • 启动 MuJoCo 查看器                                │
└──────────────┬──────────────────────────────────────┘
               │
               ↓
    ┌─────────────────────────────────────────────────┐
    │ 主循环 while viewer.is_running():               │
    │                                                  │
    │ 📍 if data.time < 0.002:                        │
    │    ├─ planner.state = initial_guess + noise    │
    │    ├─ best_pose, energy = planner.anneal()     │
    │    │  (内部 2500 次迭代)                        │
    │    └─ 获取 target_palm_pos, final_grasp, etc   │
    │                                                  │
    │ 📍 if planning_done:                            │
    │    ├─ 0.0s~1.0s：手掌移动 (直接吸附)           │
    │    ├─ 1.0s~5.0s：手指闭合 (协同变量递增)       │
    │    ├─ 每帧：ctrl_cmd = qpos_to_ctrl()          │
    │    └─ mujoco.mj_step() 物理步进                │
    │                                                  │
    │ ✨ viewer.sync() 渲染                          │
    │ ⏱️  帧率控制                                    │
    └─────────────────────────────────────────────────┘
```

---

## 💡 关键问题详解

### 问题 1：move() 中的 state[7] 扰动是否有效？

**核心问题：** 规划阶段优化出的 state[7] 在执行阶段被完全忽略，这会不会导致矛盾？

**答案：✅ 不会产生矛盾，设计是合理的。**

**原因分析：**

```
规划阶段 (0.0s ~ 0.001s)：
├─ 目标：找到最优的【手掌位置+旋转】
├─ move() 中扰动 state[7], state[8]：用来探索整个搜索空间
│  （因为手指配置会影响手掌是否能靠近瓶子，或避免碰撞）
├─ 关键：energy() 计算的是【整体配置】下的最优适应度
│  即：手掌在这个【特定手指姿态】下到达瓶子的最优性
└─ 结果：best_pose[7] ≈ 规划过程中找到的握力值

执行阶段 (1.0s ~ 5.0s)：
├─ 目标：演示抓取动作（五指从张开→闭合）
├─ 忽略 best_pose[7]：因为动画要求【手指递进闭合】
├─ 重新定义 final_grasp_synergy = best_pose[7]
│  （或使用其他固定值作为目标握力）
└─ current_grasp_val 从 0 → final_grasp_synergy（线性增长）

🎯 关键：这两个阶段的 state[7] 服务于不同目的
   • 规划阶段：state[7] 是【优化变量】，影响能量计算
   • 执行阶段：final_grasp_synergy 是【动画目标】，由规划结果确定
```

**具体流程：**

```python
# 规划阶段
best_pose, _ = planner.anneal()  # 2500 次迭代，move()+energy() 交替
final_grasp_synergy = best_pose[7]  # 获取规划出的握力值

# 执行阶段（0-1秒）
data.qpos[0:3] = target_palm_pos         # 手掌瞬移
data.qpos[7] = 0.0                       # 【忽略】规划的握力，强制手指张开
# ↑ 这里与 best_pose[7] 不同，但这是【有意的】

# 执行阶段（1-5秒）
current_grasp_val = final_grasp_synergy * grasp_progress  # 从 0 递增
# ↑ 现在才使用规划出的握力值
```

**结论：**
- ✅ move() 中的 state[7] 扰动**完全有效**，它探索搜索空间并找到最优手掌姿态
- ✅ 最终的 best_pose[7] **确实被保存**并在执行阶段 1-5 秒使用
- ✅ 执行阶段 0-1 秒忽略 best_pose[7] 是**设计决策**，用于演示动画流畅性

---

### 问题 2：执行器命令的积分问题详解

**当前实现：**

```python
def qpos_to_ctrl(model, planner, target_pose):
    """每帧计算一次控制命令"""
    ctrl_cmd = np.zeros(model.nu)
    grasp_val = target_pose[7]  # 当前协同值 [0, 1]
    
    for i in range(model.nu):
        # ... 根据 grasp_val 计算 ctrl[i]
        ctrl_cmd[i] = f(grasp_val)  # 非线性映射
    
    return ctrl_cmd
```

**问题分析：**

```
执行器的两层控制：

┌─────────────────────────────────────────┐
│ 第1层：协同变量 → 目标关节角度 (Set-point)  │
│                                         │
│ set_hand_pose(state):                   │
│   flex_angle = grasp_synergy * 1.5      │
│   qpos[adr] = flex_angle                │
│                                         │
│ 特点：纯几何映射，无动力学                │
└─────────────────────────────────────────┘
                   ↑
        这是你【想要】的关节角度
                   
                   ↓

┌─────────────────────────────────────────┐
│ 第2层：目标关节角度 → 执行器命令          │
│                                         │
│ qpos_to_ctrl(target_pose):              │
│   ctrl[i] = (grasp_val ^ 1.1) * 1.7     │
│                                         │
│ 问题：这里没有闭环反馈！                │
└─────────────────────────────────────────┘
                   ↑
        这是【实际】发送给执行器的指令
```

**具体问题（"积分"在这里的含义）：**

```
假设目标握力从 0 → 0.8（1秒内）：

┌──────────────────────────────────────────────────────┐
│ 时刻 t=0.0s：grasp_val = 0.0                         │
│   ctrl[i] = (0.0^1.1) * 1.7 = 0.0                    │
│   ✓ 执行器收到 0，保持静止                            │
├──────────────────────────────────────────────────────┤
│ 时刻 t=0.2s：grasp_val = 0.2                         │
│   ctrl[i] = (0.2^1.1) * 1.7 ≈ 0.056                  │
│   ? 执行器收到 0.056，但实际有延迟和动力学            │
├──────────────────────────────────────────────────────┤
│ 时刻 t=0.5s：grasp_val = 0.5                         │
│   ctrl[i] = (0.5^1.1) * 1.7 ≈ 0.577                  │
│   ? 执行器应该弯曲到某个角度，但可能未到达             │
├──────────────────────────────────────────────────────┤
│ 时刻 t=1.0s：grasp_val = 1.0                         │
│   ctrl[i] = (1.0^1.1) * 1.7 ≈ 1.7                    │
│   ? 执行器是否完全闭合？是否有过冲？                  │
└──────────────────────────────────────────────────────┘

【积分】是指：关节角度的【累积变化】

实际关节角度：
  qpos_actual(t) = ∫ velocity(τ) dτ  从 0 到 t

如果控制命令变化太快（grasp_val 线性增长），
执行器可能跟不上，导致【追踪误差】

例如：
• 期望：flexion_angle = grasp_val * 1.5（线性增长）
• 实际：flexion_angle = 0, 0, 0, ..., 0, 0.05, 0.10, ...
        （由于执行器惯性，滞后 50ms）
```

**问题的根源：**

```
当前代码假设：
✗ 执行器是【理想的】，ctrl 指令 = 实际关节角度

实际情况：
✓ 执行器有【动力学】，需要【时间】来响应
  
  MuJoCo 中的执行器配置：
  <actuator>
    <motor name="..." ...
           ctrlrange="[-1, 1]"       ← 控制范围
           gear="..."                 ← 齿比
           forcelimit="..."           ← 最大力
           ctrllimited="true" />      
  </actuator>
  
  执行器的实际行为：
  1. 接收 ctrl 指令
  2. 转换为力/扭矩
  3. 通过物理仿真逐步改变关节角度
  4. 关节速度受限于执行器的最大力
```

**影响分析：**

```
┌──────────────────────────────────────┐
│ 场景 1：执行器很快（无延迟）          │
│ • 每帧 ctrl 指令立即转化为关节运动    │
│ • 当前代码工作良好                    │
│ • 握力曲线平滑递增                    │
├──────────────────────────────────────┤
│ 场景 2：执行器有延迟（现实情况）      │
│ • ctrl 指令与实际关节位置有时差       │
│ • 如果 grasp_val 变化速度太快        │
│   可能导致执行器【饱和】（满力工作）   │
│ • 或导致【跟踪误差】（关节跟不上）     │
├──────────────────────────────────────┤
│ 场景 3：遇到阻力（抓住物体时）        │
│ • 执行器需要克服接触力                │
│ • 如果 ctrl 固定为某个值             │
│   关节可能停止运动（力达到最大值）     │
│ • 【开环控制】无法调整                │
└──────────────────────────────────────┘
```

**解决方案（3个等级）：**

#### **等级 1：验证当前实现（推荐，快速）**

```python
# 在执行阶段添加调试输出
def main():
    # ...
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # ...
            if planning_done and anim_time > grasp_start_time:
                grasp_progress = (anim_time - grasp_start_time) / grasp_duration
                current_grasp_val = final_grasp_synergy * grasp_progress
                
                # 【新增】调试：检查目标vs实际的关节角度
                temp_pose[7] = current_grasp_val
                temp_pose[8] = final_spread_synergy
                
                # 计算目标关节角度
                planner.set_hand_pose(temp_pose)
                target_flex_angle = current_grasp_val * 1.5
                actual_flex_angle = data.qpos[planner.flex_adrs[0]]  # 第一个屈肌关节
                
                print(f"t={anim_time:.2f}s | grasp={current_grasp_val:.2f} | "
                      f"target_flex={target_flex_angle:.3f} | actual={actual_flex_angle:.3f}")
```

#### **等级 2：使用 PID 控制（推荐，稳健）**

```python
class PIDController:
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target, actual, dt):
        error = target - actual
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

# 使用 PID
def qpos_to_ctrl_with_pid(model, planner, target_pose, pid_controllers):
    ctrl_cmd = np.zeros(model.nu)
    
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_adr = model.jnt_qposadr[jnt_id]
        
        # 计算目标关节角度（同前）
        if jnt_adr in planner.flex_adrs:
            target_angle = target_pose[7] * 1.5
        elif jnt_adr in planner.thumb_adrs:
            target_angle = target_pose[7] * 1.2
        elif jnt_adr in planner.abd_adrs:
            target_angle = target_pose[8] * 0.5
        else:
            target_angle = 0.0
        
        # 获取实际关节角度
        actual_angle = data.qpos[jnt_adr]
        
        # 使用 PID 计算控制命令
        pid_output = pid_controllers[i].update(target_angle, actual_angle, dt=0.001)
        
        # 裁剪到执行器范围
        ctrl_range = model.actuator_ctrlrange[i]
        ctrl_cmd[i] = np.clip(pid_output, ctrl_range[0], ctrl_range[1])
    
    return ctrl_cmd
```

#### **等级 3：使用 MuJoCo 内置 IK 求解器（高级）**

```python
# 使用 mjd_inverse() 求解逆动力学
# 确保执行器命令能够达成目标关节轨迹
# 需要在 XML 中配置 kinematic solver

mujoco.mj_inverse(model, data)  # 计算所需的关节力
```

---

### 问题 3：五指位姿的有效性

**你的疑问：** 规划阶段的 state[7] 和执行阶段 0-1 秒的 qpos[7]=0.0 是否矛盾？

**答案：✅ 完全合理，没有矛盾。**

**具体分析：**

```
规划阶段的作用：
├─ move() 中的 state[7] 扰动范围：[0.1, 0.9]
├─ 为什么要探索整个范围？
│  ├─ 手指弯曲会改变 reach space（可达范围）
│  ├─ 手指弯曲会产生碰撞检测
│  ├─ 能量函数会惩罚碰撞
│  └─ → 找到最优的【手掌位置+手指配置】组合
│
└─ 规划结果 best_pose[7] 的含义：
   "在这个最优手掌位置，最好的抓取握力是 X"

执行阶段 0-1 秒（演示动画）：
├─ 约束：【手指从张开逐渐闭合】
├─ 设定 qpos[7] = 0.0 的原因：
│  ├─ 美观：展示完整的抓取动作
│  ├─ 物理：手指张开时与瓶子接触最少（易于靠近）
│  └─ 动画：从 0 → 1 的线性插值直观
│
└─ 这【违反】了规划结果吗？
   → 不违反，因为我们不是【追踪】规划结果
   → 而是使用规划结果作为【最终状态】

执行阶段 1-5 秒（真正抓取）：
├─ current_grasp_val 从 0 → final_grasp_synergy
├─ 此时才使用规划出的握力值
├─ 手指按目标握力闭合
└─ ✓ 规划结果被【有效使用】
```

**时间轴对比：**

```
0-1 秒：qpos[7] = 0.0        (手指张开，靠近瓶子)
        best_pose[7] = 0.12  (被忽略，暂时不用)

1-5 秒：current_grasp_val 从 0 → 0.12  (手指逐渐闭合)
        使用规划结果：best_pose[7] = 0.12

✓ 没有矛盾，是【有序的阶段化控制】
```

---

## 📝 总结你的三个问题

| 问题 | 答案 | 理由 |
|------|------|------|
| move() 的 state[7] 扰动有效吗？ | ✅ 有效 | 用于探索搜索空间，找到最优手掌位置 |
| energy 应基于五指伸直还是弯曲？ | ✅ 弯曲 | 当前实现正确，基于 move() 后的 state[7] |
| 0-1秒与规划结果矛盾吗？ | ✅ 不矛盾 | 阶段化设计：0-1秒展示，1-5秒执行 |

---

## 🔧 执行器积分控制的建议

**如果你观察到的现象：**
- ✓ 手指闭合平滑，无抖动 → 当前实现足够
- ✗ 手指闭合有延迟或卡顿 → 考虑等级 1 调试
- ✗ 手指无法克服接触力 → 升级到等级 2（PID 控制）

**推荐方案：**
1. **先用等级 1** 验证现有控制是否有问题
2. 如果有问题，**升级到等级 2**（PID）
3. **不建议用等级 3**（除非需要高精度轨迹跟踪）

