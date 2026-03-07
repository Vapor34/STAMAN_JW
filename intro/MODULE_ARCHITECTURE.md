"""
===============================================================================
📋 项目模块架构详细梳理
===============================================================================

这是对 sdhand_grasp 项目核心模块的完整分析，阐明每个模块的职责、数据流和交互方式。

"""

# ✅ 1. StateStruct (core/hand_state.py) - 【状态表示层】
# ===============================================================================
"""
💡 核心职责：统一表示灵巧手的完整状态，提供多种访问方式

🎯 管理的状态信息 (13D状态向量)：
   ├─ 基座位置 (3D): x, y, z
   ├─ 基座姿态 (4D): 四元数 qw, qx, qy, qz
   └─ 协同变量 (6D): grasp, curl, spread, thumb_base, thumb_flex, wrist

📊 数据结构特点：
   ✓ 面向对象设计：提供 @property 属性访问 (e.g., state.x, state.grasp)
   ✓ 数组序列化：支持 13D numpy数组格式 (用于优化算法)
   ✓ 字典映射：state_dic 存储每个身体的位姿 (用于可视化/分析)

🔧 主要方法：
   - to_array() / from_array()       ➜ 13D数组转换 (规划器需要)
   - get/set_position/quaternion()   ➜ 位姿操作
   - get_state_dic()                 ➜ 构建身体状态快照
   - copy()                          ➜ 深拷贝 (保存多个规划候选)
   - @property (x, y, z, qw, qx, qy, qz, grasp, curl, ...) ➜ 单变量快速访问

💾 存储的MuJoCo引用：
   - self.model: 模型拓扑（不可变）
   - self.data: 仿真数据（随仿真更新）
   
⚠️  重要：StateStruct是"快照"而非"实时视图"
    - 创建时记录当前状态
    - 修改StateStruct不会改变MuJoCo仿真（需要通过HandControl应用）
"""


# ✅ 2. HandControl (core/hand_control.py) - 【执行控制层】
# ===============================================================================
"""
💡 核心职责：管理执行器映射和控制信号应用

🎯 两层结构映射：
   层1 (高级): 协同变量 (synergy) ──────┐
                                         ├─➜ 关节角度 ──➜ MuJoCo仿真
   层2 (低级): 执行器 (actuator) ────────┘

📊 主要数据结构：
   
   ① act_name_list (平面列表)
      └─ ['lh_WRJ2', 'lh_WRJ1', 'lh_THJ5', ..., 'lh_LFJ0']
      └─ 20个执行器的有序名称
   
   ② act_name_to_id / act_id_to_name (映射字典)
      └─ {'lh_WRJ2': 0, 'lh_WRJ1': 1, ...}
      └─ 用于名称 ↔ 索引转换
   
   ③ synergy_map (结构化分组)
      └─ {
           'grasp':      ['lh_FFJ3', 'lh_MFJ3', 'lh_RFJ3', 'lh_LFJ3', 'lh_LFJ5'],
           'curl':       ['lh_FFJ0', 'lh_MFJ0', 'lh_RFJ0', 'lh_LFJ0'],
           'spread':     ['lh_FFJ4', 'lh_MFJ4', 'lh_RFJ4', 'lh_LFJ4'],
           'thumb_base': ['lh_THJ5', 'lh_THJ4'],
           'thumb_flex': ['lh_THJ3', 'lh_THJ2', 'lh_THJ1'],
           'wrist':      ['lh_WRJ1', 'lh_WRJ2']
         }
      └─ 协同变量 ➜ 多个执行器的映射

🔧 主要方法：
   
   【低级执行器操作】
   - set_act_val(act_name, value)    ➜ 设置单个执行器
   - get_act_val(act_name)           ➜ 读取单个执行器
   - get_act_name(act_id)            ➜ 名称↔ID转换
   
   【高级协同操作】
   - set_syn_val(group, value)       ➜ 设置协同组 (一次控制多个执行器)
   - set_hand_state(state: StateStruct) ➜ 应用完整手部状态
   - get_synergy_map()               ➜ 获取协同映射
   
   【调试工具】
   - print_all_act_val()             ➜ 打印所有执行器值

💾 MuJoCo接口：
   - self.data.ctrl[act_id] = value  ➜ 直接写入MuJoCo控制信号
   
⚠️  重要：HandControl直接修改MuJoCo仿真状态
    - 每次set_*操作都会改变data.ctrl数组
    - 然后需要调用 mujoco.mj_step() 或 mujoco.mj_forward() 更新仿真
"""


# ✅ 3. StateStruct 与 HandControl 的关系
# ===============================================================================
"""
🔗 单向流动关系：

    StateStruct (状态表示)
         ↓ (通过set_hand_state方法)
    HandControl (执行控制)
         ↓ (写入data.ctrl和data.qpos)
    MuJoCo仿真 (物理计算)
         ↓ (mj_forward/mj_step)
    更新位置和力学特性


🔄 交互流程示例：

    step 1: 创建状态
    ────────────────────────────────────────
    state = StateStruct(model, data, 
                        position=[0.4, 0.4, 0.3],
                        quaternion=[1, 0, 0, 0],
                        grasp=0.5)
    
    step 2: 将状态应用到仿真
    ────────────────────────────────────────
    ctrl = HandControl(model, data)
    ctrl.set_hand_state(state)
    mujoco.mj_forward(model, data)  # 更新物理
    
    step 3: 读取仿真结果
    ────────────────────────────────────────
    new_state = StateStruct(model, data)  # 创建新快照
    print(new_state.position)  # 输出更新后的位置


⚠️  关键点：
    ✓ StateStruct 不修改仿真，只是表示状态
    ✓ HandControl 直接修改仿真 (通过data.ctrl和data.qpos)
    ✓ 必须在HandControl操作后调用mj_forward/mj_step
"""


# ✅ 4. 其他模块如何使用它们
# ===============================================================================
"""
📍 main.py (主程序)
════════════════════════════════════════════════════════════════

用途：协调整个仿真过程

典型流程：
    step 1: 初始化
    ────────────────────────────────────────
    model, data = mujoco_load("configs/scene_left.xml")
    ctrl = HandControl(model, data)
    
    step 2: 创建初始状态
    ────────────────────────────────────────
    initial_state = StateStruct(model, data, 
                                position=[0.4, 0.4, 0.3],
                                quaternion=[1, 0, 0, 0])
    
    step 3: 传给规划器
    ────────────────────────────────────────
    planner = GraspPlanner(initial_state.to_array(), 
                          model, data, 
                          bottle_body_name='bottle_body')
    best_pose, energy = planner.anneal()
    
    step 4: 应用规划结果
    ────────────────────────────────────────
    target_state = StateStruct(model, data)
    target_state.from_array(best_pose)  # 从规划结果恢复
    ctrl.set_hand_state(target_state)   # 应用到仿真
    mujoco.mj_step(model, data)
    
    step 5: 执行动画
    ────────────────────────────────────────
    while viewer.is_running():
        # 平滑过渡目标状态
        exec_state = StateStruct(model, data, ...)
        ctrl.set_hand_state(exec_state)
        mujoco.mj_step(model, data)
        viewer.sync()


📍 planning/grasp_planner.py (抓取规划器)
════════════════════════════════════════════════════════════════

用途：使用模拟退火算法搜索最优抓取姿态

典型流程：

    step 1: 初始化规划器
    ────────────────────────────────────────
    planner = GraspPlanner(
        state=initial_state.to_array(),  ← 13D数组
        model=model,
        data=data,
        bottle_body_name='bottle_body'
    )
    
    step 2: 内部状态操作 (move方法)
    ────────────────────────────────────────
    def move(self):
        current = StateStruct(self.model, self.data)
        current.from_array(self.state)       ← 从13D恢复状态
        current.position += noise            ← 修改状态
        current.grasp += noise
        self.state = current.to_array()      ← 转回13D
    
    step 3: 评估能量 (energy方法)
    ────────────────────────────────────────
    def energy(self):
        state_struct = StateStruct(self.model, self.data)
        state_struct.from_array(self.state)  ← 恢复状态
        self.set_hand_pose(state_struct)     ← 应用到仿真
        # ... 计算抓取质量指标
        return energy
    
    step 4: 规划器返回最优解
    ────────────────────────────────────────
    best_pose, energy = planner.anneal()
    # best_pose是13D数组，可以恢复到StateStruct


🔑 关键设计模式：

   【模式1】：状态序列化/反序列化
   ─────────────────────────────────────
   StateStruct ←→ 13D numpy array
   
   用途：
   - 规划器需要连续的数值向量用于优化
   - StateStruct提供高级接口（可读性好）
   - to_array/from_array转换两种表示
   
   
   【模式2】：状态快照
   ─────────────────────────────────────
   state_1 = StateStruct(model, data)  # 时刻T1的快照
   # ... 仿真运行 ...
   state_2 = StateStruct(model, data)  # 时刻T2的快照
   
   用途：
   - 记录不同时刻的状态
   - 支持规划算法中的多个候选保存
   - 不会互相影响（copy()支持深拷贝）
   
   
   【模式3】：控制管道
   ─────────────────────────────────────
   state (13D或StateStruct)
     ↓ (通过HandControl应用)
   data.ctrl[], data.qpos[]  (MuJoCo内部)
     ↓ (通过mj_forward/mj_step)
   物理仿真更新
     ↓ (通过StateStruct读取)
   新state (快照)
   
   用途：
   - 清晰的单向控制流
   - 每层职责明确
   - 便于调试和扩展
"""


# ✅ 5. 实际代码示例
# ===============================================================================
"""

【例1】最简单的使用：设置手部并拍一个快照

    from core.hand_state import StateStruct
    from core.hand_control import HandControl
    from core.simulator import mujoco_load
    import mujoco
    
    model, data = mujoco_load("configs/scene_left.xml")
    ctrl = HandControl(model, data)
    
    # 创建目标状态
    target = StateStruct(model, data,
                        position=[0.5, 0.5, 0.5],
                        grasp=0.8)
    
    # 应用到仿真
    ctrl.set_hand_state(target)
    mujoco.mj_forward(model, data)
    
    # 读取仿真结果
    result = StateStruct(model, data)
    print(f"Hand position: {result.position}")


【例2】规划器中的使用方式

    def energy(self):
        # 1. 从13D恢复状态对象
        state = StateStruct(self.model, self.data)
        state.from_array(self.state)  # self.state是13D
        
        # 2. 应用状态到仿真
        self.set_hand_pose(state)
        
        # 3. 评估抓取质量
        distance = compute_distance(state)
        collision = check_collision()
        
        return distance + collision


【例3】复杂的时间序列控制

    initial = StateStruct(model, data, position=[0.3, 0.3, 0.3])
    target = StateStruct(model, data, position=[0.5, 0.5, 0.5])
    
    for t in np.linspace(0, 1, 100):
        # 线性插值
        current_pos = (1-t) * initial.position + t * target.position
        current_grasp = (1-t) * initial.grasp + t * target.grasp
        
        # 创建中间状态
        current = StateStruct(model, data,
                             position=current_pos,
                             grasp=current_grasp)
        
        # 应用
        ctrl.set_hand_state(current)
        mujoco.mj_step(model, data)


【例4】独立执行器控制

    ctrl = HandControl(model, data)
    
    # 方式1：通过协同变量 (推荐，易读)
    ctrl.set_syn_val('grasp', 0.8)    # 闭合所有手指
    ctrl.set_syn_val('spread', 0.2)   # 展开
    
    # 方式2：单个执行器 (细粒度)
    ctrl.set_act_val('lh_THJ1', 0.5)  # 拇指第一个关节
    
    # 方式3：完整状态 (最方便)
    state = StateStruct(model, data, grasp=0.8, spread=0.2)
    ctrl.set_hand_state(state)

"""


# ✅ 总结架构图
# ===============================================================================
"""

┌─────────────────────────────────────────────────────────────────────┐
│                    【项目模块架构总览】                              │
└─────────────────────────────────────────────────────────────────────┘

                          main.py (协调器)
                              │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    simulator.py         hand_state.py      hand_control.py
    (加载模型)    (状态表示13D)      (执行器控制)
          │                    │                    │
          │         ┌──────────┴────────────┐       │
          │         │ 13D数组 ↔ 对象转换  │       │
          │         │ (to_array/from_array) │       │
          │         └──────────┬────────────┘       │
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                      planning/grasp_planner.py
                     (模拟退火搜索最优解)
                               │
                               ▼
                        MuJoCo仿真引擎
                         (物理计算)


【数据流向】：

    创建 StateStruct (位置、四元数、协同变量)
           │
           ├─➜ 序列化为13D数组
           │        │
           │        ├─➜ 传给规划器
           │        │   • move(): 随机扰动
           │        │   • energy(): 评估质量
           │        └─➜ 返回最优13D
           │
           ├─➜ 反序列化为 StateStruct
           │
           ├─➜ 通过 HandControl.set_hand_state() 应用
           │        │
           │        └─➜ 写入 data.ctrl[], data.qpos[]
           │
           └─➜ 调用 mj_forward()/mj_step() 更新
                    │
                    └─➜ 创建新 StateStruct 读取结果


【设计原则】：

    ✓ 单一职责：
      - StateStruct: 状态表示
      - HandControl: 执行器控制
      - GraspPlanner: 优化算法
      
    ✓ 清晰接口：
      - 高级API (协同变量)
      - 低级API (单执行器)
      - 序列化API (13D数组)
      
    ✓ 易于测试：
      - 各层可独立验证
      - 状态快照便于调试
      - 控制信号明确

"""
