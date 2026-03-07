"""
===============================================================================
🔗 StateStruct 与 HandControl 交互关系详解
===============================================================================
"""

# 一图胜千言：数据流转过程
# ═══════════════════════════════════════════════════════════════════════════

"""

【场景1】：初始化手部状态

    代码:
    ─────────────────────────────────────────
    state = StateStruct(
        model, data,
        position=[0.4, 0.4, 0.3],          ← (1) 创建状态快照
        quaternion=[1, 0, 0, 0],
        grasp=0.5,
        curl=0.3
    )
    
    
    数据流:
    ─────────────────────────────────────────
    状态对象内存:
    ┌────────────────────┐
    │ StateStruct        │
    ├────────────────────┤
    │ position: [0.4,...]│ ← 独立保存
    │ quaternion: [1,0..]│   (不修改MuJoCo)
    │ grasp_synergy: 0.5 │
    │ curl_synergy: 0.3  │
    │ ...                │
    └────────────────────┘
    
    MuJoCo仿真状态: (不变)
    ┌────────────────────┐
    │ data.qpos[0:3]     │ ← 保持前一时刻值
    │ data.qpos[3:7]     │
    │ data.ctrl[:]       │
    └────────────────────┘


【场景2】：应用状态到仿真

    代码:
    ─────────────────────────────────────────
    ctrl = HandControl(model, data)         ← (1) 创建控制器
    ctrl.set_hand_state(state)              ← (2) 应用状态
    mujoco.mj_forward(model, data)          ← (3) 物理更新
    
    
    StateStruct内部流程:
    ─────────────────────────────────────────
    set_hand_state(state: StateStruct):
        ├─ set_syn_val('grasp', 0.5)       ← 遍历synergy_map
        │  ├─ act_names = ['lh_FFJ3', 'lh_MFJ3', ...]
        │  └─ for act in act_names:
        │      └─ set_act_val(act, 0.5)
        │          └─ data.ctrl[ID] = 0.5  ◄─ 直接写MuJoCo!
        │
        ├─ set_syn_val('curl', 0.3)        ← 继续...
        ├─ set_syn_val('spread', ...)
        ├─ ... (其他协同变量)
        │
    
    MuJoCo仿真状态: (已修改!)
    ┌────────────────────────────┐
    │ data.ctrl[7] = 0.5         │ ← lh_FFJ3
    │ data.ctrl[10] = 0.5        │ ← lh_MFJ3
    │ data.ctrl[13] = 0.5        │ ← lh_RFJ3
    │ ... (所有关联的执行器)      │
    │ data.ctrl[20] = 0.3        │ ← curl值
    │ ...                        │
    └────────────────────────────┘
    
    
    调用mj_forward():
    ─────────────────────────────────────────
    MuJoCo物理计算:
    
    data.ctrl[] ──➜ 关节角度 ──➜ 体链前向运动学 ──➜ data.xpos[]
                                              ├─➜ data.xquat[]
                                              ├─➜ data.xmat[]
                                              └─➜ ...


【场景3】：读取仿真结果

    代码:
    ─────────────────────────────────────────
    result = StateStruct(model, data)       ← 创建新快照
    
    
    构造过程:
    ─────────────────────────────────────────
    StateStruct.__init__():
        ├─ self.model = model              ← 保存引用
        ├─ self.data = data                ← 当前时刻的仿真数据
        ├─ self.position = ...             ← 初始值参数 (或None用默认)
        ├─ self.grasp_synergy = ...        ← 从参数设置
        └─ self.get_state_dic()            ← 读取model/data构建body状态字典
            └─ for each body in model:
                ├─ pos = data.xpos[body_id]   ◄─ 从MuJoCo读取
                ├─ quat = data.xquat[body_id] ◄─ 从MuJoCo读取
                └─ state_dic[body_name] = [pos, quat]
    
    
    对比:
    ─────────────────────────────────────────
    时刻T1的快照 (initial):
    ┌──────────────────────────┐
    │ StateStruct.position     │ ← 可能是参数传入
    │ StateStruct.state_dic    │ ← 包含所有body的当前位置
    │   lh_palm: [x,y,z,...]  │
    │   lh_ff_base: [x,y,z,...]│
    │   ...                    │
    └──────────────────────────┘
    
    时刻T2的快照 (after physics):
    ┌──────────────────────────┐
    │ StateStruct.position     │ ← 初始化时的值 (参数)
    │ StateStruct.state_dic    │ ← 更新后的所有body位置
    │   lh_palm: [x',y',z',..]│ ◄─ 物理计算后的新位置
    │   lh_ff_base: [x',y',z'.]│
    │   ...                    │
    └──────────────────────────┘
    
    ⚠️  注意：position和state_dic可能不同步
        - position: 显式设置的值
        - state_dic: 从MuJoCo实时读取


【场景4】：规划器中的使用

    代码:
    ─────────────────────────────────────────
    planner = GraspPlanner(
        state=initial_state.to_array(),    ← (1) 13D数组
        model=model,
        data=data
    )
    best_pose, energy = planner.anneal()   ← (2) 搜索
    
    
    规划器内部流程 (迭代):
    ─────────────────────────────────────────
    
    iteration i:
    
    step 1: move() - 扰动当前解
    ────────────────────────────────────
    self.state: [x, y, z, qw, qx, qy, qz, g, c, s, tb, tf, w]
        │
        ├─ 转为StateStruct
        │  current = StateStruct(model, data)
        │  current.from_array(self.state)
        │       │
        │       ├─ current.position = array[0:3]
        │       ├─ current.quaternion = array[3:7]
        │       ├─ current.grasp = clip(array[7], 0, 1)
        │       └─ ...
        │
        ├─ 加噪声
        │  current.position += random_noise()
        │  current.grasp += random_noise()
        │
        └─ 转回13D
           self.state = current.to_array()
    
    step 2: energy() - 评估新解
    ────────────────────────────────────
    self.state: [x', y', z', ..., g', ...]
        │
        ├─ 转为StateStruct
        │  state = StateStruct(model, data)
        │  state.from_array(self.state)
        │
        ├─ 应用到仿真
        │  self.set_hand_pose(state)
        │  │
        │  ├─ data.qpos[0:3] = state.position
        │  ├─ data.qpos[3:7] = state.quaternion
        │  ├─ ctrl.set_hand_state(state)
        │  │   └─ data.ctrl[:] = ...
        │  └─ mujoco.mj_forward(model, data)
        │
        ├─ 评估质量
        │  distance = compute_contact_distance()
        │  collision = check_penetration()
        │  energy = distance * 100 + collision * 50
        │
        └─ 返回 energy
    
    step 3: 比较和决定
    ────────────────────────────────────
    if new_energy < current_energy:
        accept = True
    else:
        accept = random() < exp(-ΔE / T)
    
    step 4: 更新温度和继续
    ────────────────────────────────────
    T *= cooling_rate
    
    
    最终结果:
    ───────────────────────────────────
    best_pose: 13D数组  ← 最优的[x, y, z, qw, qx, qy, qz, g, c, s, tb, tf, w]
    energy: float       ← 对应的能量值


【场景5】：完整工作流

    ┌─────────────────────────────────────────────────────────┐
    │  main.py                                                │
    └─────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 加载模型                                              │
    │    model, data = mujoco_load(...)                       │
    └─────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 2. 创建初始状态                                          │
    │    initial = StateStruct(model, data, ...)              │
    │    ↓ (13D)                                              │
    │    planner = GraspPlanner(initial.to_array(), ...)      │
    └─────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 3. 规划器搜索                                            │
    │    best_pose, energy = planner.anneal()                 │
    │    ↓ (13D数组)                                          │
    │    target.from_array(best_pose)  ← 恢复为StateStruct   │
    └─────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 4. 应用控制                                              │
    │    ctrl = HandControl(model, data)                      │
    │    ctrl.set_hand_state(target)  ← 应用状态              │
    │    mujoco.mj_step(model, data)  ← 物理更新              │
    └─────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 5. 动画展示                                              │
    │    result = StateStruct(model, data)  ← 读取结果         │
    │    viewer.sync()                                        │
    └─────────────────────────────────────────────────────────┘


【类和方法的职责划分】

    StateStruct:
    ───────────────────────────────────────
    "我是状态的高级表示"
    
    ✓ 提供直观的属性访问 (state.x, state.grasp)
    ✓ 支持序列化/反序列化 (to_array/from_array)
    ✓ 管理position、quaternion、synergy variables
    ✗ 不修改MuJoCo仿真
    ✗ 不涉及执行器细节


    HandControl:
    ───────────────────────────────────────
    "我是执行的底层接口"
    
    ✓ 管理执行器映射 (名称↔ID)
    ✓ 提供协同变量控制 (synergy_map)
    ✓ 直接修改data.ctrl (写入MuJoCo)
    ✗ 不处理高级状态表示
    ✗ 不涉及物理计算


    GraspPlanner:
    ───────────────────────────────────────
    "我是优化算法"
    
    ✓ 使用13D数组表示状态 (for numpy operations)
    ✓ 调用StateStruct进行high-level操作
    ✓ 调用HandControl进行low-level操作
    ✓ 评估能量函数
    ✗ 不存储永久状态 (各iteration独立)


【常见的错误使用】

    ❌ 错误1：期望修改StateStruct会改变仿真
    ─────────────────────────────────────────
    state = StateStruct(model, data)
    state.grasp = 0.8
    result = StateStruct(model, data)
    print(result.grasp)  # 还是原值！data.ctrl没变
    
    ✓ 正确做法：
    state = StateStruct(model, data, grasp=0.8)
    ctrl = HandControl(model, data)
    ctrl.set_hand_state(state)
    mujoco.mj_forward(model, data)
    result = StateStruct(model, data)


    ❌ 错误2：StateStruct和实际仿真不同步
    ─────────────────────────────────────────
    state = StateStruct(model, data, position=[1, 2, 3])
    # ... 仿真运行 ...
    print(state.position)  # [1, 2, 3] (原值，不是当前仿真位置!)
    
    ✓ 正确做法：
    current = StateStruct(model, data)  # 创建新快照
    print(current.state_dic['lh_palm'])  # 读取当前真实位置


    ❌ 错误3：忘记mj_forward()
    ─────────────────────────────────────────
    ctrl.set_hand_state(state)  # data.ctrl改了
    result = StateStruct(model, data)  # 但positions没更新
    
    ✓ 正确做法：
    ctrl.set_hand_state(state)
    mujoco.mj_forward(model, data)  # ← 必须调用
    result = StateStruct(model, data)


【总结】

    设计目标：分离"状态表示"和"执行控制"

    StateStruct  → 声明性 (what)
    HandControl  → 命令式 (how)
    
    State定义了目标状态是什么
    Control定义了如何把它应用到仿真
    
    这样设计的好处：
    ✓ 关注点分离 (SoC)
    ✓ 可独立测试
    ✓ 易于扩展 (新协同变量, 新执行器)
    ✓ 代码清晰

"""
