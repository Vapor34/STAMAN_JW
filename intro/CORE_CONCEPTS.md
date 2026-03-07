"""
===============================================================================
🎯 【核心答案】StateStruct vs HandControl 梳理
===============================================================================
"""

# 三句话总结
# ═══════════════════════════════════════════════════════════════════════════

"""

💡 核心概念：

1️⃣  StateStruct (hand_state.py) - 【你想怎样】
    ───────────────────────────────────
    是灵巧手状态的高级表示，用13维向量(位置、方向、协同变量)来描述一个姿态。
    不修改MuJoCo仿真，只是表示和存储状态，支持高级属性访问和13D数组序列化。
    
    关键词：表示、声明、快照、高级接口


2️⃣  HandControl (hand_control.py) - 【怎样去做】
    ─────────────────────────────────────
    是执行器控制的底层接口，管理20个执行器到6个协同变量的映射关系。
    直接修改MuJoCo的data.ctrl数组应用控制信号，支持多层级操作(执行器/协同/完整状态)。
    
    关键词：执行、命令、控制流、低级接口


3️⃣  它们的关系 - 【流程】
    ──────────────────────
    StateStruct 定义"目标是什么" → HandControl 执行"怎样达到" → MuJoCo 计算"结果是什么"
    
    state.grasp = 0.8 (StateStruct) 
         ↓ 通过HandControl.set_hand_state()
    data.ctrl[7,10,13,16,2] = 0.8 (HandControl控制5个执行器)
         ↓ 调用mj_forward()
    物理计算更新手指位置


═════════════════════════════════════════════════════════════════════════════
"""


# 一个表格说明所有
# ═══════════════════════════════════════════════════════════════════════════

"""

┌──────────────────────────────────────────────────────────────────────────────┐
│                         StateStruct ↔ HandControl 交互完全指南               │
└──────────────────────────────────────────────────────────────────────────────┘

【StateStruct的职责】- 状态表示
─────────────────────────────────────────────────────────────────────────────

什么是StateStruct?
  
  13维向量的面向对象包装:
  
  ┌─────────────────────────────────┐
  │ position (3D)  :  x, y, z       │
  │ quaternion(4D) : qw, qx, qy, qz │  ← 手掌位姿
  ├─────────────────────────────────┤
  │ grasp (1D)          ← 闭合手指   │
  │ curl (1D)           ← 弯曲      │
  │ spread (1D)         ← 展开      │  ← 协同变量
  │ thumb_base (1D)     ← 拇指基部  │
  │ thumb_flex (1D)     ← 拇指弯曲  │
  │ wrist (1D)          ← 手腕      │
  └─────────────────────────────────┘


StateStruct能做什么?

  ✓ 创建和初始化:
    state = StateStruct(model, data, 
                        position=[0.4, 0.4, 0.3],
                        quaternion=[1, 0, 0, 0],
                        grasp=0.5)
  
  ✓ 属性访问 (@property):
    x = state.x              # 获取X坐标
    state.y = 0.5            # 设置Y坐标
    state.grasp = 0.8        # 设置抓取值
    quat = state.quaternion  # 获取完整四元数
  
  ✓ 数组序列化:
    array = state.to_array()     # 转为13D numpy数组
    state.from_array(array)      # 从13D数组恢复
  
  ✓ 快照和复制:
    state_copy = state.copy()    # 深拷贝
  
  ✓ 状态查询:
    bodies = state.state_dic     # 所有body的位置字典


StateStruct做不了什么?

  ❌ 修改MuJoCo仿真
     state.grasp = 0.9  (只改对象，仿真不变!)
  
  ❌ 直接控制执行器
     state不知道如何映射到data.ctrl[]
  
  ❌ 应用物理计算
     state不调用mj_forward()


【HandControl的职责】- 执行控制
─────────────────────────────────────────────────────────────────────────────

什么是HandControl?

  执行器映射管理器 + 控制信号应用器:
  
  ┌─ 20个执行器 ◄─ act_name_list ◄─ 硬编码名称列表
  │   (lh_WRJ2, lh_WRJ1, ..., lh_LFJ0)
  │   
  ├─ 名称↔ID映射 ◄─ act_name_to_id, act_id_to_name
  │   {'lh_WRJ2': 0, 'lh_WRJ1': 1, ...}
  │
  └─ 协同组织 ◄─ synergy_map
      {
        'grasp': [lh_FFJ3, lh_MFJ3, lh_RFJ3, lh_LFJ3, lh_LFJ5],
        'curl':  [lh_FFJ0, lh_MFJ0, lh_RFJ0, lh_LFJ0],
        'spread': [lh_FFJ4, lh_MFJ4, lh_RFJ4, lh_LFJ4],
        ...
      }


HandControl能做什么?

  ✓ 创建和初始化:
    ctrl = HandControl(model, data)
  
  ✓ 低级操作 (单执行器):
    ctrl.set_act_val('lh_THJ1', 0.5)   # 设置拇指关节
    value = ctrl.get_act_val('lh_THJ1')  # 读取值
  
  ✓ 中级操作 (协同组):
    ctrl.set_syn_val('grasp', 0.8)     # 同时控制5个执行器
    ctrl.set_syn_val('spread', 0.2)
  
  ✓ 高级操作 (完整状态):
    state = StateStruct(model, data, grasp=0.8, spread=0.2)
    ctrl.set_hand_state(state)  ← 最方便!
  
  ✓ 查询和调试:
    synergy_map = ctrl.get_synergy_map()
    ctrl.print_all_act_val()  ← 打印所有执行器值
  
  ✓ 直接修改MuJoCo:
    data.ctrl[0] = 0.5  ← HandControl在set_*中做的事


HandControl做不了什么?

  ❌ 序列化状态
     HandControl没有to_array/from_array
  
  ❌ 高级属性访问
     ctrl.grasp = 0.5  (不支持)
  
  ❌ 物理计算
     HandControl只设置data.ctrl，不调用mj_forward()


【两者的交互方式】
─────────────────────────────────────────────────────────────────────────────

方式1: StateStruct → HandControl (正向流)
  
  step 1: 定义状态
  state = StateStruct(model, data, grasp=0.8, curl=0.3)
  
  step 2: 应用控制
  ctrl = HandControl(model, data)
  ctrl.set_hand_state(state)
          ↓
  HandControl内部:
    for group, acts in synergy_map.items():
        if group == 'grasp':
            for act in acts:
                data.ctrl[ID] = state.grasp_synergy
  
  step 3: 更新物理
  mujoco.mj_forward(model, data)
  
  step 4: 读取结果
  result = StateStruct(model, data)  ← 创建新快照


方式2: 13D数组 → StateStruct → HandControl (规划流)
  
  step 1: 规划器内部状态 (13D数组)
  self.state = [x, y, z, qw, qx, qy, qz, g, c, s, tb, tf, w]
  
  step 2: 转为StateStruct便于操作
  current = StateStruct(model, data)
  current.from_array(self.state)
  
  step 3: 应用到仿真
  self.set_hand_pose(current)  ← 内部调用HandControl.set_hand_state()
  
  step 4: 评估能量
  energy = compute_quality()
  
  step 5: 转回13D继续规划
  self.state = current.to_array()


方式3: 独立执行器控制 (低级操作)
  
  ctrl = HandControl(model, data)
  ctrl.set_act_val('lh_THJ1', 0.3)  ← 直接设置执行器
  mujoco.mj_forward(model, data)


【真实使用示例】
─────────────────────────────────────────────────────────────────────────────

场景1: 简单的"设置并保持"
┌───────────────────────────────┐
│ state = StateStruct(...)      │  创建状态
│ ctrl.set_hand_state(state)    │  应用控制
│ mujoco.mj_forward(model, data)│  物理计算
└───────────────────────────────┘


场景2: 规划 → 应用 → 展示
┌─────────────────────────────────────────┐
│ 规划:                                    │
│   planner = GraspPlanner(...)           │
│   best_pose, _ = planner.anneal()       │  返回13D
│                                         │
│ 应用:                                    │
│   target = StateStruct(model, data)     │
│   target.from_array(best_pose)          │  恢复为State
│   ctrl.set_hand_state(target)           │
│   mujoco.mj_forward(model, data)        │
│                                         │
│ 展示:                                    │
│   result = StateStruct(model, data)     │  读取结果
│   viewer.sync()                         │
└─────────────────────────────────────────┘


场景3: 平滑过渡
┌─────────────────────────────────────────┐
│ for t in np.linspace(0, 1, 100):        │
│   # 线性插值两个状态                     │
│   current_grasp = start.grasp*(1-t)     │
│                + end.grasp*t            │
│                                         │
│   # 创建中间状态                         │
│   mid = StateStruct(model, data,        │
│          grasp=current_grasp)           │
│                                         │
│   # 应用控制                             │
│   ctrl.set_hand_state(mid)              │
│   mujoco.mj_step(model, data)           │
└─────────────────────────────────────────┘


【常见操作速查】
─────────────────────────────────────────────────────────────────────────────

需求                              代码示例
──────────────────────────────────────────────────────────────────────────

保存当前状态                      state = StateStruct(model, data)

创建指定状态                      s = StateStruct(model, data,
                                     position=[x,y,z],
                                     grasp=0.8)

修改单个变量                      s.grasp = 0.9
                                state.x = 0.5

获取完整状态值                    array = state.to_array()

恢复状态从数组                    state.from_array(array_13d)

深拷贝状态                        s_copy = state.copy()

读取所有body位置                  bodies = state.state_dic

应用完整状态                      ctrl.set_hand_state(state)
                                mujoco.mj_forward(model, data)

控制单个执行器                    ctrl.set_act_val('lh_THJ1', 0.5)
                                mujoco.mj_forward(model, data)

控制协同组                        ctrl.set_syn_val('grasp', 0.8)
                                mujoco.mj_forward(model, data)

查看执行器映射                    synergy = ctrl.get_synergy_map()
                                names = ctrl.get_act_name_list()

调试执行器值                      ctrl.print_all_act_val()

获取协同组执行器列表              acts = ctrl.synergy_map['grasp']


【关键设计模式】
─────────────────────────────────────────────────────────────────────────────

模式1: 状态快照 (State Snapshot)
  
  问题: 需要保存多个不同时刻的状态
  解决: 创建多个StateStruct实例
  
  s1 = StateStruct(model, data)  # 时刻1
  # ... 仿真运行 ...
  s2 = StateStruct(model, data)  # 时刻2
  # s1和s2互不影响，各自独立存储


模式2: 状态序列化 (State Serialization)
  
  问题: 规划器需要13D数值向量
  解决: 提供to_array/from_array
  
  array = state.to_array()       # 序列化
  state.from_array(array)        # 反序列化
  # 两端无损转换


模式3: 多层API (Multi-level API)
  
  问题: 既要易用又要灵活
  解决: 提供不同抽象级别
  
  # 最高层: 完整状态
  ctrl.set_hand_state(state)
  
  # 中层: 协同变量
  ctrl.set_syn_val('grasp', 0.8)
  
  # 底层: 单执行器
  ctrl.set_act_val('lh_THJ1', 0.5)


模式4: 控制管道 (Control Pipeline)
  
  问题: 清晰的控制流程
  解决: 定义明确的数据流向
  
  State (定义) 
    → HandControl (应用)
    → data.ctrl[] (低级)
    → mj_forward() (物理)
    → 新State (读取)


【重点理解】
─────────────────────────────────────────────────────────────────────────────

1. StateStruct是"声明式"（declarative）
   表达"我想要什么"，不表达"怎样做"

2. HandControl是"命令式"（imperative）
   表达"怎样执行"具体的控制操作

3. 协同变量是"高层抽象"
   一个协同变量映射到多个执行器
   例: grasp=0.8 ➜ 5个执行器都设为0.8

4. 13D数组是"中间表示"
   - 规划器需要连续数值
   - StateStruct提供高级接口
   - to/from_array转换两种形式

5. MuJoCo的data.ctrl[]是"最终命令"
   - HandControl最终修改这个数组
   - mj_forward()根据ctrl计算物理
   - 没有mj_forward()就没有物理更新


【陷阱和最佳实践】
─────────────────────────────────────────────────────────────────────────────

❌ 陷阱1: 忘记mj_forward()
  ctrl.set_hand_state(state)
  result = StateStruct(model, data)  ← positions还没更新!

✅ 正确:
  ctrl.set_hand_state(state)
  mujoco.mj_forward(model, data)  ← 必须调用
  result = StateStruct(model, data)  ← 现在有更新


❌ 陷阱2: 混淆StateStruct.position和state_dic
  s = StateStruct(model, data, position=[1,2,3])
  print(s.position)           # [1,2,3] (参数值)
  print(s.state_dic['palm'])  # [?,?,?] (MuJoCo值，可能不同!)

✅ 正确: 理解两者的含义
  - position: 显式初始化的值
  - state_dic: 从MuJoCo读取的当前值


❌ 陷阱3: 期望修改State会改仿真
  state.grasp = 0.9
  # 这只改了对象，仿真不变

✅ 正确:
  state = StateStruct(model, data, grasp=0.9)
  ctrl.set_hand_state(state)  ← 才会改仿真


✅ 最佳实践:
  1. 使用set_hand_state()应用完整状态（最常用）
  2. 只在需要单个关节时使用set_act_val()
  3. 总是在修改后调用mj_forward()或mj_step()
  4. 创建新StateStruct来读取最新结果
  5. 使用copy()保存不同时刻的状态

"""
