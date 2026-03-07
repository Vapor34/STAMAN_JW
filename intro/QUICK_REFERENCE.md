"""
===============================================================================
📊 模块对比和总结表
===============================================================================
"""

# 模块功能对比表
# ═══════════════════════════════════════════════════════════════════════════

"""

┌─────────────────┬──────────────────────┬──────────────────────┬────────────────────┐
│   特性          │   StateStruct        │   HandControl        │   GraspPlanner     │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 定位            │ 状态表示层           │ 执行控制层           │ 优化算法层         │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 主要职责        │ 统一表示手部状态     │ 执行器映射和控制     │ 搜索最优手部姿态   │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 数据结构        │ 13个标量 (13D)       │ 20个执行器+映射表    │ 不存储状态(迭代)   │
│                 │ - position (3D)      │ - act_name_list      │ - self.state 13D   │
│                 │ - quaternion (4D)    │ - act_name_to_id     │ - act_ctrl引用     │
│                 │ - 6个协同变量 (6D)   │ - synergy_map        │ - 多个geometry     │
│                 │ - state_dic (bodies) │ - joint_range        │ - energy计数器     │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 提供的接口      │ 【高级】             │ 【多层级】           │ 【Annealer协议】   │
│                 │ - @property x/y/z    │ 低级:                │ - move()           │
│                 │ - @property qw/x/y/z │ - set_act_val()      │ - energy()         │
│                 │ - @property grasp..  │ - get_act_val()      │ - anneal()         │
│                 │ - to_array()         │ 中级:                │ (继承自Annealer)   │
│                 │ - from_array()       │ - set_syn_val()      │                    │
│                 │ - copy()             │ 高级:                │                    │
│                 │ - get_state_dic()    │ - set_hand_state()   │                    │
│                 │                      │ - print_all_act_val()│                    │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 修改MuJoCo     │ ❌ 不修改            │ ✅ 直接修改          │ ✅ 通过control修改 │
│ 仿真            │    (只是快照)        │    (data.ctrl等)     │    (间接)          │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 是否有状态      │ ✅ 有状态对象        │ ✅ 有持久化对象      │ ❌ 无持久状态      │
│ 持久化          │    (独立副本)        │    (引用model/data)  │    (各iteration)   │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 序列化方式      │ ✅ 13D数组           │ ❌ 不支持            │ ✅ 13D数组(内部)   │
│                 │    (to/from_array)   │    序列化            │                    │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 典型创建方式    │ StateStruct(         │ HandControl(         │ GraspPlanner(      │
│                 │   model, data,       │   model, data)       │   state_13d,       │
│                 │   position=[...],    │                      │   model, data,     │
│                 │   grasp=0.5)         │                      │   bottle_body)     │
├─────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
│ 典型使用场景    │ 【1】存储状态        │ 【1】应用状态到仿真  │ 【1】搜索最优解    │
│                 │ 【2】修改状态        │ 【2】细粒度控制      │ 【2】评估状态质量  │
│                 │ 【3】读取仿真结果    │ 【3】查询执行器      │                    │
│                 │ 【4】数据序列化      │ 【4】协同变量映射    │                    │
│                 │ 【5】状态插值        │                      │                    │
└─────────────────┴──────────────────────┴──────────────────────┴────────────────────┘


【协同变量映射示例】

State (高级 - 易读):
┌─────────────────────────────────────┐
│ grasp       = 0.8  ◄─ 闭合手指       │
│ curl        = 0.3  ◄─ 弯曲程度      │
│ spread      = 0.1  ◄─ 展开程度      │
│ thumb_base  = 0.5                  │
│ thumb_flex  = 0.6                  │
│ wrist       = 0.2                  │
└─────────────────────────────────────┘
         │ (via synergy_map)
         ▼
Control (低级 - 细粒度):
┌─────────────────────────────────────┐
│ ctrl[7]  = 0.8  (lh_FFJ3)          │
│ ctrl[10] = 0.8  (lh_MFJ3)  ◄─ grasp │
│ ctrl[13] = 0.8  (lh_RFJ3)          │
│ ctrl[16] = 0.8  (lh_LFJ3)          │
│ ctrl[2]  = 0.8  (lh_LFJ5)          │
│                                     │
│ ctrl[9]  = 0.3  (lh_FFJ0)          │
│ ctrl[12] = 0.3  (lh_MFJ0)  ◄─ curl │
│ ctrl[15] = 0.3  (lh_RFJ0)          │
│ ctrl[19] = 0.3  (lh_LFJ0)          │
│                                     │
│ ctrl[8]  = 0.1  (lh_FFJ4)          │
│ ctrl[11] = 0.1  (lh_MFJ4)  ◄─ spread
│ ctrl[14] = 0.1  (lh_RFJ4)          │
│ ctrl[17] = 0.1  (lh_LFJ4)          │
│                                     │
│ ... (其他协同变量映射) ...          │
└─────────────────────────────────────┘


【核心数据流向】

初始化:
state_1 = StateStruct(model, data, ...)  ← 创建快照
                     │
                     └─ .to_array() ──────→ [x, y, z, qw, qx, qy, qz, g, c, s, tb, tf, w]
                                              │
                                              └─ GraspPlanner.state ──┐
                                                                       │
应用:                                                                  │
state_2 = StateStruct(model, data, ...)  ← .from_array(best_pose) ◄──┘
   │
   └─ HandControl.set_hand_state(state_2)
      │
      └─ data.ctrl[:] = ...
      └─ data.qpos[:] = ...
      └─ mujoco.mj_forward() ──→ 物理计算 ──→ data.xpos[], data.xquat[] 更新
                                                 │
                                                 └─ StateStruct(model, data) ← 读取


【使用流程示例】

【简单场景】：设置一个静止姿态
───────────────────────────────────────

step 1: 定义目标状态
  state = StateStruct(model, data, 
                      position=[0.5, 0.5, 0.5],
                      grasp=0.8)

step 2: 应用控制
  ctrl = HandControl(model, data)
  ctrl.set_hand_state(state)
  
step 3: 更新物理
  mujoco.mj_forward(model, data)
  
step 4: 读取结果
  result = StateStruct(model, data)


【复杂场景】：规划 → 应用 → 动画
───────────────────────────────────────

step 1: 规划最优抓取
  initial = StateStruct(model, data, ...)
  planner = GraspPlanner(initial.to_array(), ...)
  best_pose, energy = planner.anneal()

step 2: 恢复最优状态
  target = StateStruct(model, data)
  target.from_array(best_pose)

step 3: 平滑过渡并应用
  ctrl = HandControl(model, data)
  for t in range(0, 100):
      progress = t / 100.0
      current = interpolate(initial, target, progress)
      ctrl.set_hand_state(current)
      mujoco.mj_step(model, data)


【main.py中的使用】
───────────────────────────────────────

主程序流程:

1. 加载模型和资源
   model, data = mujoco_load(args.model_path)

2. 创建控制器
   ctrl = HandControl(model, data)

3. 创建初始状态
   initial_state = StateStruct(model, data, ...)

4. 创建规划器并规划
   planner = GraspPlanner(initial_state.to_array(), ...)
   best_pose, energy = planner.anneal()

5. 应用规划结果
   target_state = StateStruct(model, data)
   target_state.from_array(best_pose)

6. 动画展示
   with mujoco.viewer.launch_passive(model, data) as viewer:
       while viewer.is_running():
           ctrl.set_hand_state(target_state)  ← 应用状态
           mujoco.mj_step(model, data)        ← 物理更新
           viewer.sync()


【planning/grasp_planner.py中的使用】
───────────────────────────────────────

规划器的内部流程:

class GraspPlanner(Annealer):
    
    def move(self):
        # 转为StateStruct进行高级操作
        current = StateStruct(self.model, self.data)
        current.from_array(self.state)
        
        # 加扰动
        current.position += noise
        current.grasp += noise
        
        # 转回13D供Annealer使用
        self.state = current.to_array()
    
    def energy(self):
        # 创建状态对象
        state = StateStruct(self.model, self.data)
        state.from_array(self.state)
        
        # 应用到仿真
        self.set_hand_pose(state)
        
        # 评估质量
        distance = evaluate_proximity()
        collision = check_collision()
        
        return distance + collision


【错误对照】

❌ 错误模式1：期望直接修改StateStruct会改变仿真
state.grasp = 0.9
new_state = StateStruct(model, data)
# new_state.grasp ≠ 0.9  (仿真没变!)

✅ 正确做法：
state = StateStruct(model, data, grasp=0.9)
ctrl = HandControl(model, data)
ctrl.set_hand_state(state)
mujoco.mj_forward(model, data)


❌ 错误模式2：忘记调用mj_forward()
ctrl.set_hand_state(state)
# data.ctrl改了，但data.xpos还是旧值
new_state = StateStruct(model, data)

✅ 正确做法：
ctrl.set_hand_state(state)
mujoco.mj_forward(model, data)  ← 必须
new_state = StateStruct(model, data)


❌ 错误模式3：混淆state.position和state_dic
state = StateStruct(model, data, position=[1,2,3])
# state.position = [1,2,3]  (参数值)
# state.state_dic['lh_palm'] = [x,y,z,...]  (MuJoCo读取)
# 这两个是不同的!

✅ 理解：
- position: 显式设置的值
- state_dic: 从MuJoCo实时读取的所有body位置


【优化建议】
───────────────────────────────────────

1. 状态管理
   ✓ 使用 copy() 保存不同时刻的状态快照
   ✓ 使用 to_array/from_array 实现状态序列化
   ✓ 使用 @property 快速访问单个变量

2. 控制应用
   ✓ 优先使用 set_hand_state() (高级接口)
   ✓ 只在需要单个关节时使用 set_act_val() (低级)
   ✓ 始终在 set_*之后调用 mj_forward()/mj_step()

3. 规划集成
   ✓ 在规划器中使用13D数组 (for numpy operations)
   ✓ 在评估能量时创建StateStruct进行应用
   ✓ 在move()中使用StateStruct进行高级操作

4. 调试技巧
   ✓ 打印 state.state_dic 查看所有body的当前位置
   ✓ 调用 ctrl.print_all_act_val() 查看执行器值
   ✓ 在mj_forward()前后创建两个StateStruct比较差异


【设计原理总结】

这个设计遵循以下原则:

1. 分离关注点 (Separation of Concerns)
   - StateStruct: 关注"状态表示"
   - HandControl: 关注"控制应用"
   - GraspPlanner: 关注"优化搜索"

2. 多层接口 (Multi-level API)
   - 高级: 协同变量 (易用)
   - 中级: set_hand_state() (平衡)
   - 低级: set_act_val() (细粒度)

3. 序列化兼容 (Serialization)
   - 13D数组用于数值计算
   - StateStruct用于逻辑操作
   - 自由转换无损失

4. 参考透明性 (Reference Transparency)
   - StateStruct是快照，不共享内部状态
   - copy()确保独立性
   - 避免副作用

"""
