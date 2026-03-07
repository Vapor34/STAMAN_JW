"""
===============================================================================
📁 项目最终结构总览和快速导航
===============================================================================
"""

# 项目文件结构
# ═══════════════════════════════════════════════════════════════════════════

"""

sdhand_grasp/  (Shadow Hand Grasp Planning Project)
│
├─ 📁 core/  ◄─ 核心功能模块
│  ├─ __init__.py           导出Hand类
│  ├─ hand_state.py         StateStruct - 状态表示层
│  ├─ hand_control.py       HandControl - 执行控制层
│  └─ simulator.py          mujoco_load - 模型加载
│
├─ 📁 planning/  ◄─ 规划模块
│  ├─ __init__.py           导出规划器
│  └─ grasp_planner.py      GraspPlanner - 优化算法层
│
├─ 📁 configs/  ◄─ 配置文件 (不变)
│  ├─ *.xml                 MuJoCo模型配置
│  └─ assets/               模型资源
│
├─ 📄 main.py               ◄─ 主程序 (协调器)
│
├─ 📖 MODULE_ARCHITECTURE.md   完整的架构梳理文档
├─ 📖 INTERACTION_DETAILS.md   StateStruct和HandControl交互详解
├─ 📖 QUICK_REFERENCE.md       使用场景速查表
├─ 📖 CORE_CONCEPTS.md         核心概念和设计模式
│
└─ 其他文件 (MUJOCO_LOG.TXT, .vscode等)


═════════════════════════════════════════════════════════════════════════════
"""


# 快速导航：查询什么去哪个文件
# ═══════════════════════════════════════════════════════════════════════════

"""

🔍 问题导航表

如果我想了解...                    → 去这个文件

【代码和实现】
──────────────────────────────────────────────────────────────────────────────
StateStruct的功能               → core/hand_state.py 源代码
HandControl的功能               → core/hand_control.py 源代码
怎样使用GraspPlanner            → planning/grasp_planner.py 源代码
main.py的完整流程               → main.py 源代码

【文档和指南】
──────────────────────────────────────────────────────────────────────────────
整个项目的架构设计               → MODULE_ARCHITECTURE.md
StateStruct和HandControl关系     → INTERACTION_DETAILS.md
快速使用示例和API查询            → QUICK_REFERENCE.md
核心概念和设计模式               → CORE_CONCEPTS.md
项目结构和导航                   → 当前文件

【特定功能】
──────────────────────────────────────────────────────────────────────────────
创建和初始化状态                 → QUICK_REFERENCE.md【常见操作速查】
应用状态到仿真                   → CORE_CONCEPTS.md【两者的交互方式】
执行器映射和协同变量             → INTERACTION_DETAILS.md【类和方法的职责】
规划器内部流程                   → INTERACTION_DETAILS.md【场景4】
平滑过渡和动画                   → QUICK_REFERENCE.md【场景3】

【调试和错误】
──────────────────────────────────────────────────────────────────────────────
常见错误和解决方案               → CORE_CONCEPTS.md【陷阱和最佳实践】
StateStruct用法错误              → CORE_CONCEPTS.md【陷阱1-3】
忘记mj_forward()问题            → QUICK_REFERENCE.md【陷阱1】
print调试信息                     → QUICK_REFERENCE.md【调用print_all_act_val】


═════════════════════════════════════════════════════════════════════════════
"""


# 核心概念速记卡
# ═══════════════════════════════════════════════════════════════════════════

"""

📌 StateStruct 记忆卡

是什么: 灵巧手状态的高级表示
包含: 13个标量 (3位置 + 4四元数 + 6协同变量)
怎样用:
  - 创建: state = StateStruct(model, data, position=[...], grasp=0.8)
  - 访问: state.x, state.grasp, state.quaternion
  - 序列化: array = state.to_array(); state.from_array(array)
特点:
  ✓ 只是表示，不修改仿真
  ✓ 支持快照和深拷贝
  ✓ 提供高级@property接口


📌 HandControl 记忆卡

是什么: 执行器控制的底层接口
包含: 20个执行器 + synergy_map映射
怎样用:
  - 创建: ctrl = HandControl(model, data)
  - 完整: ctrl.set_hand_state(state)
  - 协同: ctrl.set_syn_val('grasp', 0.8)
  - 单个: ctrl.set_act_val('lh_THJ1', 0.5)
特点:
  ✓ 直接修改data.ctrl
  ✓ 多层级API
  ✓ 提供低级执行器访问


📌 它们的关系 记忆卡

StateStruct (定义)
   ↓ 通过set_hand_state()
HandControl (执行)
   ↓ 修改data.ctrl[], data.qpos[]
MuJoCo (物理)
   ↓ 通过mj_forward()/mj_step()
新StateStruct (读取)

关键: 必须在HandControl操作后调用mj_forward()


═════════════════════════════════════════════════════════════════════════════
"""


# 最简单的使用示例
# ═══════════════════════════════════════════════════════════════════════════

"""

【最小工作示例】

from core.hand_state import StateStruct
from core.hand_control import HandControl
from core.simulator import mujoco_load
import mujoco

# 1. 加载模型
model, data = mujoco_load("configs/scene_left.xml")

# 2. 创建控制器
ctrl = HandControl(model, data)

# 3. 定义目标状态
target = StateStruct(
    model, data,
    position=[0.5, 0.5, 0.5],
    grasp=0.8,
    curl=0.3
)

# 4. 应用控制
ctrl.set_hand_state(target)

# 5. 更新物理
mujoco.mj_forward(model, data)

# 6. 读取结果
result = StateStruct(model, data)
print(f"Hand position: {result.position}")
print(f"Palm body state: {result.state_dic}")


【规划示例】

from planning.grasp_planner import GraspPlanner

# 创建规划器
planner = GraspPlanner(
    state=target.to_array(),  # ← 13D数组
    model=model,
    data=data,
    bottle_body_name='bottle_body'
)
planner.steps = 5000

# 搜索最优解
best_pose, best_energy = planner.anneal()

# 应用最优解
optimal_state = StateStruct(model, data)
optimal_state.from_array(best_pose)
ctrl.set_hand_state(optimal_state)
mujoco.mj_forward(model, data)


═════════════════════════════════════════════════════════════════════════════
"""


# 重点知识检查表
# ═══════════════════════════════════════════════════════════════════════════

"""

你现在应该能够回答这些问题：

【StateStruct相关】

Q1: StateStruct有多少维度？
A: 13维 (3位置 + 4四元数 + 6协同变量)

Q2: 修改StateStruct的grasp值会改变仿真吗？
A: 不会。StateStruct只是表示，需要通过HandControl.set_hand_state()应用。

Q3: StateStruct.position和state_dic有什么区别？
A: position是初始化参数值，state_dic是从MuJoCo实时读取的所有body位置。

Q4: 怎样保存状态快照？
A: 创建新的StateStruct实例，或使用copy()深拷贝。

Q5: StateStruct支持序列化吗？
A: 支持。to_array()转为13D数组，from_array()反向恢复。


【HandControl相关】

Q6: HandControl管理多少个执行器？
A: 20个（灵巧手的所有关节）。

Q7: 协同变量是什么？
A: 一个高级控制变量映射到多个执行器。例如grasp映射到5个手指。

Q8: synergy_map有多少个协同变量？
A: 6个（grasp, curl, spread, thumb_base, thumb_flex, wrist）。

Q9: HandControl可以单独修改一个执行器吗？
A: 可以，使用set_act_val()方法。

Q10: HandControl会调用mj_forward()吗？
A: 不会。需要手动调用。


【流程相关】

Q11: 应用状态到仿真的正确步骤是？
A: 1) ctrl.set_hand_state(state) 2) mujoco.mj_forward(model, data)

Q12: 规划器中state的维度是？
A: 13D（与StateStruct.to_array()相同）。

Q13: 怎样从规划器结果恢复StateStruct？
A: best_pose是13D数组，使用state.from_array(best_pose)恢复。

Q14: 应用完整状态的最简单方法是什么？
A: state = StateStruct(...); ctrl.set_hand_state(state)

Q15: 平滑过渡两个状态怎样实现？
A: 线性插值两个StateStruct的属性，创建中间状态应用。

"""

# 答题检查：
# 如果能全部正确回答，说明你已经完全理解了项目架构。
# 某些问题答不对，可以回到对应文档重新学习。

"""


═════════════════════════════════════════════════════════════════════════════
"""


# 下一步建议
# ═══════════════════════════════════════════════════════════════════════════

"""

现在你已经理解了核心架构，可以进行以下操作：

1️⃣  学习phase - 深入学习
   □ 阅读源代码 (core/*.py) 理解实现细节
   □ 在main.py中添加print语句调试理解流程
   □ 修改参数（初始位置、协同变量）观察效果

2️⃣  测试phase - 验证理解
   □ 编写简单脚本测试StateStruct功能
   □ 编写简单脚本测试HandControl功能
   □ 验证13D数组的序列化/反序列化
   □ 尝试平滑过渡动画

3️⃣  扩展phase - 增加功能
   □ 添加新的协同变量
   □ 添加新的执行器
   □ 修改能量函数
   □ 实现新的规划策略

4️⃣  生产phase - 实际应用
   □ 优化规划参数
   □ 处理实际控制问题
   □ 集成真实硬件
   □ 性能优化

"""


# 常见操作快速参考（超浓缩版）
# ═══════════════════════════════════════════════════════════════════════════

"""

任务                                代码
────────────────────────────────────────────────────────────────────────────

初始化                              model, data = mujoco_load(path)
                                    ctrl = HandControl(model, data)

创建状态                            state = StateStruct(model, data,
                                                        position=[x,y,z],
                                                        grasp=0.8)

修改状态                            state.grasp = 0.9
                                    state.position = [x,y,z]

应用状态                            ctrl.set_hand_state(state)
                                    mujoco.mj_forward(model, data)

读取结果                            result = StateStruct(model, data)
                                    print(result.position)

序列化                              array = state.to_array()
反序列化                            state.from_array(array)

深拷贝                              state_copy = state.copy()

控制协同组                          ctrl.set_syn_val('grasp', 0.8)

控制单个执行器                      ctrl.set_act_val('lh_THJ1', 0.5)

规划                                planner = GraspPlanner(s.to_array(), ...)
                                    best, energy = planner.anneal()

从规划结果恢复                      state.from_array(best)

调试                                ctrl.print_all_act_val()

"""

