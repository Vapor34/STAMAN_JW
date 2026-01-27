# GraspControl 类设计文档

## 设计目的

`GraspControl` 类是**控制变量管理中枢**，解决以下问题：

| 问题 | 解决方案 |
|-----|--------|
| 23个执行器的控制信号如何组织？ | 提供扁平化和结构化两种表示 |
| 优化算法需要1D向量，但人类需要理解？ | 自动互转，无缝切换 |
| 如何方便地操纵单个关节？ | 提供面向关节的API |
| 控制变量如何保存/加载？ | 序列化支持 |

---

## 核心概念

### 两种表示方式

#### 1. **扁平化表示**（Flattened）
```python
[j3_1, j3_2, j3_3, j3_4,      # 四指弯曲 J3 (4个)
 j4_1, j4_2, j4_3, j4_4,      # 四指侧摆 J4 (4个)
 thumb_1, thumb_2, ..., thumb_5,  # 大拇指 J1-5 (5个)
 tendon_1, tendon_2, tendon_3, tendon_4]  # 腱执行器 (4个)
```
**用途**：
- ✅ 优化算法（Annealer）喜欢1D向量
- ✅ 便于扰动和变异操作
- ✅ 紧凑，内存效率高

#### 2. **结构化表示**（Structured）
```python
{
    'flex_j3': {
        'lh_FFJ3': 0.5,    # 食指J3
        'lh_MFJ3': 0.5,    # 中指J3
        'lh_RFJ3': 0.5,    # 无名指J3
        'lh_LFJ3': 0.5     # 小指J3
    },
    'abd_j4': {
        'lh_FFJ4': 0.2,    # 食指J4
        'lh_MFJ4': 0.2,    # 中指J4
        'lh_RFJ4': 0.2,    # 无名指J4
        'lh_LFJ4': 0.2     # 小指J4
    },
    'thumb': {
        'lh_THJ1': 0.3,    # 大拇指J1
        'lh_THJ2': 0.4,    # 大拇指J2
        'lh_THJ3': 0.5,    # 大拇指J3
        'lh_THJ4': 0.6,    # 大拇指J4
        'lh_THJ5': 0.7     # 大拇指J5
    },
    'tendon': {
        'lh_FFJ0': 1.5,    # 食指腱
        'lh_MFJ0': 1.5,    # 中指腱
        'lh_RFJ0': 1.5,    # 无名指腱
        'lh_LFJ0': 1.5     # 小指腱
    }
}
```
**用途**：
- ✅ 易于理解和调试
- ✅ 关节名称清晰可读
- ✅ 便于单关节修改

---

## 使用流程

### 基本初始化
```python
from src.mujoco_utils import mujoco_load
from src.grasp_planner import GraspPlanner
from src.grasp_control import GraspControl

model, data = mujoco_load("shadow_hand/scene_left.xml")
planner = GraspPlanner(np.zeros(9), model, data, 'bottle_body')

# 创建控制管理器
control = GraspControl(model, planner)
```

### 扁平化 ↔ 结构化 互转
```python
# 获取扁平化（用于优化算法）
flat_ctrl = control.get_control_flat()  # shape: (23,)

# 修改并设置回去
flat_ctrl[0] = 0.5
control.set_control_flat(flat_ctrl)

# 获取结构化（用于理解和调试）
struct_ctrl = control.get_control_struct()
print(struct_ctrl['flex_j3']['lh_FFJ3'])  # 访问特定关节

# 修改结构化并设置回去
struct_ctrl['flex_j3']['lh_FFJ3'] = 0.5
control.set_control_struct(struct_ctrl)
```

### 单关节操作
```python
# 设置单个关节
control.set_single_joint('lh_FFJ3', 0.5)
control.set_single_joint('lh_THJ5', 0.7)

# 获取单个关节
val = control.get_single_joint('lh_FFJ3')

# 打印信息
control.print_control_info()      # 扁平化视图
control.print_control_struct()    # 结构化视图
```

### 序列化
```python
# 保存到文件
import numpy as np
saved = control.to_array()
np.save('control_state.npy', saved)

# 从文件加载
loaded = np.load('control_state.npy')
control.from_array(loaded)
```

---

## 在项目中的位置

```
项目结构
├── hnb6.py                    # 主控制脚本
│   └── 使用 StateStruct       # 规划状态 (9D: 位置+姿态+协同变量)
│   └── 使用 GraspPlanner      # 优化规划器
│   └── 使用 qpos_to_ctrl()    # 状态 → 控制信号
│
├── src/
│   ├── grasp_state.py         # ✅ StateStruct (规划状态)
│   ├── grasp_planner.py       # ✅ GraspPlanner (优化器)
│   ├── grasp_control.py       # ✅ GraspControl (控制变量) ← 你的文件
│   └── mujoco_utils.py        # 工具函数
```

### 数据流
```
StateStruct (9D)
    ↓ [规划]
GraspPlanner.energy()
    ↓ [最优化]
最优规划状态
    ↓ [转换]
qpos_to_ctrl() → 控制信号
    ↓ [应用]
MuJoCo 执行
```

---

## 关键方法详解

### `flatten_to_structured(control_flat)`
将1D向量 → 字典
```python
control_flat = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, ...])
control_struct = control.flatten_to_structured(control_flat)
# 返回结构化字典，方便按类别查看
```

### `structured_to_flatten(control_struct)`
将字典 → 1D向量
```python
control_struct = {
    'flex_j3': {'lh_FFJ3': 0.5, ...},
    'abd_j4': {...},
    'thumb': {...},
    'tendon': {...}
}
control_flat = control.structured_to_flatten(control_struct)
# 返回 1D 向量，可用于优化
```

### `set_single_joint(joint_name, value)`
面向关节的写入接口
```python
control.set_single_joint('lh_FFJ3', 0.5)
# 自动找到关节所属类别，更新扁平化向量
```

### `get_single_joint(joint_name)`
面向关节的读取接口
```python
val = control.get_single_joint('lh_FFJ3')  # 返回 0.5
```

---

## 与其他模块的交互

### 与 StateStruct 的关系
```python
StateStruct:
  9D 状态 = [位置(3D), 姿态(4D), grasp(1D), spread(1D)]
  用途：描述规划的目标状态

GraspControl:
  23D 控制 = [J3(4个), J4(4个), Thumb(5个), Tendon(4个), ...]
  用途：描述执行器的控制信号
```

### 与 Annealer 的交互
```python
class GraspPlanner(Annealer):
    def move(self):
        # 修改 self.state (9D StateStruct)
        
    def energy(self):
        # 评估规划状态的能量
        
# 优化完成后，通过 qpos_to_ctrl() 将规划状态转换为控制信号
```

---

## 扩展建议

未来可以添加的功能：

```python
# 1. 约束支持
def set_constraint(joint_name, min_val, max_val):
    """为单个关节设置约束"""
    pass

# 2. 平滑操作
def smooth_transition(target_ctrl, steps=20):
    """平滑过渡到目标控制"""
    pass

# 3. 记录回放
def record_trajectory(duration=10.0):
    """录制轨迹"""
    pass

def playback_trajectory(trajectory):
    """回放轨迹"""
    pass

# 4. 统计分析
def get_statistics(control_list):
    """分析多个控制状态的统计"""
    pass
```

---

## 测试

运行示例：
```bash
python example_grasp_control.py
```

这将演示5个典型使用场景：
1. 基本创建和初始化
2. 扁平化 ↔ 结构化互转
3. 单关节控制
4. 优化工作流
5. 序列化/反序列化

