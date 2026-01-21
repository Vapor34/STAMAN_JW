# hnb4.py 改进方案 - 分级控制详解

## 问题诊断

**现象：** 手指闭合时，只有近端指节（j1）弯曲，中间和远端指节几乎不动

**根本原因：** 
```
原始策略（hnb3）：
  所有屈肌关节 (j1, j2, j3) 都用同一个 grasp_val 控制
  
  flex_angle = grasp_val * 1.5
  qpos[j1] = flex_angle  ← 全部用这个值
  qpos[j2] = flex_angle
  qpos[j3] = flex_angle
  
问题：
  • 执行器力度相同，但不同关节的初始状态和刚性不同
  • j1 最容易动（最前面的关节）
  • j2 需要 j1 先推动
  • j3 需要 j1+j2 都推动
  
结果：
  • 执行器力度只够弯曲 j1
  • j2, j3 的反弹力大于驱动力，保持伸直
```

---

## 解决方案：分级控制

### 新的关节分类策略

```python
# 原始分类（混乱）
flex_indices = [j1, j2, j3, j1, j2, j3, ...]  # 无序混合

# 改进分类（清晰）
flex_j1_indices = [FFJ1, MFJ1, RFJ1, LFJ1]  # 所有近端
flex_j2_indices = [FFJ2, MFJ2, RFJ2, LFJ2]  # 所有中间
flex_j3_indices = [FFJ3, MFJ3, RFJ3, LFJ3]  # 所有远端
```

### 执行器力度的分级激活

```
grasp_val ∈ [0, 1]

│
│ j1 (近端)      ████████────────────
│ j2 (中间)      ────████████────────
│ j3 (远端)      ────────████████████
│
└─────────────────────────────────────
  0.0   0.2   0.4   0.6   0.8   1.0

具体映射：
• j1：grasp_val ∈ [0.0, 0.4] → progress ∈ [0, 1] → force
• j2：grasp_val ∈ [0.2, 0.7] → progress ∈ [0, 1] → force  (重叠激活)
• j3：grasp_val ∈ [0.4, 1.0] → progress ∈ [0, 1] → force
```

### 控制代码

```python
for i in range(model.nu):
    jnt_adr = model.jnt_qposadr[model.actuator_trnid[i, 0]]
    
    if jnt_adr in planner.flex_j1_indices:
        # j1：0→0.4 时从 0→max
        progress = np.clip(grasp_val / 0.4, 0.0, 1.0)
        val = progress * 2.5 * 2.5  # 力度 = 6.25
        
    elif jnt_adr in planner.flex_j2_indices:
        # j2：0.2→0.7 时从 0→max
        progress = np.clip((grasp_val - 0.2) / 0.5, 0.0, 1.0)
        val = progress * 2.5 * 2.5
        
    elif jnt_adr in planner.flex_j3_indices:
        # j3：0.4→1.0 时从 0→max
        progress = np.clip((grasp_val - 0.4) / 0.6, 0.0, 1.0)
        val = progress * 2.5 * 2.5
```

---

## 时间轴分析

```
握力递增过程：grasp_val 从 0 → 1（4秒内线性增长）

时刻      grasp_val   j1进度   j2进度   j3进度   现象
───────────────────────────────────────────────────
0.0s        0.0       0%      0%      0%     五指张开
0.5s        0.125     31%     0%      0%     j1 快速弯曲
1.0s        0.25      62%     0%      0%     j1 继续弯曲
1.5s        0.375     93%     35%     0%     j1 快完成，j2 开始
2.0s        0.5       100%    60%     0%     j1 完全，j2 中等
2.5s        0.625     100%    85%     21%    j2 快完成，j3 开始
3.0s        0.75      100%    100%    58%    j2 完全，j3 中等
3.5s        0.875     100%    100%    79%    j3 快完成
4.0s        1.0       100%    100%    100%   五指完全闭合
```

---

## 预期改进效果

### 原始状态（hnb3）
```
握住瓶子时的手指形态：
    
    ╔═══════════════╗
    ║ 近端 ▁▁▁▁ ╱   ║  ← 明显弯曲
    ║ 中间 ─────     ║  ← 几乎不动
    ║ 远端 ─────     ║  ← 几乎不动
    ║ 掌心 ▔▔▔▔▔     ║
    ╚═══════════════╝
```

### 改进后（hnb4）
```
握住瓶子时的手指形态：
    
    ╔═══════════════╗
    ║ 近端 ▁▁▁▁ ╱   ║  ← 充分弯曲
    ║ 中间 ▂▂▂▂ ╱   ║  ← 明显弯曲
    ║ 远端 ▃▃▃▃ ╱   ║  ← 明显弯曲
    ║ 掌心 ▔▔▔▔▔     ║
    ╚═══════════════╝
```

---

## 可调参数

如果效果仍不理想，可调整以下参数：

### 1. 分级阈值（激活点）
```python
# 当前设置
j1: 0.0 ~ 0.4
j2: 0.2 ~ 0.7
j3: 0.4 ~ 1.0

# 调整建议
# 想让 j2/j3 更早激活 → 降低阈值 (e.g., 0.15, 0.3)
# 想让 j2/j3 更晚激活 → 提高阈值 (e.g., 0.3, 0.6)
```

### 2. 力度系数
```python
# 当前：2.5 * 2.5 = 6.25 最大力度
# 增加力度：改成 3.0 * 2.5 = 7.5
# 减少力度：改成 2.0 * 2.5 = 5.0
```

### 3. 非线性指数
```python
# 当前：progress ** 1.0 (线性)
# 改成 progress ** 1.2 (先快后慢)
# 改成 progress ** 0.8 (先慢后快)
```

---

## 技术细节

### 函数签名变化

```python
# hnb3
return flex_indices, abd_indices, thumb_indices

# hnb4
return flex_j1_indices, flex_j2_indices, flex_j3_indices, abd_indices, thumb_indices
```

### 影响的地方

1. `get_synergy_mapping()` - 分类逻辑
2. `qpos_to_ctrl_improved()` - 执行器力度计算
3. `GraspPlanner.__init__()` - 初始化关节映射
4. `set_hand_pose()` - 规划阶段的关节设置

---

## 验证方法

运行 `python hnb4.py` 并观察：

1. **初始状态**：五指应该自然张开，不应该反曲
2. **0-1 秒**：手掌快速移动到瓶子附近
3. **1-2 秒**：近端指节（j1）快速弯曲
4. **2-3 秒**：中间指节（j2）逐渐弯曲
5. **3-5 秒**：远端指节（j3）最后弯曲，形成自然的握拳姿态

如果没有看到预期的分级弯曲效果，可以调整上述参数。
