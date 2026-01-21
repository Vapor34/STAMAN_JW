import mujoco
import mujoco.viewer
import numpy as np
import time

model_path = 'shadow_hand/scene_left.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 获取所有手部关节
hand_joints = []
hand_prefix = 'lh_'

for i in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if jnt_name and hand_prefix in jnt_name and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
        qpos_adr = model.jnt_qposadr[i]
        hand_joints.append({
            'name': jnt_name,
            'id': i,
            'qpos_adr': qpos_adr,
            'range': model.jnt_range[i].copy()
        })

print(f"找到 {len(hand_joints)} 个手部关节:\n")
for joint in hand_joints:
    print(f"  {joint['name']:30s} range: [{joint['range'][0]:7.3f}, {joint['range'][1]:7.3f}]")

# 获取所有执行器
actuators = []
for i in range(model.nu):
    jnt_id = model.actuator_trnid[i, 0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    ctrl_range = model.actuator_ctrlrange[i]
    actuators.append({
        'name': act_name,
        'joint': jnt_name,
        'id': i,
        'ctrl_range': ctrl_range
    })

print(f"\n找到 {len(actuators)} 个执行器:\n")
for act in actuators:
    print(f"  {act['name']:30s} → {act['joint']:30s} ctrl: [{act['ctrl_range'][0]:7.3f}, {act['ctrl_range'][1]:7.3f}]")

# 映射执行器到关节
joint_to_actuators = {}
for joint in hand_joints:
    joint_to_actuators[joint['name']] = []

for act in actuators:
    if act['joint'] in joint_to_actuators:
        joint_to_actuators[act['joint']].append(act['name'])

# 测试每个关节
print("\n" + "="*80)
print("关节动作测试 - 缓慢运动")
print("="*80)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 初始化到稳定状态
    mujoco.mj_resetData(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.5)
    
    for joint_idx, joint in enumerate(hand_joints):
        joint_name = joint['name']
        qpos_adr = joint['qpos_adr']
        low, high = joint['range']
        
        # 获取执行器
        controlling_actuators = joint_to_actuators.get(joint_name, [])
        
        print(f"\n[{joint_idx+1}/{len(hand_joints)}] 测试: {joint_name}")
        print(f"  范围: [{low:.3f}, {high:.3f}]")
        print(f"  执行器: {controlling_actuators if controlling_actuators else '【无执行器 - 被动关节】'}")
        
        if not controlling_actuators:
            print(f"  → 跳过（被动关节）")
            continue
        
        # 稳定当前状态
        data.ctrl[:] = 0.0
        for _ in range(50):
            mujoco.mj_step(model, data)
            viewer.sync()
        
        # 保存初始状态
        initial_qpos = data.qpos.copy()
        
        # 1. 从低端到高端的缓慢运动
        print(f"  → 运动到最大值...")
        motion_steps = 300
        
        for step in range(motion_steps):
            # 线性插值目标位置
            progress = step / motion_steps
            target_pos = low + (high - low) * progress
            
            # 施加控制力使关节向目标位置运动
            for act_name in controlling_actuators:
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                # 如果目标是较大的值，施加正力；否则负力
                data.ctrl[act_id] = 1.0 if progress > 0.5 else -1.0
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 显示进度
            if step % 50 == 0:
                current_pos = data.qpos[qpos_adr]
                print(f"      进度: {progress*100:.0f}% - 当前位置: {current_pos:.3f}")
        
        time.sleep(0.5)
        
        # 2. 从高端到低端的缓慢运动
        print(f"  → 运动到最小值...")
        for step in range(motion_steps):
            progress = step / motion_steps
            target_pos = high - (high - low) * progress
            
            for act_name in controlling_actuators:
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                data.ctrl[act_id] = -1.0 if progress > 0.5 else 1.0
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            if step % 50 == 0:
                current_pos = data.qpos[qpos_adr]
                print(f"      进度: {progress*100:.0f}% - 当前位置: {current_pos:.3f}")
        
        # 3. 缓慢回到初始位置
        print(f"  → 回归原位...")
        for step in range(200):
            progress = step / 200
            
            # 关闭所有控制力
            for act_name in controlling_actuators:
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                data.ctrl[act_id] = 0.0
            
            mujoco.mj_step(model, data)
            viewer.sync()
        
        time.sleep(1.0)
        print(f"  ✓ 完成")

print("\n" + "="*80)
print("测试完成")
print("="*80)
