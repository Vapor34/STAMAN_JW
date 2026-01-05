import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

model_path = "kuka_iiwa_14/scene.xml"

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Kuka iiwa 6-DOF 运动控制中...")
        
        ee_id = model.site("attachment_site").id
        
        # --- 设定目标 ---
        target_pos = np.array([0.4, 0.4, 0.3])
        # 目标姿态：绕Y轴转180度（末端垂直向下）
        r = R.from_euler('xyz', [180, 180, 0], degrees=True)
        quat_scipy = r.as_quat() # [x,y,z,w]
        target_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]) # [w,x,y,z]

        # --- 在循环外定义阻尼参数 ---
        damping = 0.01  # 阻尼系数 lambda
        POS_THRESHOLD = 0.01  # 2毫米死区
        ROT_THRESHOLD = 0.01   # 约0.5度旋转死区

        while viewer.is_running():
            step_start = time.time()

            # 1. 获取当前末端位姿
            current_pos = data.site(ee_id).xpos
            current_mat = data.site(ee_id).xmat.reshape(3, 3)

            print(f"当前空间坐标{current_pos}")
            print(f"当前角度{current_mat}")

            # 1.5 获取当前四元数 (从 xmat 转换)
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten()) # 也可以通过 xmat 转换，但 site 本身有 xquat
            
            # 2. 计算位置误差
            error_pos = target_pos - current_pos


            # 2. 计算旋转误差 (使用更鲁棒的矩阵相减法或四元数法)
            # 推荐使用 MuJoCo 内置函数直接计算 6D 误差位移
            error_6dof = np.zeros(6)
            # site_xmat 是 3x3 旋转矩阵，target_quat 先转成矩阵
            target_mat = np.zeros(9)
            mujoco.mju_quat2Mat(target_mat, target_quat)
            
            # 计算当前和目标的 6 维误差 (pos + orientation)
            # 这个函数会自动处理四元数到角速度向量的转换
            error_rot = np.zeros(3)
            mujoco.mju_subQuat(error_rot, target_quat, current_quat) # 相对旋转

            # 4. 合并误差并进行归一化（解决接近目标点变慢的问题）
            error_6dof = np.concatenate([error_pos, error_rot])

            # 3. 重新整理雅可比计算
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
            J = np.vstack([jacp, jacr])[:, :7] # 6x7 矩阵

            # 4. 【核心修改】更稳健的 DLS 求解器
            # 使用 SVD 分解可以更安全地处理奇异点，或者使用你原来的公式但增加自适应性
            # 这里我们采用一种更简洁的写法：
            JJT = J @ J.T
            # 动态阻尼：如果矩阵接近奇异，增加阻尼
            # lambda_sq = damping**2 if np.linalg.det(JJT) < 1e-6 else 0
            lambda_sq = damping**2



            dist_pos = np.linalg.norm(error_pos)
            dist_rot = np.linalg.norm(error_rot)

            # 2. 【核心优化】判断是否进入死区
            if dist_pos < POS_THRESHOLD and dist_rot < ROT_THRESHOLD:
                # 如果足够接近，直接停止更新关节
                dq = np.zeros(7)
            else:
                # 3. 【核心优化】自适应增益 (接近目标时减小增益)
                # 距离越近，k 越小，防止过冲
                k_pos = np.clip(dist_pos * 2.0, 0.01, 0.5) 
                k_rot = np.clip(dist_rot * 1.5, 0.01, 0.2)
                
                # 对误差向量进行缩放
                error_scaled = np.zeros(6)
                error_scaled[:3] = error_pos * k_pos
                error_scaled[3:] = error_rot * k_rot

                # 4. DLS 求解
                try:
                    task_force = np.linalg.solve(JJT + lambda_sq * np.eye(6), error_scaled)
                    dq = J.T @ task_force
                except np.linalg.LinAlgError:
                    dq = np.zeros(7)

            # 5. 限制单步最大位移（物理防抖）
            dq = np.clip(dq, -0.2, 0.2) 

            # 6. 更新并施加重力补偿
            data.ctrl[:7] = data.qpos[:7] + dq
            # data.qfrc_applied[:7] = data.qfrc_gravcomp[:7]
            
            



            # 物理步进
            mujoco.mj_step(model, data)
            viewer.sync()

            # 维持频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    print(f"发生错误：{e}")