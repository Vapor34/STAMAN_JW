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
        target_pos = np.array([0.5, 0.5, 0.3])
        # 目标姿态：绕Y轴转180度（末端垂直向下）
        r = R.from_euler('xyz', [0, 180, 0], degrees=True)
        quat_scipy = r.as_quat() # [x,y,z,w]
        target_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]) # [w,x,y,z]

        while viewer.is_running():
            step_start = time.time()

            # 1. 获取当前末端位姿
            current_pos = data.site(ee_id).xpos
            current_mat = data.site(ee_id).xmat.reshape(3, 3)

            print(f"当前空间坐标{current_pos}")

            # 1.5 获取当前四元数 (从 xmat 转换)
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten()) # 也可以通过 xmat 转换，但 site 本身有 xquat
            
            # 2. 计算位置误差
            error_pos = target_pos - current_pos

            # 3. 【重要修改】计算旋转误差 (Orientation Error)
            # 计算从当前四元数到目标四元数的相对旋转 (q_target * q_current_inv)
            res_quat = np.zeros(4)
            cur_quat_inv = np.zeros(4)
            mujoco.mju_negQuat(cur_quat_inv, current_quat)
            mujoco.mju_mulQuat(res_quat, target_quat, cur_quat_inv)
            
            error_rot = np.zeros(3)
            # 将相对四元数转为轴角误差向量
            mujoco.mju_quat2Vel(error_rot, res_quat, 1.0)

            # 4. 合并误差并进行归一化（解决接近目标点变慢的问题）
            error_6dof = np.concatenate([error_pos, error_rot])
            
            # 如果误差很大，限制步长；如果误差极小，停止更新防止震荡
            norm_err = np.linalg.norm(error_6dof)
            if norm_err > 0.0001:
                # 这种方法可以让机械臂以较恒定的速度靠近，直到最后 1cm
                scale = np.clip(norm_err * 1.0, 0.1, 0.5) 
                error_scaled = (error_6dof / norm_err) * scale
            else:
                error_scaled = np.zeros(6)

            # 5. 计算雅可比并解算
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
            jac_arm = np.vstack([jacp[:, :7], jacr[:, :7]]) # 直接截取前7列

            # 6. 阻尼最小二乘法 (接近目标时，diag可以设小一点)
            diag = 1e-4 * np.eye(6)
            dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, error_scaled)

            # 7. 更新控制量
            # 限制单步最大转动量，保证安全
            dq = np.clip(dq, -0.1, 0.1)
            data.ctrl[:7] = data.qpos[:7] + dq

            # 8. 【新增】重力补偿 (解决 0.02 误差的关键)
            # 这会计算抵消重力所需的力矩，直接施加在关节上
            data.qfrc_applied[:7] = data.qfrc_gravcomp[:7]
            
            # 物理步进
            mujoco.mj_step(model, data)
            viewer.sync()

            # 维持频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    print(f"发生错误：{e}")