#机械臂末端从初始状态移动至指定位姿
#存在未知问题

import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# 路径指向场景文件，这样会有地面、灯光和好看的渲染效果
# 如果你的文件夹名字不同，请相应修改
model_path = "kuka_iiwa_14/scene.xml"

try:
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("成功加载 Kuka iiwa！正在运行正向动力学演示...")
        
        start_time = time.time()

        # 1. 获取末端执行器（site或body）的ID
        ee_name = "attachment_site" # 换成你模型中的名称
        ee_id = model.site(ee_name).id
        # 假设目标位姿
        target_pos = np.array([0.5, 0.5, 0.4])
        # 目标姿态：例如让末端朝下（这里用 3x3 旋转矩阵表示）
        # 你可以用 scipy.spatial.transform.Rotation 来方便地生成矩阵
        # target_quat_xyz = np.array([1, 0, 0, 0]) # 四元数 [w, x, y, z]

        


        # 定义欧拉角 (单位：度)
        # 假设你想让它绕 Y 轴转 180 度来实现“向下”
        # 'zyx' 表示旋转顺序，180 表示旋转角度
        r = R.from_euler('xyz', [0, 180, 0], degrees=True)

        # 转换为 MuJoCo 使用的 [w, x, y, z] 格式
        # 注意：scipy 默认输出的是 [x, y, z, w]，需要手动调换位置
        quat_scipy = r.as_quat()
        target_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, target_quat)

        target_mat = target_mat.reshape(3, 3)



        while viewer.is_running():
            step_start = time.time()


            # 在循环中添加打印
            curr_mat = data.site(ee_id).xmat.reshape(3, 3)
            print("--- 当前末端坐标系轴向 ---")
            print(f"X轴指向: {curr_mat[:, 0]}")
            print(f"Y轴指向: {curr_mat[:, 1]}")
            print(f"Z轴指向: {curr_mat[:, 2]}") # 看看这行是不是 [0, 0, -1]


            #-------------------抓取操作---------------

            # 1. 获取当前位姿
            current_pos = data.site(ee_id).xpos
            current_mat = data.site(ee_id).xmat.reshape(3, 3)

            print(f"当前空间坐标{current_pos}")

            # 1.5 获取当前四元数 (从 xmat 转换)
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten())

            # 2. 计算位置误差 (3x1)
            error_pos = target_pos - current_pos

            # 3. 计算旋转误差 (3x1)
            # 使用 MuJoCo 内置函数计算两个旋转矩阵之间的误差向量
            error_rot = np.zeros(3)

            # (1)简易法或用矩阵法：
            # mujoco.mju_subQuat(np.zeros(4), target_quat, data.site(ee_id).xquat) 
            
            # # (2)推荐下面这种更稳健的方法：
            # res_mat = target_mat @ current_mat.T
            # relative_quat = np.zeros(4)
            # mujoco.mju_mat2Quat(relative_quat, res_mat.flatten())
            # error_rot = relative_quat[1:] * np.sign(relative_quat[0]) # 提取轴角部分的简化表达


            mujoco.mju_quat2Vel(error_rot, current_quat, 1.0)

            # 4. 合并误差向量 (6x1)
            error_6dof = np.concatenate([error_pos, error_rot])

            # 5. 计算全雅可比矩阵 (6xnv)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
            jac_full = np.vstack([jacp, jacr]) # 合并成 6xnv

            # 6. 只取机械臂前 7 个关节
            jac_arm = jac_full[:, :7]

            # 7. 解算关节增量 (Damped Least Squares)
            diag = 1e-6 * np.eye(6)
            dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, error_6dof)

            # 8. 限制与更新
            # dq = np.clip(dq, -0.2, 0.2)
            data.ctrl[:7] = data.qpos[:7] + dq * 0.1
            #-------------------抓取操作---------------
            
            # 物理步进
            mujoco.mj_step(model, data)

            # 同步渲染
            viewer.sync()

            # 维持仿真频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    print(f"发生错误：{e}")
