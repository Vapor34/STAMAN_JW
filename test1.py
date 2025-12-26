#机械臂末端从初始状态移动至指定空间坐标（不含末端姿态）

import mujoco
import mujoco.viewer
import numpy as np
import time

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
        target_pos = np.array([0.5, 0.5, 0.5])
        step_size = 0.1

        while viewer.is_running():
            step_start = time.time()

            #-------------------抓取操作---------------

            # 1. 获取当前位置和误差
            current_pos = data.site(ee_id).xpos
            error = target_pos - current_pos

            # 2. 计算雅可比矩阵 (全量)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)

            # 3. 【关键修正】只取前 7 个关节对应的雅可比列
            # 这样计算出的 dq 长度就正好是 7
            jacp_arm = jacp[:, :7] 

            # 4. 计算关节速度增量 (使用截取后的雅可比)
            diag = 0.01 * np.eye(3)
            dq = jacp_arm.T @ np.linalg.solve(jacp_arm @ jacp_arm.T + diag, error)

            # 5. 限制最大步进
            # max_dq = 0.05
            # dq = np.clip(dq, -max_dq, max_dq)

            # 6. 更新控制量
            # 现在 dq 是 (7,)，data.qpos[:7] 也是 (7,)，data.ctrl[:7] 也是 (7,)
            # 三者维度统一，不会报错
            arm_joint_ids = 7
            current_qpos = data.qpos[:arm_joint_ids]
            data.ctrl[:arm_joint_ids] = current_qpos + dq * step_size # 这里的 0.1 即 step_size

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
