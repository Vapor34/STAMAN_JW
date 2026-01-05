import trimesh
import numpy as np
import mujoco
import mujoco.viewer

def get_surface_samples(obj_path, num_samples=100):
    # 1. 加载 Mesh
    mesh = trimesh.load(obj_path)
    
    # 2. 在表面进行随机采样
    # samples: 采样点的坐标 [num_samples, 3]
    # face_indices: 每个点所属的面索引，用于获取法向量
    samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    
    # 3. 获取对应的法向量
    # normals: 采样点处的法向量 [num_samples, 3]
    normals = mesh.face_normals[face_indices]
    
    return samples, normals

def transform_samples_to_world(samples, normals, obj_pos, obj_quat, scale=0.3):
    """
    scale 必须与你 XML 中 <mesh> 标签里的 scale 保持一致
    """
    import mujoco
    import numpy as np
    
    world_samples = []
    world_normals = []
    
    # 构造旋转矩阵
    obj_mat = np.zeros(9)
    mujoco.mju_quat2Mat(obj_mat, obj_quat)
    obj_mat = obj_mat.reshape(3, 3)
    
    for i in range(len(samples)):
        # 核心修正点：在旋转和平移之前，先对原始采样点应用缩放
        scaled_sample = samples[i] * scale 
        
        # 1. 位置变换：R * (p_local * scale) + t
        p_world = obj_mat @ scaled_sample + obj_pos
        
        # 2. 法向量变换：法向量只需要旋转，不需要平移和缩放（保持模长为1）
        n_world = obj_mat @ normals[i]
        
        world_samples.append(p_world)
        world_normals.append(n_world)
        
    return np.array(world_samples), np.array(world_normals)





# 运行采样
obj_file = "kuka_iiwa_14/assets/bottle.obj"
samples, normals = get_surface_samples(obj_file)

print(f"采样完成，共得到 {len(samples)} 个候选点。")


model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
data = mujoco.MjData(model)

# 获取物体在场景中的 ID (假设 body 叫 'object_body')
obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'object_body')

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 1. 获取物体当前在仿真中的姿态
        current_pos = data.xpos[obj_id]
        current_quat = data.xquat[obj_id]
        
        # 2. 变换采样点到当前位置
        w_points, w_normals = transform_samples_to_world(samples, normals, current_pos, current_quat)
        
        # 3. 在 Viewer 中绘制点和线
        viewer.user_scn.ngeom = 0 # 清除上一帧的绘制
        for i in range(len(w_points)):
            # 画采样点 (小红球)
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], 
                                w_points[i], np.eye(3).flatten(), [1, 0, 0, 1])
            viewer.user_scn.ngeom += 1
            
            # 画法向量 (绿线)
            pick_end = w_points[i] + w_normals[i] * 0.02 # 线段长度 2cm
            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                mujoco.mjtGeom.mjGEOM_LINE, [0.001, 0, 0], 
                                w_points[i], np.eye(3).flatten(), [0, 1, 0, 1])
            # 设置线段的终点（MuJoCo 线段绘制较为特殊，此处为示意，实际可用两个点连线）
            viewer.user_scn.geoms[viewer.user_scn.ngeom].pos = w_points[i]
            viewer.user_scn.geoms[viewer.user_scn.ngeom].size = [0.001, 0.001, 0.02]
            # 简单的朝向处理
            viewer.user_scn.ngeom += 1

        viewer.sync()
        mujoco.mj_step(model, data)