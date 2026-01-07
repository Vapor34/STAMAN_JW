import mujoco

model = mujoco.MjModel.from_xml_path("shadow_hand/scene_left.xml")
data = mujoco.MjData(model)

# 查找关节 ID
jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "palm_freejoint")

if jnt_id != -1:
    qpos_adr = model.jnt_qposadr[jnt_id]
    jnt_type = model.jnt_type[jnt_id]
    print(f"成功找到关节！索引地址: {qpos_adr}, 类型代码: {jnt_type} (0=Free)")
else:
    print("未找到关节，请检查 XML 中的 name 是否匹配。")

for i in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(jnt_name)

for j in len(data.qpos()):
    print(data.qpos(j))
