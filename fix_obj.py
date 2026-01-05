import trimesh

# 加载原始模型
mesh = trimesh.load('kuka_iiwa_14/assets/decomposed.obj')

# 如果模型是由多个部分组成的（Scene对象），将其合并
if isinstance(mesh, trimesh.Scene):
    print("检测到多个几何体，正在合并...")
    mesh = mesh.dump(concatenate=True)

# 确保所有面朝外（修复透明/看不见的问题）
mesh.fix_normals()

# 导出为一个新的、纯净的 OBJ
mesh.export('kuka_iiwa_14/assets/bottle.obj')
print("修复完成！请在 XML 中改用 bottle.obj")