import trimesh

# 1. 加载你在 XML 中找到的路径
mesh = trimesh.load("/home/joe/staman/kuka_model/kuka_iiwa_14/assets/decomposed.obj")

# 2. 打印基础数据
print(f"顶点数量: {len(mesh.vertices)}")
print(f"面数量: {len(mesh.faces)}")

# 3. 查看物体的质心 (Center of Mass) —— 这是计算 Sum F 的关键
print(f"质心坐标: {mesh.center_mass}")

# 4. 预览模型 (会弹出一个 3D 窗口)
mesh.show()