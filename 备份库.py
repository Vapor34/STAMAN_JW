
    def energy(self):
        """抓取能量函数（优化目标）
        
        目标：
        1. 最小化手指到物体的距离 → 手靠近物体
        2. 手掌对准物体 → 便于抓取
        3. 避免碰撞和超出关节限位
        4. 倾向于中等抓取力度 → 既能接触又不过度闭合
        """
        # ===== 权重配置 =====
        W_PROXIMITY = 25.0        # 接近性权重
        W_ORIENTATION = 1    # 朝向性权重
        W_COLLISION_HAND = 2  # 手-物碰撞惩罚
        W_JOINT_LIMIT = 100    # 关节超限惩罚
        
        # 将13D数组转换为StateStruct
        state_struct = StateStruct(self.model, self.data)
        state_struct.from_array(self.state)
        self.set_hand_pose(state_struct)


        bottle_pos = self.data.xpos[self.bottle_body_id]
        palm_center_pos = self.data.site_xpos[self.palm_center_site_id]
        # ===== 1. 接近性能量：手指到物体的平均距离 =====
        total_dist = 0.0


        for b_id in self.contact_body_ids:
            total_dist += np.linalg.norm(self.data.xpos[b_id] - bottle_pos)
        total_dist += np.linalg.norm(palm_center_pos - bottle_pos)
        proximity_energy = total_dist / (len(self.contact_body_ids)+1)



        
        # ===== 2. 朝向性能量：手掌法向量与指向物体方向的夹角 =====
        # 手掌法向量 = 手掌坐标系的-Y轴（指向掌心侧）
        rot_mat = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_normal = -rot_mat[:, 1]  # 指向掌心方向
        
        # 从手掌指向物体的方向
        vec_to_bottle = bottle_pos - palm_center_pos
        vec_to_bottle_norm = vec_to_bottle / (np.linalg.norm(vec_to_bottle) + 1e-6)
        
        # cos_angle = dot(palm_normal, vec_to_bottle_norm)
        # 当朝向一致时 = 1.0，反向时 = -1.0
        # energy = (1.0 - cos_angle) → 朝向一致时最小
        orientation_energy = (1.0 - np.dot(palm_normal, vec_to_bottle_norm)) * 1.5
        
        # ===== 3. 碰撞惩罚 =====
        collision_penalty = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            is_bottle = (con.geom1 in self.bottle_geom_ids or con.geom2 in self.bottle_geom_ids)
            is_hand = (con.geom1 in self.hand_geom_ids or con.geom2 in self.hand_geom_ids)
            if is_bottle and is_hand and con.dist < 0.03:
                # 穿透越深，惩罚越大，而不是一刀切
                collision_penalty += abs(con.dist) * W_COLLISION_HAND * 100


        
        # ===== 4. 关节限位惩罚 =====
        joint_limit_penalty = 0.0
        for i in range(1, self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name and self.hand_prefix in jnt_name:
                q_addr = self.model.jnt_qposadr[i]
                q_val = self.data.qpos[q_addr]
                low, high = self.model.jnt_range[i]
                
                # 超出下限
                if q_val < low:
                    joint_limit_penalty += np.square(low - q_val)
                # 超出上限
                elif q_val > high:
                    joint_limit_penalty += np.square(q_val - high)
        

        # ===== 总能量 =====
        total_energy = (
            proximity_energy * W_PROXIMITY +
            orientation_energy * W_ORIENTATION +
            collision_penalty +
            joint_limit_penalty * W_JOINT_LIMIT
        )
        return total_energy
    
    def energy(self):
        # 1. 更新物理状态
        state_struct = StateStruct(self.model, self.data)
        state_struct.from_array(self.state)
        self.set_hand_pose(state_struct) # 内部建议改用 mj_fwdPosition

        # 2. 获取瓶子和手的关键数据
        b_pos = self.data.xpos[self.bottle_body_id]
        b_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        points_world = self.data.xpos[self.contact_body_ids]
        
        # 3. 【核心修正】坐标系转换：将手部点转到瓶子局部坐标系
        points_local = (points_world - b_pos) @ b_mat
        
        # 4. 【性能优化】批量计算距离
        from trimesh.proximity import closest_point
        _, dist, _ = closest_point(self.bottle_mesh, points_local)
        
        # 5. 【逻辑修正】计算平均表面距离
        proximity_energy = np.mean(dist)
        
        # 6. 【功能增强】手掌中心也要靠近（给它更高权重）
        palm_center_pos = self.data.site_xpos[self.palm_center_site_id]
        palm_dist = np.linalg.norm(palm_center_pos - b_pos)
        
        # 7. 【关键】手掌必须朝向瓶子（否则它会用手背去贴）
        palm_rot = self.data.xmat[self.palm_body_id].reshape(3, 3)
        palm_normal = -palm_rot[:, 1] # 假设-Y是掌心，请根据RGB颜色核实
        vec_to_bottle = (b_pos - palm_center_pos) / (palm_dist + 1e-6)
        orientation_energy = 1.0 - np.dot(palm_normal, vec_to_bottle)

        # 8. 简单的碰撞惩罚（如果没有这个，手会飞进瓶子中心）
        collision_penalty = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.dist < -0.002: # 穿透超过2mm
                collision_penalty += 10.0

        # 总能量（权重你可以微调）
        total_energy = (
            proximity_energy * 100.0 + 
            palm_dist * 50.0 +
            orientation_energy * 10.0 +
            collision_penalty * 50.0
        )
        
        return total_energy
