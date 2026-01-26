"""
Shadow Hand 手动控制工具
在 MuJoCo 查看器中实时控制机械手的每个关节

按键映射：
  数字键 0-9：控制不同关节
  上/下方向键：增加/减少关节角度
  R 键：重置所有关节到初始位置
  Q 键：退出程序
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from src.mujoco_utils import mujoco_load


class ShadowHandController:
    """Shadow Hand 手动控制器"""
    
    def __init__(self, model_path="shadow_hand/scene_left.xml"):
        """初始化模型和数据"""
        self.model, self.data = mujoco_load(model_path)
        
        # 构建关节映射
        self.joint_info = self._build_joint_info()
        self.selected_joint_idx = 0  # 当前选中的关节
        
        # 控制参数
        self.angle_step = 0.05  # 每次增减的角度（弧度）
        self.max_step_size = 0.1  # 最大步长
        
        print("\n" + "="*70)
        print("Shadow Hand 手动控制器")
        print("="*70)
        print("\n【可控关节列表】")
        self._print_joint_info()
        print("\n【按键说明】")
        print("  0-9：选择关节（按照上面的编号）")
        print("  ↑/↓：增加/减少选中关节角度")
        print("  R：重置所有关节")
        print("  Q：退出")
        print("="*70 + "\n")
    
    def _build_joint_info(self):
        """构建所有手部关节的信息表"""
        joint_info = []
        
        for jnt_id in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            
            # 只考虑手部关节（以 lh_ 开头的左手或 rh_ 开头的右手）
            if jnt_name and ('lh_' in jnt_name or 'rh_' in jnt_name):
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                qpos_range = self.model.jnt_range[jnt_id]
                
                joint_info.append({
                    'id': jnt_id,
                    'name': jnt_name,
                    'qpos_adr': qpos_adr,
                    'range': qpos_range,
                    'current_val': self.data.qpos[qpos_adr]
                })
        
        return joint_info
    
    def _print_joint_info(self):
        """打印所有关节信息"""
        for idx, info in enumerate(self.joint_info):
            range_str = f"[{info['range'][0]:.3f}, {info['range'][1]:.3f}]"
            current_val = self.data.qpos[info['qpos_adr']]
            marker = "→ 当前选中" if idx == self.selected_joint_idx else ""
            print(f"  [{idx:2d}] {info['name']:20s} 范围: {range_str}  当前: {current_val:7.3f}  {marker}")
    
    def select_joint(self, idx):
        """选择要控制的关节"""
        if 0 <= idx < len(self.joint_info):
            self.selected_joint_idx = idx
            info = self.joint_info[idx]
            print(f"✓ 已选中: {info['name']} (范围: [{info['range'][0]:.3f}, {info['range'][1]:.3f}])")
            return True
        return False
    
    def increase_angle(self):
        """增加选中关节的角度"""
        info = self.joint_info[self.selected_joint_idx]
        current_val = self.data.qpos[info['qpos_adr']]
        new_val = current_val + self.angle_step
        new_val = np.clip(new_val, info['range'][0], info['range'][1])
        self.data.qpos[info['qpos_adr']] = new_val
        delta = new_val - current_val
        print(f"  {info['name']}: {current_val:.4f} → {new_val:.4f} (Δ={delta:+.4f})")
    
    def decrease_angle(self):
        """减少选中关节的角度"""
        info = self.joint_info[self.selected_joint_idx]
        current_val = self.data.qpos[info['qpos_adr']]
        new_val = current_val - self.angle_step
        new_val = np.clip(new_val, info['range'][0], info['range'][1])
        self.data.qpos[info['qpos_adr']] = new_val
        delta = new_val - current_val
        print(f"  {info['name']}: {current_val:.4f} → {new_val:.4f} (Δ={delta:+.4f})")
    
    def reset_all(self):
        """重置所有关节到初始位置"""
        mujoco.mj_resetData(self.model, self.data)
        print("✓ 所有关节已重置到初始位置")
    
    def set_joint_angle(self, joint_idx, angle):
        """直接设置指定关节的角度"""
        if 0 <= joint_idx < len(self.joint_info):
            info = self.joint_info[joint_idx]
            angle = np.clip(angle, info['range'][0], info['range'][1])
            self.data.qpos[info['qpos_adr']] = angle
            return True
        return False
    
    def get_joint_angle(self, joint_idx):
        """获取指定关节的当前角度"""
        if 0 <= joint_idx < len(self.joint_info):
            info = self.joint_info[joint_idx]
            return self.data.qpos[info['qpos_adr']]
        return None
    
    def print_current_state(self):
        """打印当前所有关节的状态"""
        print("\n【当前关节状态】")
        for idx, info in enumerate(self.joint_info):
            current_val = self.data.qpos[info['qpos_adr']]
            range_str = f"[{info['range'][0]:.3f}, {info['range'][1]:.3f}]"
            marker = "← 当前选中" if idx == self.selected_joint_idx else ""
            print(f"  [{idx:2d}] {info['name']:20s} = {current_val:7.3f}  范围: {range_str}  {marker}")


def run_interactive_control():
    """运行交互式控制"""
    controller = ShadowHandController()
    
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        print("启动 MuJoCo 查看器...\n")
        
        while viewer.is_running():
            step_start = time.time()
            
            # 执行物理仿真步骤
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            
            # 控制帧率
            time_until_next_step = controller.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def run_demo():
    """运行演示：依次控制各关节"""
    model, data = mujoco_load("shadow_hand/scene_left.xml")
    controller = ShadowHandController()
    
    print("运行演示：依次控制每个关节...")
    print("="*70 + "\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for demo_step in range(5):  # 循环5次
            for joint_idx in range(len(controller.joint_info)):
                info = controller.joint_info[joint_idx]
                mid_range = (info['range'][0] + info['range'][1]) / 2
                
                print(f"\n演示: {info['name']}")
                
                # 缓慢移动到中间位置
                current = data.qpos[info['qpos_adr']]
                target = mid_range
                steps = 20
                
                for step in range(steps):
                    progress = (step + 1) / steps
                    new_val = current + (target - current) * progress
                    data.qpos[info['qpos_adr']] = new_val
                    
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.02)
                
                # 返回初始位置
                print(f"  返回初始位置...")
                current = data.qpos[info['qpos_adr']]
                steps = 10
                
                for step in range(steps):
                    progress = (step + 1) / steps
                    new_val = current + (info['range'][0] - current) * progress
                    data.qpos[info['qpos_adr']] = new_val
                    
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.02)
                
                # 重置
                mujoco.mj_resetData(model, data)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # 运行自动演示
        run_demo()
    else:
        # 运行交互式控制
        print("\n使用方法:")
        print("  python manual_control.py          # 交互式控制")
        print("  python manual_control.py demo     # 运行自动演示\n")
        
        try:
            run_interactive_control()
        except KeyboardInterrupt:
            print("\n程序已退出")
