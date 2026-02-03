"""
状态结构封装模块

用于机械手位姿和协同变量的统一管理
"""

import numpy as np
import mujoco


class StateStruct:
    """
    hand state
    
    
        base: x, y, z, qw, qx, qy, qz 
        syn_grasp, syn_curl, syn_spread, syn_thumb_base, syn_thumb_flex, syn_wrist

        get and set each variable individually, or get/set the whole state as a 13D array
        
    """
    
    def __init__(self, model, data, position=None, quaternion=None, grasp=0.0, curl=0.0, spread=0.0, thumb_base=0.0, thumb_flex=0.0, wrist=0.0, hand_body_prefix='lh_'):
        """
        initialize StateStruct
        """
        self.model = model
        self.data = data
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.quaternion = np.array(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._normalize_quaternion()
        
        self.grasp_synergy = grasp #synergy value
        self.curl_synergy = curl
        self.spread_synergy = spread
        self.thumb_base_synergy = thumb_base
        self.thumb_flex_synergy = thumb_flex
        self.wrist_synergy = wrist
        self.hand_prefix = hand_body_prefix

        self.state_dic = {}
        self.get_state_dic()
    
    def get_state_dic(self):
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is not 'None' and self.hand_prefix in body_name:
                pos = self.data.xpos[i]
                quaternion = self.data.xquat[i]
                comp = pos.tolist() + quaternion.tolist()
                self.state_dic[body_name] = comp
                # print(body_name, *(f'{x:.2f}' for x in comp))




    



    # ===== 位置访问 =====
    @property
    def x(self):
        """获取X坐标"""
        return self.position[0]
    
    @x.setter
    def x(self, value):
        """设置X坐标"""
        self.position[0] = value
    
    @property
    def y(self):
        """获取Y坐标"""
        return self.position[1]
    
    @y.setter
    def y(self, value):
        """设置Y坐标"""
        self.position[1] = value
    
    @property
    def z(self):
        """获取Z坐标"""
        return self.position[2]
    
    @z.setter
    def z(self, value):
        """设置Z坐标"""
        self.position[2] = value
    
    def get_position(self):
        """获取完整位置向量"""
        return self.position.copy()
    
    def set_position(self, position):
        """设置完整位置向量"""
        self.position = np.array(position, dtype=np.float64)
    
    # ===== 姿态访问 =====
    @property
    def qw(self):
        """获取四元数W分量"""
        return self.quaternion[0]
    
    @qw.setter
    def qw(self, value):
        """设置四元数W分量"""
        self.quaternion[0] = value
        self._normalize_quaternion()
    
    @property
    def qx(self):
        """获取四元数X分量"""
        return self.quaternion[1]
    
    @qx.setter
    def qx(self, value):
        """设置四元数X分量"""
        self.quaternion[1] = value
        self._normalize_quaternion()
    
    @property
    def qy(self):
        """获取四元数Y分量"""
        return self.quaternion[2]
    
    @qy.setter
    def qy(self, value):
        """设置四元数Y分量"""
        self.quaternion[2] = value
        self._normalize_quaternion()
    
    @property
    def qz(self):
        """获取四元数Z分量"""
        return self.quaternion[3]
    
    @qz.setter
    def qz(self, value):
        """设置四元数Z分量"""
        self.quaternion[3] = value
        self._normalize_quaternion()
    
    def get_quaternion(self):
        """获取完整四元数向量"""
        return self.quaternion.copy()
    
    def set_quaternion(self, quaternion):
        """设置完整四元数向量并归一化"""
        self.quaternion = np.array(quaternion, dtype=np.float64)
        self._normalize_quaternion()
    
    def _normalize_quaternion(self):
        """四元数归一化"""
        norm = np.linalg.norm(self.quaternion)
        if norm > 1e-6:
            self.quaternion /= norm
    
    # ===== 协同变量访问 =====
    @property
    def grasp(self):
        """获取抓取强度"""
        return self.grasp_synergy
    
    @grasp.setter
    def grasp(self, value):
        """设置抓取强度 [0, 1]"""
        self.grasp_synergy = np.clip(value, 0.0, 1.0)

    @property
    def curl(self):
        """获取弯曲程度"""
        return self.curl_synergy
    
    @curl.setter
    def curl(self, value):
        """设置弯曲程度 [0, 1]"""
        self.curl_synergy = np.clip(value, 0.0, 1.0)
    
    @property
    def spread(self):
        """获取展开程度"""
        return self.spread_synergy
    
    @spread.setter
    def spread(self, value):
        """设置展开程度 [-0.2, 0.3]"""
        self.spread_synergy = np.clip(value, -0.2, 0.3)

    @property
    def thumb_base(self):
        """获取拇指基部弯曲程度"""
        return self.thumb_base_synergy
    
    @thumb_base.setter
    def thumb_base(self, value):
        """设置拇指基部弯曲程度 [0, 1]"""
        self.thumb_base_synergy = np.clip(value, 0.0, 1.0)

    @property
    def thumb_flex(self):
        """获取拇指指节弯曲程度"""
        return self.thumb_flex_synergy
    
    @thumb_flex.setter
    def thumb_flex(self, value):
        """设置拇指指节弯曲程度 [0, 1]"""
        self.thumb_flex_synergy = np.clip(value, 0.0, 1.0)

    @property
    def wrist(self):
        """获取手腕弯曲程度"""
        return self.wrist_synergy
    
    @wrist.setter
    def wrist(self, value):
        """设置手腕弯曲程度 [0, 1]"""
        self.wrist_synergy = np.clip(value, 0.0, 1.0)
    
    # ===== 完整状态访问 =====
    def to_array(self):
        """
        convert into 13D array [x, y, z, qw, qx, qy, qz, 
                                grasp, curl, spread, thumb_base, thumb_flex, wrist]
        """
        return np.array([
            self.position[0], self.position[1], self.position[2],
            self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3],
            self.grasp_synergy,
            self.curl_synergy,
            self.spread_synergy,
            self.thumb_base_synergy,
            self.thumb_flex_synergy,
            self.wrist_synergy
        ], dtype=np.float64)
    
    def from_array(self, arr):
        """
        load from 13D array
        """
        if len(arr) != 13:
            raise ValueError(f"Expected 13D array, got {len(arr)}D")
        self.position = np.array(arr[0:3], dtype=np.float64)
        self.quaternion = np.array(arr[3:7], dtype=np.float64)
        self._normalize_quaternion()

        self.grasp_synergy = np.clip(arr[7], 0.0, 1.0)
        self.curl_synergy = np.clip(arr[8], 0.0, 1.0)
        self.spread_synergy = np.clip(arr[9], -0.2, 0.3)
        self.thumb_base_synergy = np.clip(arr[10], 0.0, 1.0)
        self.thumb_flex_synergy = np.clip(arr[11], 0.0, 1.0)
        self.wrist_synergy = np.clip(arr[12], 0.0, 1.0)
    
    def copy(self):
        """
        深拷贝当前状态
        
        Returns:
            StateStruct: 新的状态对象
        """
        return StateStruct(
            self.model,
            self.data,
            position=self.position.copy(),
            quaternion=self.quaternion.copy(),
            grasp=self.grasp_synergy,
            curl=self.curl_synergy,
            spread=self.spread_synergy,
            thumb_base=self.thumb_base_synergy,
            thumb_flex=self.thumb_flex_synergy,
            wrist=self.wrist_synergy
        )
    
    def __repr__(self):
        """字符串表示"""
        return (f"StateStruct("
                f"pos=[{self.x:.3f}, {self.y:.3f}, {self.z:.3f}], "
                f"quat=[{self.qw:.3f}, {self.qx:.3f}, {self.qy:.3f}, {self.qz:.3f}], "
                f"grasp={self.grasp:.3f},"
                f"curl={self.curl:.3f}, "
                f"spread={self.spread:.3f}, "
                f"thumb_base={self.thumb_base:.3f}, "
                f"thumb_flex={self.thumb_flex:.3f}, "
                f"wrist={self.wrist:.3f}"
                )
    
    
    def __str__(self):
        """详细字符串表示"""
        return (f"StateStruct:\n"
                f"  Position: ({self.x:.4f}, {self.y:.4f}, {self.z:.4f})\n"
                f"  Quaternion: (w={self.qw:.4f}, x={self.qx:.4f}, y={self.qy:.4f}, z={self.qz:.4f})\n"
                f"  Grasp: {self.grasp:.4f}\n"
                f"  Curl: {self.curl:.4f}\n"
                f"  Spread: {self.spread:.4f}\n"
                f"  Thumb Base: {self.thumb_base:.4f}\n"
                f"  Thumb Flex: {self.thumb_flex:.4f}\n"
                f"  Wrist: {self.wrist:.4f}"
        )
