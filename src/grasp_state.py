"""
状态结构封装模块

用于机械手位姿和协同变量的统一管理
"""

import numpy as np


class StateStruct:
    """
    机械手状态结构封装类
    
    封装内容 (9D向量):
    ├─ 位置 (3D): x, y, z
    ├─ 姿态 (4D): qw, qx, qy, qz (四元数)
    ├─ 抓取协同: grasp_synergy [0, 1]
    └─ 展开协同: spread_synergy [-0.2, 0.3]
    """
    
    def __init__(self, position=None, quaternion=None, grasp=0.0, spread=0.0):
        """
        初始化状态结构
        
        Args:
            position (np.ndarray): 3D位置 [x, y, z]，默认为 [0, 0, 0]
            quaternion (np.ndarray): 4D四元数 [qw, qx, qy, qz]，默认为 [1, 0, 0, 0]
            grasp (float): 抓取强度 [0, 1]，默认为 0.0
            spread (float): 展开程度 [-0.2, 0.3]，默认为 0.0
        """
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.quaternion = np.array(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.grasp_synergy = np.clip(grasp, 0.0, 1.0)
        self.spread_synergy = np.clip(spread, -0.2, 0.3)
        self._normalize_quaternion()
    
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
    def spread(self):
        """获取展开程度"""
        return self.spread_synergy
    
    @spread.setter
    def spread(self, value):
        """设置展开程度 [-0.2, 0.3]"""
        self.spread_synergy = np.clip(value, -0.2, 0.3)
    
    # ===== 完整状态访问 =====
    def to_array(self):
        """
        转换为9D数组 [x, y, z, qw, qx, qy, qz, grasp, spread]
        
        Returns:
            np.ndarray: 9维状态向量
        """
        return np.array([
            self.position[0], self.position[1], self.position[2],
            self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3],
            self.grasp_synergy,
            self.spread_synergy
        ], dtype=np.float64)
    
    def from_array(self, arr):
        """
        从9D数组加载状态
        
        Args:
            arr (np.ndarray): 9维状态向量
        """
        if len(arr) != 9:
            raise ValueError(f"Expected 9D array, got {len(arr)}D")
        self.position = np.array(arr[0:3], dtype=np.float64)
        self.quaternion = np.array(arr[3:7], dtype=np.float64)
        self._normalize_quaternion()
        self.grasp_synergy = np.clip(arr[7], 0.0, 1.0)
        self.spread_synergy = np.clip(arr[8], -0.2, 0.3)
    
    def copy(self):
        """
        深拷贝当前状态
        
        Returns:
            StateStruct: 新的状态对象
        """
        return StateStruct(
            position=self.position.copy(),
            quaternion=self.quaternion.copy(),
            grasp=self.grasp_synergy,
            spread=self.spread_synergy
        )
    
    def __repr__(self):
        """字符串表示"""
        return (f"StateStruct("
                f"pos=[{self.x:.3f}, {self.y:.3f}, {self.z:.3f}], "
                f"quat=[{self.qw:.3f}, {self.qx:.3f}, {self.qy:.3f}, {self.qz:.3f}], "
                f"grasp={self.grasp:.3f}, spread={self.spread:.3f})")
    
    def __str__(self):
        """详细字符串表示"""
        return (f"StateStruct:\n"
                f"  Position: ({self.x:.4f}, {self.y:.4f}, {self.z:.4f})\n"
                f"  Quaternion: (w={self.qw:.4f}, x={self.qx:.4f}, y={self.qy:.4f}, z={self.qz:.4f})\n"
                f"  Grasp: {self.grasp:.4f}\n"
                f"  Spread: {self.spread:.4f}")
