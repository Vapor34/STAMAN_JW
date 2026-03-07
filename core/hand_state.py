"""
Hand state structure module

Provides unified management of hand pose and synergy variables
"""

import numpy as np
import mujoco


class StateStruct:
    """
    Hand state representation
    
    Contains:
        - Base position(palm center): x, y, z 
        - Base orientation: qw, qx, qy, qz (quaternion)
        - Synergy variables: grasp, curl, spread, thumb_base, thumb_flex, wrist
        
    Supports both individual variable access and complete state as 13D array
    """
    
    def __init__(self, model, data, position=None, quaternion=None, grasp=0.0, curl=0.0, spread=0.0, thumb_base=0.0, thumb_flex=0.0, wrist=0.0, hand_body_prefix='lh_'):
        """
        Initialize StateStruct
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            position: [x, y, z] position vector
            quaternion: [qw, qx, qy, qz] quaternion vector
            grasp: Grasp synergy value [0, 1]
            curl: Curl synergy value [0, 1]
            spread: Spread synergy value [-0.2, 0.3]
            thumb_base: Thumb base synergy value [0, 1]
            thumb_flex: Thumb flex synergy value [0, 1]
            wrist: Wrist synergy value [0, 1]
            hand_body_prefix: Prefix for hand body names
        """
        self.model = model
        self.data = data
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.quaternion = np.array(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._normalize_quaternion()
        
        self.grasp_synergy = grasp
        self.curl_synergy = curl
        self.spread_synergy = spread
        self.thumb_base_synergy = thumb_base
        self.thumb_flex_synergy = thumb_flex
        self.wrist_synergy = wrist
        self.hand_prefix = hand_body_prefix
        self.palm_center_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{hand_body_prefix}palm_center")
        self.offset = np.array([0.0, 0.0, 0.05]) #Base to palm center offset

        self.state_dic = {}
        self.get_state_dic()

    # ===== Position and Quaternion Access =====

    def get_position(self):
        """Get complete position vector"""
        return self.position.copy()
    
    def set_position(self, position):
        """Set complete position vector"""
        self.position = np.array(position, dtype=np.float64)
    
    def get_quaternion(self):
        """Get complete quaternion vector"""
        return self.quaternion.copy()
    
    def set_quaternion(self, quaternion):
        """Set complete quaternion vector and normalize"""
        self.quaternion = np.array(quaternion, dtype=np.float64)
        self._normalize_quaternion()
    
    def _normalize_quaternion(self):
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(self.quaternion)
        if norm > 1e-6:
            self.quaternion /= norm

    def get_state_dic(self):
        """Build dictionary of all hand body states"""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and self.hand_prefix in body_name:
                pos = self.data.xpos[i]
                quaternion = self.data.xquat[i]
                comp = pos.tolist() + quaternion.tolist()
                self.state_dic[body_name] = comp
    
    # ===== Complete State Access (13D Array) =====
    
    def to_array(self):
        """
        Convert state to 13D array format
        
        Returns:
            np.ndarray: [x, y, z, qw, qx, qy, qz, grasp, curl, spread, thumb_base, thumb_flex, wrist]
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
        Load state from 13D array
        
        Args:
            arr: 13D array with state values
            
        Raises:
            ValueError: If array is not 13D
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
        Deep copy current state
        
        Returns:
            StateStruct: New state object with copied values
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
    
    # ===== Position Properties =====
    
    @property
    def x(self):
        """Get X coordinate"""
        return self.position[0]
    
    @x.setter
    def x(self, value):
        """Set X coordinate"""
        self.position[0] = value
    
    @property
    def y(self):
        """Get Y coordinate"""
        return self.position[1]
    
    @y.setter
    def y(self, value):
        """Set Y coordinate"""
        self.position[1] = value
    
    @property
    def z(self):
        """Get Z coordinate"""
        return self.position[2]
    
    @z.setter
    def z(self, value):
        """Set Z coordinate"""
        self.position[2] = value

    # ===== Quaternion Properties =====
    
    @property
    def qw(self):
        """Get quaternion W component"""
        return self.quaternion[0]
    
    @qw.setter
    def qw(self, value):
        """Set quaternion W component"""
        self.quaternion[0] = value
        self._normalize_quaternion()
    
    @property
    def qx(self):
        """Get quaternion X component"""
        return self.quaternion[1]
    
    @qx.setter
    def qx(self, value):
        """Set quaternion X component"""
        self.quaternion[1] = value
        self._normalize_quaternion()
    
    @property
    def qy(self):
        """Get quaternion Y component"""
        return self.quaternion[2]
    
    @qy.setter
    def qy(self, value):
        """Set quaternion Y component"""
        self.quaternion[2] = value
        self._normalize_quaternion()
    
    @property
    def qz(self):
        """Get quaternion Z component"""
        return self.quaternion[3]
    
    @qz.setter
    def qz(self, value):
        """Set quaternion Z component"""
        self.quaternion[3] = value
        self._normalize_quaternion()

    # ===== Synergy Properties =====
    
    @property
    def grasp(self):
        """Get grasp strength synergy"""
        return self.grasp_synergy
    
    @grasp.setter
    def grasp(self, value):
        """Set grasp strength synergy [0, 1]"""
        self.grasp_synergy = np.clip(value, 0.0, 1.0)

    @property
    def curl(self):
        """Get curl synergy"""
        return self.curl_synergy
    
    @curl.setter
    def curl(self, value):
        """Set curl synergy [0, 1]"""
        self.curl_synergy = np.clip(value, 0.0, 1.0)
    
    @property
    def spread(self):
        """Get spread synergy"""
        return self.spread_synergy
    
    @spread.setter
    def spread(self, value):
        """Set spread synergy [-0.2, 0.3]"""
        self.spread_synergy = np.clip(value, -0.2, 0.3)

    @property
    def thumb_base(self):
        """Get thumb base synergy"""
        return self.thumb_base_synergy
    
    @thumb_base.setter
    def thumb_base(self, value):
        """Set thumb base synergy [0, 1]"""
        self.thumb_base_synergy = np.clip(value, 0.0, 1.0)

    @property
    def thumb_flex(self):
        """Get thumb flex synergy"""
        return self.thumb_flex_synergy
    
    @thumb_flex.setter
    def thumb_flex(self, value):
        """Set thumb flex synergy [0, 1]"""
        self.thumb_flex_synergy = np.clip(value, 0.0, 1.0)

    @property
    def wrist(self):
        """Get wrist synergy"""
        return self.wrist_synergy
    
    @wrist.setter
    def wrist(self, value):
        """Set wrist synergy [0, 1]"""
        self.wrist_synergy = np.clip(value, 0.0, 1.0)
    
    def __repr__(self):
        """String representation"""
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
        """Detailed string representation"""
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
