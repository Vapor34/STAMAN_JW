"""
grasp control module
you can use this module to get and set the actuator(joint/tendon) control values if you know the actuator names.

two representations are provided:
1. (Flattened)act_name_list:[actuator_name1, actuator_name2, ...]
2. (Structured)act_name_dic:{act_name:act_index, ...}

two main methods are provided:
- get_act_name_list(): get actuator name list
- get_act_name_dic(): get actuator name dictionary
- set_act_val(joint_name, value): set single joint control value
- get_act_val(joint_name): get single joint control value
"""

import numpy as np
import mujoco
from src.grasp_state import StateStruct


class GraspControl:

    def __init__(self, model, data):
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
            planner: GraspPlanner 实例（包含 flex_adrs, abd_adrs, thumb_adrs)
        Attributes:
            act_name_list: [actuator_name1, actuator_name2, ...]
            act_name_dic:{act_name:act_index, ...}
        """
        self.model = model
        self.data = data

        self.act_name_list = [
            "lh_WRJ2","lh_WRJ1",
            "lh_THJ5","lh_THJ4","lh_THJ3","lh_THJ2","lh_THJ1",
            "lh_FFJ4","lh_FFJ3","lh_FFJ0",
            "lh_MFJ4","lh_MFJ3","lh_MFJ0",
            "lh_RFJ4","lh_RFJ3","lh_RFJ0",
            "lh_LFJ5","lh_LFJ4","lh_LFJ3","lh_LFJ0"
        ]
        

                
        # mapping from actuator qpos address to joint name
        self._build_act_name_to_id_mapping()
        
        # 初始化控制变量（列表表示）
        num_acts = len(self.act_name_list)
        self.ctrl_act_list = np.zeros(num_acts)
    
    def _build_act_name_to_id_mapping(self):
        self.act_name_dic = {}
        for act_id in range(self.model.nu):
        # 判断执行器是否作用于腱 (mjTRN_TENDON = 3)
            if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_TENDON:
                # 获取该执行器关联的 Tendon ID
                t_id = self.model.actuator_trnid[act_id, 0]
                # 获取腱的名字
                t_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, t_id)
                # 记录映射
                self.act_name_dic[t_name] = act_id
            elif self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                # 获取该执行器关联的 Joint ID
                j_id = self.model.actuator_trnid[act_id, 0]
                # 获取关节的名字
                j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
                # 记录映射
                self.act_name_dic[j_name] = act_id
        """act_name_dic: {'lh_WRJ2': 0, 'lh_WRJ1': 1,
                            'lh_THJ5': 2, 'lh_THJ4': 3, 'lh_THJ3': 4, 'lh_THJ2': 5, 'lh_THJ1': 6,
                            'lh_FFJ4': 7, 'lh_FFJ3': 8, 'lh_FFJ0': 9, 
                            'lh_MFJ4': 10, 'lh_MFJ3': 11, 'lh_MFJ0': 12, 
                            'lh_RFJ4': 13, 'lh_RFJ3': 14, 'lh_RFJ0': 15, 
                            'lh_LFJ5': 16, 'lh_LFJ4': 17, 'lh_LFJ3': 18, 'lh_LFJ0': 19}"""
   
    def get_act_name_list(self):
        """get actuator name list"""
        return self.act_name_list
    
    def get_act_name_dic(self):
        """get actuator name dictionary"""
        return self.act_name_dic

    def set_act_val(self, joint_name, value):
        """set single joint control value"""
        for act_name, act_id in self.act_name_dic.items():
            if act_name == joint_name:
                self.data.ctrl[act_id] = value
        
        
    
    def get_act_val(self, joint_name):
        """get single joint control value"""
        
        for act_name, act_id in self.act_name_dic.items():
            if act_name == joint_name:
                return self.data.ctrl[act_id]
        else:
            raise ValueError(f"unknown actuator name: {joint_name}")
    
