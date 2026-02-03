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
            act_name_to_id:{act_name:act_index, ...}
            act_id_to_name: {act_index: act_name, ...}
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
        self.act_name_to_id ={'lh_WRJ2': 0, 'lh_WRJ1': 1,
                            'lh_THJ5': 2, 'lh_THJ4': 3, 'lh_THJ3': 4, 'lh_THJ2': 5, 'lh_THJ1': 6,
                            'lh_FFJ4': 7, 'lh_FFJ3': 8, 'lh_FFJ0': 9, 
                            'lh_MFJ4': 10, 'lh_MFJ3': 11, 'lh_MFJ0': 12, 
                            'lh_RFJ4': 13, 'lh_RFJ3': 14, 'lh_RFJ0': 15, 
                            'lh_LFJ5': 16, 'lh_LFJ4': 17, 'lh_LFJ3': 18, 'lh_LFJ0': 19}
        self.act_id_to_name = {v: k for k, v in self.act_name_to_id.items()}
        self.synergy_map = {
            'grasp':['lh_FFJ3', 'lh_MFJ3', 'lh_RFJ3', 'lh_LFJ3', 'lh_LFJ5'],
            'curl':['lh_FFJ0', 'lh_MFJ0', 'lh_RFJ0', 'lh_LFJ0'],
            'spread':['lh_FFJ4', 'lh_MFJ4', 'lh_RFJ4', 'lh_LFJ4'],
            'thumb_base':['lh_THJ5', 'lh_THJ4'],
            'thumb_flex':['lh_THJ3', 'lh_THJ2', 'lh_THJ1'],
            'wrist':['lh_WRJ1', 'lh_WRJ2']
        }    
        self.joint_range = {}
        self._get_joint_ranges()

    def _get_joint_ranges(self):
        for i in range(1, self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name is not None:
                low, high = self.model.jnt_range[i]
                self.joint_range[jnt_name] = (low, high)
                


    # def _build_act_name_to_id_mapping(self):
    #     self.act_name_dic = {}
    #     for act_id in range(self.model.nu):
    #     # 判断执行器是否作用于腱 (mjTRN_TENDON = 3)
    #         if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_TENDON:
    #             # 获取该执行器关联的 Tendon ID
    #             t_id = self.model.actuator_trnid[act_id, 0]
    #             # 获取腱的名字
    #             t_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, t_id)
    #             # 记录映射
    #             self.act_name_dic[t_name] = act_id
    #         elif self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
    #             # 获取该执行器关联的 Joint ID
    #             j_id = self.model.actuator_trnid[act_id, 0]
    #             # 获取关节的名字
    #             j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
    #             # 记录映射
    #             self.act_name_dic[j_name] = act_id
    #     """OUTPUT: act_name_dic: {'lh_WRJ2': 0, 'lh_WRJ1': 1,
    #                         'lh_THJ5': 2, 'lh_THJ4': 3, 'lh_THJ3': 4, 'lh_THJ2': 5, 'lh_THJ1': 6,
    #                         'lh_FFJ4': 7, 'lh_FFJ3': 8, 'lh_FFJ0': 9, 
    #                         'lh_MFJ4': 10, 'lh_MFJ3': 11, 'lh_MFJ0': 12, 
    #                         'lh_RFJ4': 13, 'lh_RFJ3': 14, 'lh_RFJ0': 15, 
    #                         'lh_LFJ5': 16, 'lh_LFJ4': 17, 'lh_LFJ3': 18, 'lh_LFJ0': 19}"""
   
    def get_act_name_list(self):
        """get actuator name list"""
        return self.act_name_list
    
    def get_act_name_to_id_dic(self):
        """get actuator name dictionary"""
        return self.act_name_to_id
    
    def get_act_id_to_name_dic(self):
        """get actuator id to name dictionary"""
        return self.act_id_to_name

    def set_act_val(self, act_name, value):
        """set single actuator control value"""
        act_id = self.act_name_to_id.get(act_name)
        if act_id is not None:
            self.data.ctrl[act_id] = value
        else:
            raise ValueError(f"Unknown actuator name: {act_name}")
            
    def get_act_val(self, act_name):
        """get single actuator control value"""
        act_id = self.act_name_to_id.get(act_name)
        if act_id is not None:
            print(f"value now: {self.data.ctrl[act_id]} ")
            return self.data.ctrl[act_id]
        else:
            raise ValueError(f"Unknown actuator name: {act_name}")
        
    def get_act_name(self, act_id):
        """get single actuator name by id"""
        act_name = self.act_id_to_name.get(act_id)
        if act_name is not None:
            return act_name
        else:
            raise ValueError(f"Unknown actuator id: {act_id}")
        
    def set_syn_val(self, group, value):
        """
        set synergy group value

        haven't decided the weight of each actuator in the synergy group yet
        """
        if group not in self.synergy_map:
            raise ValueError(f"Unknown synergy group: {group}")
        act_names = self.synergy_map[group]
        for act_name in act_names:
            self.set_act_val(act_name, value)

    def set_hand_state(self, state: StateStruct):
        """
        set hand state from StateStruct
        """
        self.set_syn_val('grasp', state.grasp_synergy)
        self.set_syn_val('curl', state.curl_synergy)
        self.set_syn_val('spread', state.spread_synergy)
        self.set_syn_val('thumb_base', state.thumb_base_synergy)
        self.set_syn_val('thumb_flex', state.thumb_flex_synergy)
        self.set_syn_val('wrist', state.wrist_synergy)
    
    def get_synergy_map(self):
        """get synergy map"""
        return self.synergy_map
    
    def print_all_act_val(self):
        """print all actuator control values"""
        print("\n【当前执行器控制值】")
        for group, act_names in self.synergy_map.items():
            print(f"  Synergy Group: {group}")
            for act_name in act_names:
                act_id = self.act_name_to_id[act_name]
                act_val = self.data.ctrl[act_id]
                print(f"    {act_name}: {act_val:.4f}")
