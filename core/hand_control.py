"""
Hand control module

Manages actuator control values and synergy variables for the Shadow Hand.
Provides both individual actuator access and structured synergy group control.
"""

import numpy as np
import mujoco
from core.hand_state import StateStruct


class HandControl:
    """
    Control interface for Shadow Hand actuators
    
    Provides:
    - Individual actuator value access
    - Synergy group mapping and control
    - Joint range queries
    
    Two representations are supported:
    1. (Flattened) act_name_list: [actuator_name1, actuator_name2, ...]
    2. (Structured) act_name_dic: {act_name: act_index, ...}
    """

    def __init__(self, model, data):
        """
        Initialize hand control system
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            
        Attributes:
            act_name_list: Ordered list of actuator names
            act_name_to_id: Mapping from actuator name to control index
            act_id_to_name: Reverse mapping from index to name
            synergy_map: Mapping of synergy groups to actuator lists
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

        self.jnt_range_dic = {}
        self._get_jnt_range_dic()

    def _get_joint_ranges(self):
        """Extract joint range limits from model"""
        for i in range(1, self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name is not None:
                low, high = self.model.jnt_range[i]
                self.joint_range[jnt_name] = (low, high)

    def set_synergy_map(self, synergy_map):
        """
        Change the synergy grouping
        
        Args:
            synergy_map: Dictionary mapping group names to actuator name lists
        """
        self.synergy_map = synergy_map

    def get_act_name_list(self):
        """Get actuator name list"""
        return self.act_name_list
    
    def get_act_name_to_id_dic(self):
        """Get actuator name to ID mapping"""
        return self.act_name_to_id
    
    def get_act_id_to_name_dic(self):
        """Get actuator ID to name mapping"""
        return self.act_id_to_name

    def set_act_val(self, act_name, value):
        """
        Set single actuator control value
        
        Args:
            act_name: Name of the actuator
            value: Control value to set
            
        Raises:
            ValueError: If actuator name is unknown
        """
        act_id = self.act_name_to_id.get(act_name)
        if act_id is not None:
            self.data.ctrl[act_id] = value
        else:
            raise ValueError(f"Unknown actuator name: {act_name}")
            
    def get_act_val(self, act_name):
        """
        Get single actuator control value
        
        Args:
            act_name: Name of the actuator
            
        Returns:
            float: Current control value
            
        Raises:
            ValueError: If actuator name is unknown
        """
        act_id = self.act_name_to_id.get(act_name)
        if act_id is not None:
            return self.data.ctrl[act_id]
        else:
            raise ValueError(f"Unknown actuator name: {act_name}")
        
    def get_act_name(self, act_id):
        """
        Get actuator name by ID
        
        Args:
            act_id: Index of the actuator
            
        Returns:
            str: Name of the actuator
            
        Raises:
            ValueError: If actuator ID is unknown
        """
        act_name = self.act_id_to_name.get(act_id)
        if act_name is not None:
            return act_name
        else:
            raise ValueError(f"Unknown actuator id: {act_id}")
        
    def set_syn_val(self, group, value):
        """
        Set all actuators in a synergy group to the same value
        
        Args:
            group: Name of the synergy group
            value: Value to apply to all actuators in the group
            
        Raises:
            ValueError: If synergy group name is unknown
        """
        if group not in self.synergy_map:
            raise ValueError(f"Unknown synergy group: {group}")
        act_names = self.synergy_map[group]
        for act_name in act_names:
            self.set_act_val(act_name, value)

    def set_hand_state(self, state: StateStruct):
        """
        Set hand state from StateStruct
        
        Args:
            state: StateStruct instance with synergy values
        """
        self.set_syn_val('grasp', state.grasp_synergy)
        self.set_syn_val('curl', state.curl_synergy)
        self.set_syn_val('spread', state.spread_synergy)
        self.set_syn_val('thumb_base', state.thumb_base_synergy)
        self.set_syn_val('thumb_flex', state.thumb_flex_synergy)
        self.set_syn_val('wrist', state.wrist_synergy)
    
    def get_synergy_map(self):
        """Get complete synergy map"""
        return self.synergy_map
    
    def print_all_act_val(self):
        """Print all current actuator control values"""
        print("\n【Current Actuator Control Values】")
        for group, act_names in self.synergy_map.items():
            print(f"  Synergy Group: {group}")
            for act_name in act_names:
                act_id = self.act_name_to_id[act_name]
                act_val = self.data.ctrl[act_id]
                print(f"    {act_name}: {act_val:.4f}")

    def _get_jnt_range_dic(self):
        """Build joint range dictionary"""
        for i in range(1, self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name:
                lower, upper = self.model.jnt_range[i]
                self.jnt_range_dic[jnt_name] = [lower, upper]
