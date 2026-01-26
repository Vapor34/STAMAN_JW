import numpy as np


class GraspControl:
    def __init__(self,):
        self.control_vars = []

    # TODO 
    # flattened control variables: [j1, j2, j3, j4, ..., t1, t2, t3, t4]
    # structured control variables: {'j1_name': j1, 'j2_name': j2, ..., 't1_name': t1, 't2_name': t2}