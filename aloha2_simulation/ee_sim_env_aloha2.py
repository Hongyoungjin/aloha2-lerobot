import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE


from utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
# import mujoco_py
import IPython
e = IPython.embed

START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.008, 0.008,  0, -0.96, 1.16, 0, -0.3, 0, 0.008, 0.008]

import os 
os.environ['MUJOCO_GL'] = 'glfw'

def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube_human' in task_name:
        
        xml_path = os.path.join(XML_DIR, f'sim_cube_transfer.xml')
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        env = SimTransferCube(model,data)
    return env


aloha2_gripper_value = lambda x: 0.002 + (0.037 - 0.002)*((x+0.04)/0.94) 

class SimTransferCube():
    def __init__(self,model,data) :
        self.model = model
        self.data = data
        self.cam_names = ['overhead_cam','worms_eye_cam','wrist_cam_left','wrist_cam_right']
        self.render_cam = 'teleoperator_pov'
    
    def reset(self):
        obs = collections.OrderedDict()
        obs['images'] = dict()

        with mujoco.Renderer(self.model,height=480,width=640) as renderer:
            mujoco.mj_step(self.model,self.data)
            self.initialize_robot()
            for cam in self.cam_names:
                obs['images'][f'{cam}'] = self.get_images(cam,renderer)
            obs['qpos'] = self.data.qpos.copy()
            obs['qvel'] = self.data.qvel.copy()
            obs['render'] = self.get_images(renderer=renderer,cam_name=self.render_cam)
        return obs

    def step(self,action):

        obs = collections.OrderedDict()
        obs['images'] = dict()
        
        with mujoco.Renderer(self.model,height=480,width=640) as renderer:
            mujoco.mj_step(self.model,self.data)    
            self.set_action(action = action)
            for cam in self.cam_names:
                obs['images'][f'{cam}'] = self.get_images(cam,renderer)
            obs['qpos'] = self.data.qpos.copy()
            obs['qvel'] = self.data.qvel.copy()
            obs['render'] = self.get_images(renderer=renderer,cam_name=self.render_cam)
        return obs
    
    def initialize_robot(self):    
        self.data.qpos[0:16] = START_ARM_POSE

    
    def set_action(self,action):
        action[6],action[13] = aloha2_gripper_value(action[6]),aloha2_gripper_value(action[13])
        self.data.ctrl[:] = action

        
    def get_images(self,cam_name,renderer):
        renderer.update_scene(self.data, camera= cam_name)
        image = renderer.render()
        
        # print(image.shape)
        return image
    
