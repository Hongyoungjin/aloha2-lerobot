import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from aloha_scripts.constants import DT, START_ARM_POSE, TASK_CONFIGS
from constants import SIM_TASK_CONFIGS
from aloha_scripts.constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from aloha_scripts.robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from aloha_scripts.robot_utils import move_arms, torque_on, torque_off, move_grippers
from aloha_scripts.real_env import make_real_env, get_action
from ee_sim_env_aloha2 import make_ee_sim_env
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import matplotlib.pyplot as plt
import IPython
e = IPython.embed


def opening_ceremony(master_bot_left, master_bot_right):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors
    
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit


    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, master_bot_right], [start_arm_qpos] * 2, move_time=1.5)

    # move grippers to starting position
    move_grippers([master_bot_left, master_bot_right], [MASTER_GRIPPER_JOINT_MID] * 2, move_time=0.5)


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -0.3
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')

def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite, task_name):
    print(f'Dataset name: {dataset_name}')

    # source of data

    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    
    
    
    # creating real and sim environments
    
    # real_env = make_real_env(init_node=False, setup_robots=False)
    sim_env = make_ee_sim_env(task_name=task_name)

    # cam_names = ['wrist_cam_left','wrist_cam_right','overhead_cam','wormseye_cam']

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(master_bot_left, master_bot_right)

    time

    # Data collection 



    
    # real_ts = make_real_env.reset(fake=True)
    sim_ts =  sim_env.reset() 
    # real_timesteps = [real_ts]
    sim_timesteps = [sim_ts]
    actions = []
    actual_dt_history = []
    # render_cam_name = 'wormseye_cam'
    render_cam_name = 'teleoperater_pov'


    #on screen rendering
    ax = plt.subplot()
    plt_img = ax.imshow(sim_ts['render'])
    plt.ion()

    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #

        # get action data from real robot
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time() #

        print(f'time to get actions from real robot: {t1-t0}s')
        #move to next time step on both the environments and using actions from real robots for sim env
        # real_ts = real_env.step(action)
        sim_ts = sim_env.step(action)

        t2 = time.time() #

        print(f'time to update in simulation: {t2-t1}s')
        # real_timesteps.append(real_ts)
        sim_timesteps.append(sim_ts)
        
        #append all actions for dataset
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])

        # onscreen rendering 
        plt_img.set_data(sim_ts['render'])
        plt.pause(0.02)
        t3 = time.time() #
        print(f'time to update on screen: {t3-t2}s')
    plt.close()

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    # Open puppet grippers
  

    freq_mean = print_dt_diagnosis(actual_dt_history)

    # if freq_mean < 42:
    #     print('freq mean error had to reset')
    #     return False

    

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    
    action                  (14,)         'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
      
        ts = sim_timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts['qpos'])
        data_dict['/observations/qvel'].append(ts['qvel'])
        
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])

    # debug()
    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes = 1024**2*2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), )
            
        _ = obs.create_dataset('qpos', (max_timesteps, 23))
        _ = obs.create_dataset('qvel', (max_timesteps, 22))
        _ = root.create_dataset('action', (max_timesteps, 14))

        for name, array in data_dict.items():

            root[name][...] = array

    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):
    task_config = SIM_TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite, task_name=args['task_name'])
        if is_healthy:
            break

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='aloha_wear_shoe')
    parser.add_argument('--episode_idx', type=int, default=None)
    args = vars(parser.parse_args())
    main(args)