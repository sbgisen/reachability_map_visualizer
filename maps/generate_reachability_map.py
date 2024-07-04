# coding: utf-8
import rospy; rospy.init_node('test_ipython2')
import numpy as np
import torch
import curobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.robot import RobotConfig
from tf.transformations import quaternion_from_euler

## preapre to solve IK
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_properties = torch.cuda.get_device_properties(device)
    total_memory = gpu_properties.total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory

    print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Allocated GPU memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Free GPU memory: {free_memory / (1024 ** 3):.2f} GB")

    # Assuming you want to store a float32 array
    dtype_size = torch.tensor(0, dtype=torch.float32).element_size()  # Size of float32 in bytes
    max_elements = free_memory // dtype_size
    max_dim_size = int(np.cbrt(max_elements))
    print(f"Max dimensions: {max_dim_size}")

tensor_args = TensorDeviceType()
robot_config = rospy.get_param('/curobo/robot_config')
robot_cfg = RobotConfig.from_dict(robot_config)
base_link = robot_cfg.kinematics.kinematics_config.base_link
ik_config = IKSolverConfig.load_from_robot_config(robot_cfg, rotation_threshold=0.05, position_threshold=0.005, num_seeds=20, self_collision_check=True, self_collision_opt=True, tensor_args=tensor_args, use_cuda_graph=True)
ik_solver = IKSolver(ik_config)

def create_positions(min_x, max_x, min_y, max_y, min_z, max_z, resolution=0.05):
    x_array = np.arange(min_x, max_x + resolution, resolution)
    y_array = np.arange(min_y, max_y + resolution, resolution)
    z_array = np.arange(min_z, max_z + resolution, resolution)
    X, Y, Z = np.meshgrid(x_array, y_array, z_array)
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return positions

def create_orientation(r, p, y):
    orientation = quaternion_from_euler(r, p, y).tolist()
    orientation.insert(0, orientation.pop()) #w, x, y, z
    return orientation

def create_goal_batch(positions, orientation, max_dim_size):
    size = positions.shape[0]
    potential_divisors = np.arange(size // 2, 0, -1)
    divisors = potential_divisors[size % potential_divisors == 0]
    for d in divisors:
        if d < max_dim_size:
            batch_size = d
            break
    else:
        batch_size = 1
    print(f'batch size {batch_size}')

    goal_list = []
    for index in range(0, size, batch_size):
        position_batch = positions[index:index + batch_size]
        goal_batch = []
        for p in position_batch:
            goal_batch.append(p.tolist() + orientation)
        goal_list.append(goal_batch)
    return goal_list

def calculate_ik(goals, tensor_args):
    result_position = []
    result_success = []
    for goal_batch in goals:
        batch = Pose.from_batch_list(goal_batch, tensor_args)
        result = ik_solver.solve_batch(batch)
        result_position.append(result.goal_pose.position.cpu().numpy())
        result_success.append(result.success.cpu().numpy().T.flatten().astype(int) * 100)
        torch.cuda.empty_cache()

    combined_data_list = []
    for p, m in zip(result_position, result_success):
        # mask = m > 50
        # filter_p = p[mask]
        # filter_m = m[mask]
        # combined_data = np.hstack((filter_p, filter_m[:, np.newaxis])) # (point, manipulability)
        combined_data = np.hstack((p, m[:, np.newaxis])) # (point, manipulability)
        combined_data_list.append(combined_data)
    return np.vstack(combined_data_list)

## parameters
ee_left = create_orientation(-np.pi, -np.pi/2, np.pi/2)
ee_right = create_orientation(0, -np.pi/2, np.pi/2)
ee_front = create_orientation(np.pi/2, -np.pi/2, np.pi/2)
ee_top = create_orientation(np.pi, 0, 0)
positions = create_positions(-0.5, 1.0, -1.0, 1.0, 0.0, 1.8, 0.05)


## save to h5
import h5py
def save_results(combined_data, resolution=0.05, frame_id='base_footprint'):
    with h5py.File('test.h5', 'w') as f:
        sphere_group = f.create_group('/Spheres')
        sphere_data = sphere_group.create_dataset('sphere_dataset', data=combined_data)
        sphere_data.attrs.create('Resolution', data=resolution)
        sphere_data.attrs.create('BaseLink', data=frame_id)
        dummy_poses = np.hstack((combined_data[:, :3], np.zeros((combined_data.shape[0], 7))))
        pose_group = f.create_group('/Poses')
        pose_data = pose_group.create_dataset('poses_dataset', dtype=float, data=dummy_poses)

### combined reachable poses
# reach_list = []
# for q in [ee_left, ee_right, ee_front, ee_top]:
#     goals = create_goal_batch(positions, q, max_dim_size)
#     combined_data = calculate_ik(goals, tensor_args)
#     reach_list.append(combined_data)
# import copy
# all_combined_data = copy.deepcopy(reach_list[0])
# all_combined_data[:, 3] = 0.0 # reset score
# for combined_data in reach_list:
#     for i, data in enumerate(combined_data):
#         score = 25.0 if d[3] > 50 else 0
#         all_combined_data[i, 3] += score
# save_results(all_combined_data)
