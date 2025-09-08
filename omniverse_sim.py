"""Script to play a checkpoint if an RL agent from RSL-RL."""
from __future__ import annotations


"""Launch Isaac Sim Simulator first."""
import argparse

import cli_args  
import time
import os
import threading

def parse_args(args=None):
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    # parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--custom_env", type=str, default="office", help="Setup the environment")
    parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
    parser.add_argument("--terrain", type=str, default="rough", help="Setup the robot")
    parser.add_argument("--robot_amount", type=int, default=1, help="Setup the robot amount")
    # append AppLauncher cli args
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(args)

# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)


def sub_keyboard_event(event, *args, **kwargs) -> bool:
    import custom_rl_env
    import carb

    if len(custom_rl_env._base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_rl_env._base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_rl_env._base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_rl_env._base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_rl_env._base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_rl_env._base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_rl_env._base_command["0"] = [0, 0, -1]

            if len(custom_rl_env._base_command) > 1:
                if event.input.name == 'I':
                    custom_rl_env._base_command["1"] = [1, 0, 0]
                if event.input.name == 'K':
                    custom_rl_env._base_command["1"] = [-1, 0, 0]
                if event.input.name == 'J':
                    custom_rl_env._base_command["1"] = [0, 1, 0]
                if event.input.name == 'L':
                    custom_rl_env._base_command["1"] = [0, -1, 0]
                if event.input.name == 'U':
                    custom_rl_env._base_command["1"] = [0, 0, 1]
                if event.input.name == 'O':
                    custom_rl_env._base_command["1"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env._base_command)):
                custom_rl_env._base_command[str(i)] = [0, 0, 0]
    return True


def setup_custom_env(args_cli=None):
    import isaaclab.sim as sim_utils
    try:
        if (args_cli.custom_env == "warehouse" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/warehouse.usd")
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if (args_cli.custom_env == "office" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/office.usd")
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))
    except:
        print("Error loading custom environment. You should download custom envs folder from: https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing")


def cmd_vel_cb(msg, num_robot):
    import custom_rl_env
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    custom_rl_env._base_command[str(num_robot)] = [x, y, z]


# This function is related with ros2
def add_cmd_sub(num_envs):
    node_test = rclpy.create_node('position_velocity_publisher')
    for i in range(num_envs):
        node_test.create_subscription(Twist, f'robot{i}/cmd_vel', lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10)
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
    thread.start()



def specify_cmd_for_robots(numv_envs):
    import custom_rl_env
    for i in range(numv_envs):
        custom_rl_env._base_command[str(i)] = [0, 0, 0]


def run_sim(args_cli=None):
    """
    Main entry. Ensures argument parsing happens before launching AppLauncher.
    """

    # Parse args if not provided
    if args_cli is None:
        args_cli = parse_args()

    # Terrain envs Check
    os.environ["TERRAIN"] = args_cli.terrain.lower()

    """Play with RSL-RL agent."""
    # parse configuration

    # Launch Simulation
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    """Rest everything follows."""
    import gymnasium as gym
    import torch
    import carb


    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlVecEnvWrapper
    )
    import isaaclab.sim as sim_utils
    import omni.appwindow
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv  # type: ignore

    # Related with ROS2
    # import rclpy
    # from ros2 import RobotBaseNode, add_camera, add_rtx_lidar, pub_robo_data_ros2
    # from geometry_msgs.msg import Twist


    from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
    from custom_rl_env import UnitreeGo2CustomEnvCfg, G1RoughEnvCfg

    # from omnigraph import create_front_cam_omnigraph

    # get rid of element related with ros2
    # ext_manager = omni.kit.app.get_app().get_extension_manager()
    # ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

    # FOR VR SUPPORT
    # ext_manager.set_extension_enabled_immediate("omni.kit.xr.core", True)
    # ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.steamvr", True)
    # ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.simulatedxr", True)
    # ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.openxr", True)
    # ext_manager.set_extension_enabled_immediate("omni.kit.xr.telemetry", True)
    # ext_manager.set_extension_enabled_immediate("omni.kit.xr.profile.vr", True)

    """After Importing Related Library"""
    # acquire input interface (Subscribe Keyboards for teleoperation)
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    # Select Environment Configuration
    if args_cli.robot == "go2":
        env_cfg = UnitreeGo2CustomEnvCfg()
        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg
    elif args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()
        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_g1_agent_cfg
    else:
        print(f"[ERROR] Unsupported robot type: {args_cli.robot}")

    # if args_cli.robot == "g1":
    #     env_cfg = G1RoughEnvCfg()

    # add N robots to env 
    num_envs = getattr(args_cli, "robot_amount", None) or args_cli.num_envs
    env_cfg.scene.num_envs = num_envs

    # create ros2 camera stream omnigraph
    # for i in range(env_cfg.scene.num_envs):
    #     create_front_cam_omnigraph(i)
    
    # Setting the Initial Command Buffer
    specify_cmd_for_robots(env_cfg.scene.num_envs)

    # agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg

    # if args_cli.robot == "g1":
    #     agent_cfg: RslRlOnPolicyRunnerCfg = unitree_g1_agent_cfg

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # # Update to fit isaaclab 2.2
    # env = ManagerBasedRLEnv(env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()

    # initialize ROS2 node
    # rclpy.init()
    # base_node = RobotBaseNode(env_cfg.scene.num_envs)
    # add_cmd_sub(env_cfg.scene.num_envs)

    # annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, False)
    # add_camera(env_cfg.scene.num_envs, args_cli.robot)
    if args_cli.terrain == "flat":
        setup_custom_env(args_cli)

    start_time = time.time()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # pub_robo_data_ros2(args_cli.robot, env_cfg.scene.num_envs, base_node, env, annotator_lst, start_time)
    env.close()

