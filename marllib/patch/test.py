import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="BoxLocking",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--scenario_name",
    type=str,
    default="empty",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--num_agents",
    type=int,
    default=2,
    help="Number of iterations to train.")
parser.add_argument(
    "--num_boxes",
    type=int,
    default=2,
    help="Number of timesteps to train.")
parser.add_argument(
    "--floor_size",
    type=float,
    default=12.0,
    help="Reward at which we stop training.")
parser.add_argument(
    "--grid_size",
    type=int,
    default=60,
    help="Number of timesteps to train.")
parser.add_argument(
    "--env_horizon",
    type=int,
    default=60,
    help="Number of timesteps to train.")
parser.add_argument(
    "--task_type",
    type=str,
    default="all-return",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--algo",
    type=str,
    default="debug",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Number of timesteps to train.")
parser.add_argument(
    "--fixed_door",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")
parser.add_argument(
    "--spawn_obs",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")

# if __name__ == '__main__':
#     args = parser.parse_args()
#     env = BoxLockingEnv(args)
#     a = env.reset()
#     print(1)

import mate

# Base environment for MultiAgentTracking
env = mate.make('MATE-4v2-9-v0')
env.seed(0)
done = False
camera_joint_observation, target_joint_observation = env.reset()
while not done:
    env.render()
    camera_joint_action, target_joint_action = env.action_space.sample()# your agent here (this takes random actions)
    (
        (camera_joint_observation, target_joint_observation),
        (camera_team_reward, target_team_reward),
        done,
        (camera_infos, target_infos)
    ) = env.step((camera_joint_action, target_joint_action))
