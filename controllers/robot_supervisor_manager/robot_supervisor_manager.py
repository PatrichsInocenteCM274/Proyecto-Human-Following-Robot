"""
More runners for continuous RL algorithms can be added here.
"""
import DDPG_runner
import optparse
# Modify these constants if needed.
EPISODE_LIMIT = 3000
STEPS_PER_EPISODE = 800 # How many steps to run each episode (changing this messes up the solved condition)

if __name__ == '__main__':
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--train", default="not", help="If you want to train the Reinforcement Learning model set to 'yes'")
    opt_parser.add_option("--yolo", default="not", help="If you want to detect human with yolo set to 'yes'")
    options, args = opt_parser.parse_args()
    DDPG_runner.run(options.train,options.yolo)

# from robot_supervisor import CartPoleRobotSupervisor
# from stable_baselines3.ppo import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import CheckpointCallback
#
#
# env = Monitor(CartPoleRobotSupervisor(), filename="")
# checkpoint_callback = CheckpointCallback(
#     save_freq=int(1e4),
#     save_path='./checkpoints/',
#     name_prefix='rl_model'
# )
#
# # Train
# model = PPO(
#     'MlpPolicy',
#     env,
#     tensorboard_log='./tb_logs/',
#     verbose=1
# )
# model.learn(total_timesteps=int(1e6), callback=checkpoint_callback)
