"""
More runners for continuous RL algorithms can be added here.
"""
import DDPG_runner
import optparse
# Modify these constants if needed.
EPISODE_LIMIT = 4000
STEPS_PER_EPISODE = 1200 # How many steps to run each episode (changing this messes up the solved condition)

if __name__ == '__main__':
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--train", default="not", help="If you want to train the Reinforcement Learning model set to 'yes'")
    opt_parser.add_option("--yolo", default="not", help="If you want to detect human with yolo set to 'yes'")
    options, args = opt_parser.parse_args()
    DDPG_runner.run(options.train,options.yolo)


