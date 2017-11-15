from dqn_pytorch import learn
import gym
import universe
from nets.qnet_pytorch import QNetwork


def main():
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=1)
    learn(env=env, q_func=QNetwork)


if __name__ == '__main__':
    main()
