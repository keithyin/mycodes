import gym
import universe  # register the universe environments
import matplotlib.pyplot as plt

env = gym.make('flashgames.DuskDrive-v0')
# [x,y] [20, 85] [815, 595]
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

# there's n environment and [[('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowUp', False)], [(...),[...]]]

while True:
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here

    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()
    if observation_n[0] is None:
        continue
    # print('observation_n', observation_n, 'reward', reward_n)
    # print(type(observation_n[0]))
    # print(observation_n[0])
    # plt.imsave('dusk.jpg',observation_n[0]['vision'])
    # exit()
