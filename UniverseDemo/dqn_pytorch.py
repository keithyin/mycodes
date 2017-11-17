import itertools
import gym.spaces
from torch.autograd import Variable
from utils.dqn_utils import *
from utils import common_algorithm
from DqnDushDrive.actions import ACTIONS
import torch
from torchvision import transforms
from targets.TdTargets import TD

cuda_available = torch.cuda.is_available()


def learn(env,
          q_func,
          exploration=LinearSchedule(10000000, 0.1),
          replay_buffer_size=50000,
          batch_size=128,
          gamma=0.99,
          learning_starts=1000,
          learning_freq=4,
          frame_history_len=4):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    """

    ###############
    # BUILD MODEL #
    ###############
    num_actions = 3
    Q_net = q_func(num_actions=num_actions)
    target_net = q_func(num_actions=num_actions)
    load_param = True
    if load_param:
        Q_net.load_state_dict(torch.load("/home/fanyang/WorkSpace/mycodes/UniverseDemo/param.pkl"))
        target_net.load_state_dict(torch.load("/home/fanyang/WorkSpace/mycodes/UniverseDemo/param.pkl"))

    if cuda_available:
        Q_net.cuda()
        target_net.cuda()

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    last_obs = env.reset()
    reward_track = []
    episode_num = 0

    for t in itertools.count():
        # ## 1. Check stopping criterion
        # #### behavior policy epsilon-greedy ##############
        # get the current state                            #
        # choose an action from the epsilon-greedy         #
        # take an action                                   #
        # get the reward and observe the next observation  #
        ####################################################
        # env.render()
        if last_obs[0] is None:
            last_obs, _, _, _ = env.step([ACTIONS[0]])
            continue
        # res_img = img[85:595, 20:815, :]
        idx = replay_buffer.store_frame(last_obs[0]['vision'][85:595:4, 20:815:4, :])

        # cur_state [12, 128, 199]
        cur_state = replay_buffer.encode_recent_observation() / 255.0
        cur_state = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])(torch.FloatTensor(cur_state))
        if cuda_available:
            cur_state = Variable(torch.unsqueeze(cur_state, dim=0).cuda(), volatile=True)
        else:
            cur_state = Variable(torch.unsqueeze(cur_state, dim=0), volatile=True)

        logits = Q_net(cur_state)

        action_idx = common_algorithm.epsilon_greedy(logits=logits, num_actions=num_actions,
                                                     epsilon=exploration.value(t))
        action = [ACTIONS[action_idx]]
        for i in range(2):
            observations, rewards, dones, infos = env.step(action)
            reward_track.append(rewards[0] / 1000)
            env.render()
            if dones[0]:
                break
        replay_buffer.store_effect(idx=idx, action=action_idx, reward=rewards[0], done=dones[0])

        # plt.imsave('img%d.png' % t, observations[0]['vision'][85:595:4, 20:815:4, :])

        if dones[0]:
            reward_track = np.array(reward_track)
            score = np.sum(reward_track)
            print('*************************************************')
            print("episode %d , score : %.1f" % (episode_num, score))
            print('*************************************************')
            episode_num += 1
            reward_track = []
            observations = env.reset()

        last_obs = observations

        # ####### experience replay ################
        # sample mini-batch from replay buffer     #
        # calculate the target                     #
        # train the online net                     #
        # parameter synchronization every n step   #
        ############################################
        if (t > learning_starts and
                replay_buffer.can_sample(batch_size)):

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = \
                replay_buffer.sample(batch_size=batch_size)
            # ndarray -> torch.autograd.Variable
            obs_batch = Variable(torch.FloatTensor(obs_batch / 255.0))
            act_batch = Variable(torch.LongTensor(act_batch.astype(np.int64)))
            rew_batch = torch.FloatTensor(rew_batch)
            next_obs_batch = torch.FloatTensor(next_obs_batch / 255.0)
            not_done_mask = torch.FloatTensor(1 - done_mask)

            if cuda_available:
                obs_batch = obs_batch.cuda()
                next_obs_batch = next_obs_batch.cuda()
                not_done_mask = not_done_mask.cuda()
                rew_batch = rew_batch.cuda()
                act_batch = act_batch.cuda()
            # calculate the target,
            td_target = TD.td_target_one_step_look_head(target_net=target_net, next_obs=next_obs_batch,
                                                               rewards=rew_batch, not_done_mask=not_done_mask,
                                                               gamma=gamma)

            Q_value = Q_net(obs_batch)

            selected_value = torch.gather(Q_value, dim=1, index=act_batch.view(-1, 1))
            loss = torch.mean(torch.pow(td_target - selected_value, 2))
            Q_net.get_optimizer().zero_grad()
            loss.backward()
            Q_net.get_optimizer().step()

            # update the target network
            if t % learning_freq == 0:
                target_net.load_state_dict(Q_net.state_dict())
                torch.save(target_net.state_dict(), 'param.pkl')


def main():
    pass


if __name__ == '__main__':
    main()
