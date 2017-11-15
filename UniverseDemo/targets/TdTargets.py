from torch.autograd import Variable
import torch


class TD(object):
    def __init__(self):
        pass

    @staticmethod
    def td_target_one_step_look_head(target_net, next_obs, rewards, not_done_mask, gamma):
        """
        calculate the dqn's td target. "one step look ahead"
        :param target_net: target net,
        :param next_obs: FloatTensor, [bs, channel, height, width]
        :param rewards: FloatTensor, [bs]
        :param not_done_mask: FloatTensor, [bs]
        :param gamma: float value, (0,1.)
        :return: Variable
        """
        rewards = rewards.cuda()
        not_done_mask = not_done_mask.cuda()

        next_obs = Variable(next_obs.cuda(), volatile=True)
        q_values = target_net(next_obs)

        # values [bs]
        values, _ = torch.max(q_values.data, dim=1)
        # td_target = r+gamma*max Q(s', a')
        values.mul_(gamma).mul_(not_done_mask).add_(rewards)
        # Variable(values) [bs]
        return Variable(values)

    @staticmethod
    def td_target_double_dqn_one_step_look_ahead(target_net, behavior_net, next_obs, rewards, not_done_mask, gamma):
        """
        calculate the double dqn's td target. "one step look ahead"
        :param target_net: target net,
        :param behavior_net: behavior net
        :param next_obs: FloatTensor, [bs, channel, height, width]
        :param rewards: FloatTensor, [bs]
        :param not_done_mask: FloatTensor, [bs]
        :param gamma: float value, (0,1.)
        :return: Variable [bs]
        """
        rewards = rewards.cuda()
        not_done_mask = not_done_mask.cuda()

        next_obs = Variable(next_obs.cuda(), volatile=True)
        behavior_q_values = behavior_net(next_obs)

        _, actions = torch.max(behavior_q_values.data, dim=1)
        target_q_values = target_net(next_obs)
        values = torch.gather(target_q_values, dim=1, index=actions.view(-1, 1))
        values = values.view(-1)
        values.mul_(gamma).mul_(not_done_mask).add_(rewards)
        # Variable(values), [bs]
        return Variable(values)

    @staticmethod
    def td_target_n_step_look_ahead(n, target_net, next_n_obs, rewards, not_done_mask, gamma):
        """
        calculate n step look ahead's target value
        :param n: int value, n
        :param target_net: target_net
        :param next_n_obs: FloatTensor, [bs, channel, height, width], the n-th obs from now on.
        :param rewards: FloatTensor, [bs, n], if encounter the terminal state, the reward is 0
        :param not_done_mask: FloatTensor, [bs], indicate if the n-th obs is Terminal state.
        :param gamma: float value (0., 1.)
        :return: Variable
        """
        assert len(next_n_obs) == len(rewards) == len(not_done_mask)
        rewards = rewards.cuda()
        not_done_mask = not_done_mask.cuda()

        next_n_obs = Variable(next_n_obs.cuda(), volatile=True)
        n_th_q_values = target_net(next_n_obs)
        values, _ = torch.max(n_th_q_values, dim=1)

        values.mul_(gamma ** n).mul_(not_done_mask)
        gammas = [gamma ** i for i in range(rewards.size()[1])]
        gammas = torch.unsqueeze(torch.FloatTensor(gammas).cuda(), dim=0)

        # sum (r_t+gamma*r_t1 + gamma^2*r_t2+...)
        rewards = torch.sum(gammas * rewards, dim=1)

        res = rewards + values
        return Variable(res)

    @staticmethod
    def td_target_double_dqn_n_step_look_ahead(n, target_net, behavior_net, next_n_obs, rewards, not_done_mask, gamma):
        """
        :param n: int value, n
        :param target_net: target_net
        :param behavior_net: behavior net
        :param next_n_obs: FloatTensor, [bs, channel, height, width], the n-th obs from now on.
        :param rewards: FloatTensor, [bs, n], if encounter the terminal state, the reward is 0
        :param not_done_mask: FloatTensor, [bs], indicate if the n-th obs is Terminal state.
        :param gamma: float value (0., 1.)
        :return: Variable, [bs]
        """
        """
        n step look ahead, what should we store in the replay buffer
        states
        [state0, state1, state2, ..., stateT-1], [T]
        [action0, action1, ..., actionT-1], [T]
        next states
        [state1, state2, ..., rewardT], [T], the terminal state's reward is 0
        [done1, done2, ..., doneT], [T]
        [reward1, reward2, ..., rewardT], [T]
        """
        assert len(next_n_obs) == len(rewards) == len(not_done_mask)
        rewards = rewards.cuda()
        not_done_mask = not_done_mask.cuda()

        next_n_obs = Variable(next_n_obs.cuda(), volatile=True)
        n_th_behavior_q_values = behavior_net(next_n_obs)
        _, actions = torch.max(n_th_behavior_q_values, dim=1)
        n_th_target_q_values = target_net(next_n_obs)
        values = torch.gather(input=n_th_target_q_values, dim=1, index=actions.view(-1, 1))

        values.mul_(gamma ** n).mul_(not_done_mask)

        gammas = [gamma ** i for i in range(rewards.size()[1])]
        gammas = torch.unsqueeze(torch.FloatTensor(gammas).cuda(), dim=0)

        # sum (r_t+gamma*r_t1 + gamma^2*r_t2+...)
        rewards = torch.sum(gammas * rewards, dim=1)

        res = rewards + values
        return Variable(res)

    @staticmethod
    def td_target_gae():
        raise NotImplementedError

    @staticmethod
    def shape_check():
        pass

    @staticmethod
    def td_loss(value, target, actions, loss_func):
        """
        compute the td loss
        :param value:  Variable, FloatTensor, [bs, num_actions]
        :param target: Variable, FloatTensor, [bs]
        :param actions: Variable, LongTensor, [bs]
        :return:
        """
        action_value = torch.gather(input=value, dim=1, index=actions.view(-1, 1))
        # action_value [bs]
        actioin_value = action_value.view(-1)
        loss = loss_func(actioin_value, target)
        return loss

