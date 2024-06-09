
from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

# from knowledge_graph import KnowledgeGraph
# from kg_env import BatchKGEnvironment
import datetime
from utils import *
from decision_env import *
from policy_model import *

# 日志，全局变量
logger = None


# 批量加载用户ID
class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None

        #每一个batch返回多个用户
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]  # 不包括end_idx
        batch_uids = self.uids[batch_idx]  #
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()


def is_all_done(self, batch_done):
    for done in batch_done:
        if not done:
            return False
    return True


def train_actorCritic(args):
    decision_env = BatchDecisionEnvironments(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                                             state_history=args.state_history, reward_weights=args.reward_weights)
    # 获取源域所有用户的ID
    uids = decision_env.kg_utils.get_users(SOURCE)
    dataloader = ACDataLoader(uids, args.batch_size)
    model = ActorCritic(decision_env.state_dim, decision_env.act_dim,  gamma=args.gamma,
                        hidden_size=args.hidden).to(args.device)

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)

    # 保存参数，model.parameters()保存的是Weights和Bais参数的值
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 总损失， 总策略网络损失，总状态价值网络损失
    total_losses, total_plosses, total_vlosses = [], [], []
    # 每100次更新参数便输出一次损失函数
    step = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        bar = tqdm(total=len(uids), desc='Epoch {}:'.format(epoch))
        # 初始化用户数据
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            batch_curr_state, batch_done = decision_env.reset(batch_uids)
            # Start batch episodes ###
            path_len = 0
            while decision_env.get_not_done_num() > 0:
                # 更新当前状态集合
                batch_act_space_embed = decision_env.batch_action_mask(args.act_dropout)
                batch_curr_state = decision_env.get_batch_state()
                batch_act_idx = model.select_action(batch_curr_state, batch_act_space_embed, args.device)
                batch_next_state, batch_reward = decision_env.batch_step(batch_act_idx)

                # 更新策略网络和状态价值网络参数
                loss, ploss, vloss = model.update(batch_reward, batch_next_state, optimizer, args.device)
                total_losses.append(loss)
                total_plosses.append(ploss)
                total_vlosses.append(vloss)
                step += 1

                if step > 0 and step % 5 == 0:
                    avg_loss = np.mean(total_losses)
                    avg_ploss = np.mean(total_plosses)
                    avg_vloss = np.mean(total_vlosses)
                    total_losses, total_plosses, total_vlosses = [], [], []
                    logger.info(
                        'epoch/step={:d}/{:d}'.format(epoch, step) +
                        ' | loss={:.5f}'.format(avg_loss) +
                        ' | ploss={:.5f}'.format(avg_ploss) +
                        ' | vloss={:.5f}'.format(avg_vloss))
                path_len += 1

            bar.update(args.batch_size)

        # 保留模型参数
        if epoch % 1 == 0:
            policy_file = '{}/policy_model_pathLen_{}_batchSize_{}_maxActs_{}_rewardW_{}_epoch_{}.ckpt'.format(
                                                                                                        args.log_dir,
                                                                                                        args.max_path_len,
                                                                                                        args.batch_size,
                                                                                                        args.max_acts,
                                                                                                        args.reward_weights,
                                                                                                        epoch)
            logger.info("Save model to " + policy_file)
            # model.state_dict()获取模型全部参数
            torch.save(model.state_dict(), policy_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default =EXAMPLE, help='for example, moive-music')
    parser.add_argument('--name', type =str, default='train_agent', help='directory name')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_acts', type=int, default=MAX_ACTS, help='Max number of actions, default: 250')
    parser.add_argument('--max_path_len', type=int, default=6, help='Max path length')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor')
    parser.add_argument('--act_dropout', type=float, default=0.0, help='action dropout rate')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, default=256, help='number of samples')
    parser.add_argument('--reward_weights', type=int, nargs='*', default=REWARD_WEIGHTS, help='weights of reward,  default: [0.7, 0.2, 0.1]')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    # 训练日志/tmp/dataset/train_agent
    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)  # format函数用于格式化字符串。可以接受无限个参数，可以指定顺序。返回结果为字符串。
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    c_time = datetime.now().timestamp()
    global logger
    # 打开训练日志 /tmp/dataset/train_agent/train_agent.txt
    logger = get_logger(args.log_dir + '/train_agent_time_{}_log.txt'.format(c_time))
    # 保存超参
    logger.info(args)

    set_random_seed(args.seed)
    # 进行ActorCritic训练
    train_actorCritic(args)


if __name__ == '__main__':
    main()