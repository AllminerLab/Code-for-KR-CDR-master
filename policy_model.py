
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
# from utils import *

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


# 演员评论者算法
class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.l1 = nn.Linear(state_dim, hidden_size)  # x= sW1 state_dim =400
        # self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  #  xW2

        self.actor = nn.Linear(hidden_size, act_dim)  # act_dim =200  xWp
        self.critic = nn.Linear(hidden_size, 1)

        self.saved_actions = []

    def forward(self, input):
        state, act_space = input
        x = self.l1(state)
        x = F.dropout(F.relu(x), p=0.5)

        actor_logits = self.actor(x).unsqueeze(1)
        actor_logits_s = torch.matmul(act_space, actor_logits).squeeze(1)
        act_probs = F.softmax(actor_logits_s, dim=0)

        state_value = self.critic(x)
        return act_probs, state_value

    def select_action(self, batch_state, batch_act_space, device):
        acts = []
        saved_actions = []
        for i in range(0, len(batch_state)):
            state = batch_state[i]
            act_space = batch_act_space[i]
            state = torch.FloatTensor(state).to(device)
            act_space = torch.FloatTensor(np.array(act_space)).to(device)
            act_probs, state_value = self((state, act_space))
            m = Categorical(act_probs)
            # 根据动作概率抽样动作
            act = m.sample()
            self.saved_actions.append(SavedAction(m.log_prob(act), state_value))
            acts.append(act.item())

        return acts

    # 更新策略网络，返回损失函数
    def update(self, batch_reward, batch_next_state, optimizer, device):
        if len(batch_reward) <= 0:
            del self.saved_actions[:]
            return 0.0, 0.0, 0.0

        num_path = len(batch_next_state)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
        batch_reward = torch.FloatTensor(batch_reward).to(device)
        # print('num_path: {}'.format(num_path))

        actor_loss = 0
        critic_loss = 0
        for i in range(num_path):
            log_prob, state_value = self.saved_actions[i]
            next_state = batch_next_state[i]
            x = self.l1(next_state)
            x = F.dropout(F.relu(x), p=0.5)
            advantage = batch_reward[i] + self.gamma * self.critic(x) - state_value
            actor_loss += -log_prob * advantage.detach()
            critic_loss += advantage.pow(2)

        actor_loss = torch.div(actor_loss, float(num_path))
        critic_loss = torch.div(critic_loss, float(num_path))
        loss = actor_loss + critic_loss
        # 常规术语
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del self.saved_actions[:]

        return loss.item(), actor_loss.item(), critic_loss.item()

