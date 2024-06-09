from __future__ import absolute_import, division, print_function

import copy
import os
import sys
import numpy as np
from numpy import exp
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime

from knowledge_graph.kg_utils import *
from utils import *
from test_meta import *

class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len
        if history_len == 0:
            self.dim = embed_size * 2
        elif history_len == 1:
            self.dim = embed_size * 4
        elif history_len == 2:
            self.dim = embed_size * 6
        else:
            raise Exception('history length should be one of {0, 1, 2}')

        # 该方法的功能类似于在类中重载()
        # 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()"的形式使用。

    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                 older_relation_embed):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                                   older_relation_embed])
        else:
            raise Exception('mode should be one of {full, current}')


class BatchDecisionEnvironments(object):
    # max_path_len为最大跳数T，暂时定义为10
    def __init__(self, dataset_str, max_acts, max_path_len=10, state_history=1, reward_weights=[0.7, 0.2, 0.1]):
        # 加载知识图谱工具
        self.kg_utils = KGUtils(GRAPH_PATH, Entities_EMBED_PATH, RELATIONS_EMBED_PATH)
        self.max_acts = max_acts
        self.max_num_nodes = max_path_len + 1  # max number of hops (= #nodes - 1)
        # self.kg = object  # 加载知识图谱   load_kg(dataset_str)
        # self.embeds = {}  # 加载训练好的实体和关系的嵌入   load_embed(dataset_str)
        self.reward_weights = reward_weights
        self.path_num = 0
        # 计算实体维度
        rea_info = self.kg_utils.get_relation_info(PURCHASE)
        self.embed_size = len(rea_info.get(EMBED))
        # 动作的维度
        self.act_dim = self.embed_size * 2
        # self.embed_size # 实体和关系的嵌入大小，即维度 self.embeds[USER].shape[1]
        self.state_gen = KGState(self.embed_size, state_history)
        self.state_dim = self.state_gen.dim
        # 调用元网络
        self.terminalReward = TerminalReward()
        # 所有共性词的嵌入集合
        self.all_words_emb = []
        entity_info_list = self.kg_utils.get_entities_by_type(WORD)
        for entity_info in entity_info_list:
            if entity_info.get(DOMAIN_NAME) == SAME:
                self.all_words_emb.append(entity_info.get(EMBED))

        # 全局变量
        self._batch_done = None
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_actions_embed = None
        self._batch_curr_state = None
        self._batch_curr_reward = None

    def get_entity_type(self, node_id):
        return self.kg_utils.get_entity_info(node_id).get(TYPE)

    def get_entity_embed(self, node_id):
        return self.kg_utils.get_entity_info(node_id).get(EMBED)

    def get_entity_domain(self, node_id):
        return self.kg_utils.get_entity_info(node_id).get(DOMAIN_NAME)

    def get_relation_embed(self, relation_type):
        if relation_type is None:
            return np.zeros(self.embed_size)
        return self.kg_utils.get_relation_info(relation_type).get(EMBED)

    def is_target_entity(self, node_id):
        node_info = self.kg_utils.get_entity_info(node_id)
        node_type = node_info.get(TYPE)
        if self.kg_utils.check_entity_domain(node_id, TARGET):
            if node_type == USER or node_type == PRODUCT:
                return True
        # overlap users
        # elif domain_name == SAME and node_type == USER:
        #     return True

        return False

    def get_neighbors(self, node_id):
        neigh_list = self.kg_utils.get_entity_neighbors(node_id)
        relations_nodes = []
        for relation_entity in neigh_list:
            relation = relation_entity.get(RELATION).get(TYPE)
            node_id = relation_entity.get(ENTITY).get(ID)
            relations_nodes.append((relation, node_id))
        return relations_nodes

    def get_inter_items_embed(self, node_id):
        neigh_list = self.kg_utils.get_entity_neighbors(node_id)
        item_embed_list = []
        for relation_entity in neigh_list:
            entity_info = relation_entity.get(ENTITY)
            if entity_info.get(TYPE) == PRODUCT:
                item_embed_list.append(entity_info.get(EMBED))
        return item_embed_list

    # 获取集合的共性实体
    def get_all_onetype_entities_emb(self, node_type):
        entity_info_list = self.kg_utils.get_entities_by_type(node_type)
        entity_embed_list = []
        for entity_info in entity_info_list:
            if entity_info.get(DOMAIN_NAME) == SAME:
                entity_embed_list.append(entity_info.get(EMBED))
        return entity_embed_list

    def get_source_domain_users(self):
        total_users = []
        source_users = self.kg_utils.get_users(SOURCE)
        total_users.extend(source_users)
        same_users = self.kg_utils.get_users(SAME)
        total_users.extend(same_users)
        return total_users

    def _get_actions_embed(self, actions):
        actions_embed = []
        # 生成由动作嵌入组成的动作空间
        for r, next_node_id in actions:
            # 下一个节点的嵌入
            next_node_embed = self.get_entity_embed(next_node_id)
            # 关系嵌入
            r_embed = self.get_relation_embed(r)
            actions_embed.append(np.concatenate([r_embed, next_node_embed]))
        return actions_embed

    def _get_actions(self, path, done):
        _, curr_node_id = path[-1]
        actions = []
        actions_embed = []

        # 如果探索结束，返回None
        if done:
            return actions, actions_embed, done

        # 获取当前节点可能的边和节点
        # relations_nodes: [(relation1,next_node_id1),(relation2,next_node_id2),...]
        relations_nodes = self.get_neighbors(curr_node_id)
        # candidate_acts: list of tuples of (relation, node_id)
        candidate_acts = []
        # path: list of tuples of (relation, node_type, node_id)
        visited_nodes = set([(v[1]) for v in path])
        for r, node_id in relations_nodes:
            # 根据当前实体curr_node_type和关系类型r，寻找所连接的实体类型
            if node_id not in visited_nodes:
                candidate_acts.append((r, node_id))
            # extend() 该方法没有返回值，但会在已存在的列表中添加新的列表内容

        # If candidate action set is empty
        if len(candidate_acts) <= 0:
            done = True
            return actions, actions_embed, done

        # 小于最大空间范围
        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions, self._get_actions_embed(actions), done

        # 超出最大动作空间范围，进行剪枝操作,通过计算user embedding与 r embedding + next_node embedding的分数排序
        user_embed = self.get_entity_embed(path[0][-1])
        scores = []
        for r, next_node_id in candidate_acts:
            # 下一个节点的嵌入
            next_node_embed = self.get_entity_embed(next_node_id)
            # 关系嵌入
            r_embed = self.get_relation_embed(r)
            score = np.dot(user_embed, next_node_embed + r_embed)

            scores.append(score)

        candidate_idxs = np.argsort(scores)[-self.max_acts:]
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions, self._get_actions_embed(actions), done

    # 对尚未终止的路径，获取下一步的动作空间，同时，对动作空间为空的路径，修改为终止状态
    def _batch_get_action(self, batch_path, batch_done):
        # print("_batch_get_action---done")
        # print(batch_done)
        batch_actions = []
        batch_actions_embed = []
        for i in range(0, len(batch_path)):
            if not batch_done[i]:
                path = batch_path[i]
                done = batch_done[i]
                actions, actions_embed, batch_done[i] = self._get_actions(path, done)
                # 如果动作空间不为空，排除掉动作空间为空
                # if not batch_done[i]:
                batch_actions.append(actions)
                batch_actions_embed.append(actions_embed)

        return batch_actions, batch_actions_embed, batch_done

    def _get_state(self, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        user_embed = self.get_entity_embed(path[0][-1])
        zero_embed = np.zeros(self.embed_size)
        if len(path) == 1:
            state = self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state

        older_relation, last_node_id = path[-2]
        last_relation, curr_node_id = path[-1]
        curr_node_embed = self.get_entity_embed(curr_node_id)
        last_node_embed = self.get_entity_embed(last_node_id)
        last_relation_embed = self.get_relation_embed(last_relation)
        if len(path) == 2:
            state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed,
                                   zero_embed)
            return state

        _, older_node_id = path[-3]
        older_node_embed = self.get_entity_embed(older_node_id)
        older_relation_embed = self.get_relation_embed(older_relation)
        state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed,
                               older_relation_embed)
        return state

    # 对尚未终止的路径，获取其当前状态
    def _batch_get_state(self, batch_path, batch_done):
        batch_curr_state = []
        for i in range(len(batch_path)):
            if not batch_done[i]:
                path = batch_path[i]
                batch_curr_state.append(self._get_state(path))

        return batch_curr_state

    # 更新当前状态集
    def get_batch_state(self):
        batch_state = copy.deepcopy(self._batch_curr_state)
        return batch_state

    def _get_reward(self, path, done):
        # 初始状态不给予奖励
        if len(path) <= 1:
            return 0.0
        # 到达目标域奖励， 路径奖励，共性奖励
        reward_target = 0.0
        reward_path = 0.0
        reward_word = 0.0

        _, curr_node_id = path[-1]
        curr_node_embed = self.get_entity_embed(curr_node_id)
        _, user_node_id = path[0]
        user_embed = self.get_entity_embed(user_node_id)
        # 到达终端节点
        if done:
            # 判断终端节点是否是目标域用户或项节点
            if self.is_target_entity(curr_node_id):
                # 获取用户的交互的项嵌入集合 list
                src_entities_emb = self.get_inter_items_embed(user_node_id)
                # 加入用户本身
                src_entities_emb.append(user_embed)
                curr_node_embed_list = [curr_node_embed]
                # 计算奖励
                reward_target = self.terminalReward.get_target_reward(src_entities_emb, curr_node_embed_list)
                reward_path = self.max_num_nodes - len(path)

            else:
                # 负奖励
                reward_target = 0

        # 计算共性奖励
        # 如果是共性实体

        if self.get_entity_type(curr_node_id) == WORD and self.get_entity_domain(curr_node_id) == SAME:
            all_words_reward = 0.0
            for word_emed in self.all_words_emb:
                user_word = np.dot(user_embed, word_emed)
                all_words_reward += np.exp(user_word)
            reward_word = np.exp(np.dot(user_embed, curr_node_embed)) / all_words_reward

        final_reward = self.reward_weights[0] * reward_target + self.reward_weights[1] * reward_path + \
                       self.reward_weights[2] * reward_word

        return final_reward

    def _batch_get_reward(self, batch_path, batch_done):
        # print("reward start")
        batch_reward = []
        for i in range(len(batch_path)):
            reward = self._get_reward(batch_path[i], batch_done[i])
            # print(reward)
            batch_reward.append(reward)
        return batch_reward

    def _is_done(self, path, done):
        if done:
            return done

        # 动作空间为空
        actions, _, _ = self._get_actions(path, done)
        if len(actions) <= 0:
            return True

        # 提前到达目标域
        _, curr_node_id = path[-1]
        # 判断节点是不是目标域的实体，如果是，则返回True
        if self.is_target_entity(curr_node_id):
            return True

        # 到达最大长度
        if len(path) >= self.max_num_nodes:
            return True

        return False

    def _batch_is_done(self, batch_path, batch_done):

        for i in range(len(batch_path)):
            batch_done[i] = self._is_done(batch_path[i], batch_done[i])
        return batch_done

    def get_not_done_num(self):
        number = 0
        for done in self._batch_done:
            if not done:
                number += 1
        return number

    def get_batch_done(self):
        return self._batch_done

    def print_args(self):
        print("self._batch_done:")
        print(self._batch_done)
        print("self._batch_path:")
        print(self._batch_path)
        # print("self._batch_curr_state:")
        # print(self._batch_curr_state)
        print("self._batch_curr_actions:")
        for actions in self._batch_curr_actions:
            print("===========")
            print(len(actions))
            print("+++++++++++")

        print("self._batch_curr_reward:")
        print(self._batch_curr_reward)
        print("print end")

    def reset(self, uids=None):
        if uids is None:
            print("uids is empty!")
            return None

        # each element is a tuple of (relation, entity_type, entity_id)
        self._batch_path = [[(None, uid)] for uid in uids]
        self.path_num = len(self._batch_path)
        self._batch_done = []
        for i in range(self.path_num):
            self._batch_done.append(False)
        # self._batch_curr_state = self._batch_get_state(self._batch_path, self._batch_done)
        # 判断起点可不可用
        self._batch_done = self._batch_is_done(self._batch_path, self._batch_done)
        # time_start = time.time()
        self._batch_curr_actions, self._batch_curr_actions_embed, self._batch_done = self._batch_get_action(
            self._batch_path, self._batch_done)
        # time_end = time.time()  # 记录结束时间
        # time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        # print("get_actions_time:"+str(time_sum))
        self._batch_curr_reward = self._batch_get_reward(self._batch_path, self._batch_done)
        self._batch_curr_state = self._batch_get_state(self._batch_path, self._batch_done)

        return self._batch_curr_state, self._batch_done

    def batch_step(self, batch_act_idx):
        assert len(self._batch_curr_actions) == len(batch_act_idx)

        j = 0
        for i in range(len(self._batch_path)):
            if not self._batch_done[i]:
                path = self._batch_path[i]
                curr_actions = self._batch_curr_actions[j]
                act_idx = batch_act_idx[j]
                j = j + 1
                # 更新路径
                # _, curr_node_id = path[-1]
                relation, next_node_id = curr_actions[act_idx]
                self._batch_path[i].append((relation, next_node_id))
        # 更新状态
        self._batch_curr_state = self._batch_get_state(self._batch_path, self._batch_done)
        last_batch_state = copy.deepcopy(self._batch_curr_state)
        # 保留更新前的终止状态,保留上一跳的路径位置
        last_batch_done = copy.deepcopy(self._batch_done)
        # 判断是否到达终点
        self._batch_done = self._batch_is_done(self._batch_path, self._batch_done)

        # self.print_args()

        # 计算奖励
        total_rewards = self._batch_get_reward(self._batch_path, self._batch_done)
        # 帅选出有效奖励
        del self._batch_curr_reward[:]
        for i in range(len(last_batch_done)):
            if not last_batch_done[i]:
                self._batch_curr_reward.append(total_rewards[i])

        # 更新动作集合
        self._batch_curr_actions, self._batch_curr_actions_embed, self._batch_done = self._batch_get_action(
            self._batch_path, self._batch_done)
        # 更新有效状态集合
        self._batch_curr_state = self._batch_get_state(self._batch_path, self._batch_done)

        return last_batch_state, self._batch_curr_reward

    def action_mask(self, curr_actions, curr_actions_embed, dropout=0.0):
        act_space_embed = []
        zero_embed = np.zeros(self.act_dim)
        # np.concatenate((a, b), axis=1)
        act_idxs = list(range(len(curr_actions)))
        if dropout > 0 and len(act_idxs) >= 5:
            keep_size = int(len(act_idxs)) * (1.0 - dropout)
            tmp = np.random.choice(act_idxs, keep_size, replace=False).tolist()
            act_idxs = tmp
        act_mask = np.zeros(self.max_acts, dtype=np.uint8)
        act_mask[act_idxs] = 1
        # 将动作嵌入空间对应的动作嵌入设置为零
        for i in range(self.max_acts):
            if act_mask[i] == 1:
                act_space_embed.append(curr_actions_embed[i])
            # else:
            #     act_space_embed.append(zero_embed)

        # act_space_embed = np.vstack(act_space_embed)
        return act_space_embed

    def batch_action_mask(self, dropout=0.0):

        return self._batch_curr_actions_embed

    def print_path(self, path, done):
        msg = str(done) + '--Path: {}({})'.format(self.get_entity_type(path[0][-1]), path[0][-1])
        for node in path[1:]:
            msg += ' =={}=> {}({})'.format(node[0], self.get_entity_type(node[1]), node[1])
        actions, _, _ = self._get_actions(path, done)
        is_not_act = False
        if len(actions) <= 0:
            is_not_act = True
        is_target = self.is_target_entity(path[-1][1])
        if is_target:
            print(msg)
        return

    def print_all_path(self):
        for i in range(len(self._batch_path)):
            path = self._batch_path[i]
            done = self._batch_done[i]
            self.print_path(path, done)
