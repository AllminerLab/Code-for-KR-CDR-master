# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import pickle
import sys
import os
from tqdm import tqdm

from decision_env import BatchDecisionEnvironments
from knowledge_graph.data_utils import *
from utils import *
import torch
from policy_model import *

# 定义全局变量，日志
logger = None


def get_uids(data_path):
    data_dict = {}

    if data_path == TEST_DATA_PATH:
        data_dict = load_test_data(data_path)

    if data_path == ATTENTION_NETWORK_TRAIN_PATH:
        data_dict = load_attention_network_train_data(data_path)

    uids = data_dict.keys()

    return list(uids)


def batch_beam_search(decision_env, model, uids, device, max_path_len, topk=[25, 5, 1]):
    def get_not_done_num(batch_done):
        num = 0
        for done in batch_done:
            if not done:
                num += 1
        return num

    state_pool, done_pool = decision_env.reset(uids)
    path_pool = decision_env._batch_path
    prob_pool = [[0.0] for _ in uids]
    reward_pool = decision_env._batch_get_reward(path_pool, done_pool)
    reward_pool = [[reward] for reward in reward_pool]
    # 不更新模型参数
    model.eval()
    for hop in range(max_path_len):
        # 如果没有未终止路径，结束搜索
        if get_not_done_num(done_pool) <= 0:
            break
        state_tensor = torch.FloatTensor(np.array(state_pool)).to(device)
        acts_pool, actmask_pool, _ = decision_env._batch_get_action(path_pool, done_pool)
        # 扩展池
        new_path_pool = []
        new_done_pool = []
        new_prob_pool = []
        new_reward_pool = []

        j = 0
        for i in range(len(path_pool)):
            # 如果路径探索结束，依然加入路径池
            if done_pool[i]:
                new_done_pool.append(done_pool[i])
                new_path_pool.append(path_pool[i])
                new_prob_pool.append(prob_pool[i])
                new_reward_pool.append(reward_pool[i])

            if not done_pool[i]:
                # 获取对应的路径
                path = path_pool[i]
                prob = prob_pool[i]
                reward = reward_pool[i]
                state = state_tensor[j].to(device)
                actmask = actmask_pool[j]
                acts = acts_pool[j]

                j += 1
                actmask = torch.FloatTensor(actmask).to(device)
                model.eval()
                with torch.no_grad():
                    probs, _ = model((state, actmask))

                if hop < len(topk):
                    if len(probs) < topk[hop]:
                        topk_probs, topk_idxs = torch.topk(probs, len(probs))
                    else:
                        topk_probs, topk_idxs = torch.topk(probs, topk[hop])
                else:
                    topk_probs, topk_idxs = torch.topk(probs, 1)
                topk_probs = topk_probs.detach().cpu().numpy()
                topk_idxs = topk_idxs.detach().cpu().numpy()

                for idx, p in zip(topk_idxs, topk_probs):
                    relation, next_node_id = acts[idx]
                    new_path = path + [(relation, next_node_id)]
                    new_path_pool.append(new_path)
                    # 获取新路径的终止状态
                    new_done = decision_env._is_done(new_path, False)
                    new_done_pool.append(new_done)
                    # 获取该状态的奖励
                    new_reward = decision_env._get_reward(new_path, new_done)
                    new_reward_pool.append(reward + [new_reward])
                    # 更新路径概率
                    new_prob_pool.append(prob + [p])

        path_pool = new_path_pool
        done_pool = new_done_pool
        prob_pool = new_prob_pool
        reward_pool = new_reward_pool
        # done_pool = decision_env._batch_is_done(path_pool, done_pool)
        state_pool = decision_env._batch_get_state(path_pool, done_pool)

    return path_pool, prob_pool, reward_pool


def path_inference(args, policy_file, path_file, data_path):
    decision_env = BatchDecisionEnvironments(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                                             state_history=args.state_history, reward_weights=args.reward_weights)
    model = ActorCritic(decision_env.state_dim, decision_env.act_dim, gamma=args.gamma,
                        hidden_size=args.hidden).to(args.device)

    if torch.cuda.is_available():
        pretrain_sd = torch.load(policy_file)
    else:
        pretrain_sd = torch.load(policy_file, map_location='cpu')
    # 返回一个参数状态字典
    model_sd = model.state_dict()
    # update() 方法可使用一个字典所包含的键值对来更新己有的字典。
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    # 获取测试集
    test_uids = get_uids(data_path)
    get_uids(data_path)
    batch_size = 64
    start_idx = 0
    all_paths, all_probs, all_rewards = [], [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs, rewards = batch_beam_search(decision_env, model, batch_uids, args.device, args.max_path_len,
                                                  topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        all_rewards.extend(rewards)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs, 'rewards': all_rewards}
    pickle.dump(predicts, open(path_file, 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=EXAMPLE, help='')
    parser.add_argument('--policy_name', type=str, default='train_agent', help='directory name')
    parser.add_argument('--name', type=str, default='path_reasoning', help='directory name')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--epochs', type=int, default=20, help='num of epochs in train_agent')
    parser.add_argument('--max_acts', type=int, default=MAX_ACTS, help='Max number of actions')
    parser.add_argument('--max_path_len', type=int, default=5, help='Max path length')
    parser.add_argument('--train_max_path_len', type=int, default=6, help='')
    parser.add_argument('--train_batch_size', type=int, default=512, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=256, help='number of samples')
    parser.add_argument('--reward_weights', type=int, nargs='*', default=REWARD_WEIGHTS, help='weights of reward')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--is_train_task', type=bool, default=True, help='train or test')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.policy_name
    path_log_dir = TMP_DIR[args.dataset] + '/' + args.name
    policy_file = args.log_dir + '/policy_model_pathLen_{}_batchSize_{}_maxActs_{}_rewardW_{}_epoch_{}.ckpt'.format(args.train_max_path_len,
                                                                                              args.train_batch_size,
                                                                                              MAX_ACTS,
                                                                                              REWARD_WEIGHTS,
                                                                                              args.epochs)

    if not os.path.isdir(path_log_dir):
        os.makedirs(path_log_dir)

    if args.is_train_task:
        # Modification 2024/04/30
        path_file = path_log_dir + '/train_policy_paths_pathLen_{}_topK_{}_acts_{}_rewar_{}_epoch_{}.pkl'.format(args.max_path_len, args.topk,
                                                                                                                            MAX_ACTS, REWARD_WEIGHTS,args.epochs)
        data_path = ATTENTION_NETWORK_TRAIN_PATH
        logger = get_logger(path_log_dir + '/train_path_reasoning_pathLen_{}_topK_{}_acts_{}_rewar_{}_epoch_{}_log.txt'.format(args.max_path_len, args.topk,
                                                                                                                                         MAX_ACTS, REWARD_WEIGHTS,args.epochs))


    else:
        path_file = path_log_dir + '/test_policy_paths_pathLen_{}_topK_{}_acts_{}_rewar_{}_epoch_{}.pkl'.format(args.max_path_len,
                                                                                               args.topk,
                                                                                               MAX_ACTS, REWARD_WEIGHTS, args.epochs)
        data_path = TEST_DATA_PATH
        logger = get_logger(
            path_log_dir + '/test_path_reasoning_pathLen_{}_topK_{}_acts_{}_rewar_{}_epoch_{}_log.txt'.format(args.max_path_len,
                                                                                             args.topk,
                                                                                             MAX_ACTS, REWARD_WEIGHTS, args.epochs))

    logger.info(args)

    path_inference(args, policy_file, path_file, data_path)


if __name__ == '__main__':
    main()
