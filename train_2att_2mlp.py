import argparse
import copy
import os
from time import time

from tqdm import tqdm

import eval_utils
from decision_env import BatchDecisionEnvironments
from knowledge_graph.data_utils import load_attention_network_train_data
from knowledge_graph.kg_utils import KGUtils
# from knowledge_graph.kg_utils import *
from utils import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from scipy import spatial
from functools import reduce
import torch.nn as nn

# 定义全局变量，日志
logger = None
logger_score = None
curr_time = None
device = None

class PathAttentionNetwork(torch.nn.Module):
    def __init__(self, emb_dim, att_hidden_dim):
        super(PathAttentionNetwork, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(emb_dim, att_hidden_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(att_hidden_dim, 1))
        self.att_softmax = torch.nn.Softmax(dim=0)

    def forward(self, path_nodes):
        atts = self.net(path_nodes)
        weights = self.att_softmax(atts)
        path_fea = torch.sum(weights * path_nodes, 0)
        return path_fea


class AttentionNetwork(torch.nn.Module):
    def __init__(self, emb_dim, att_hidden_dim):
        super(AttentionNetwork, self).__init__()
        self.attnet = torch.nn.Sequential(torch.nn.Linear(emb_dim, att_hidden_dim), torch.nn.ReLU(),
                                          torch.nn.Linear(att_hidden_dim, 1))
        self.att_softmax = torch.nn.Softmax(dim=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.path_net = PathAttentionNetwork(emb_dim, att_hidden_dim)
        # 预测评分
        self.predict = torch.nn.Sequential(torch.nn.Linear(emb_dim * 2, emb_dim), torch.nn.Linear(emb_dim, 1))

        self.save_preds = []

        # 获取单个用户嵌入

    def forward(self, paths_nodes):
        paths_fea = []
        atts = []
        atts_copy = []
        path_num = len(paths_nodes)
        for i in range(path_num):
            paths_nodes_emb = torch.FloatTensor(paths_nodes[i]).to(device)
            path_fea = self.path_net(paths_nodes_emb).unsqueeze(0)
            path_att = self.attnet(path_fea)
            paths_fea.append(path_fea)
            atts.append(path_att)

        paths_fea_emb = torch.cat(paths_fea, dim=0)
        atts = torch.cat(atts, dim=0)
        atts_soft = self.att_softmax(atts)
        user_fea = torch.sum(atts_soft * paths_fea_emb, 0)

        return user_fea

    def get_pred_scores(self, paths_nodes_embeds, train_product_embeds):
        user_feas_embed = self.forward(paths_nodes_embeds)
        train_product_embeds_tensor = torch.FloatTensor(train_product_embeds).to(device)
        user_feas_embeds = user_feas_embed.repeat(len(train_product_embeds), 1)
        user_product_cat = torch.cat([train_product_embeds_tensor, user_feas_embeds], dim=1)
        scores_s = self.predict(user_product_cat).squeeze(1)
        self.save_preds.append(scores_s)

        return scores_s.detach().cpu().numpy()

    # 使用重叠用户的评分训练注意力网络
    def update(self, optimizer, batch_scores, device):
        if len(batch_scores) <= 0:
            del self.save_preds[:]
            return 0.0

        att_loss = 0
        # loss函数
        criterion = torch.nn.MSELoss()
        for i in range(len(batch_scores)):
            # 转化为张量
            preds = self.save_preds[i]
            scores = torch.FloatTensor(batch_scores[i]).to(device)
            # 归一化处理
            preds_soft = F.softmax(preds, 0)
            preds_final = preds_soft * 5

            # 计算loss
            loss = criterion(preds_final, scores)
            att_loss += loss
        att_loss = torch.div(att_loss, float(len(batch_scores)))
        # 清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        att_loss.backward()
        # 根据梯度更新当前网络
        optimizer.step()
        # 返回损失的值

        del self.save_preds[:]

        return att_loss.item()



# 打乱顺序，分批返回uid
class AttentionDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        # 产生一个随机序列 0-->num_users-1
        # np.random.permutation函数的作用就是按照给定列表生成一个打乱后的随机列表
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None

            # 每一个batch返回多个用户
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]  # 不包括end_idx
        # 批量返回用户id
        batch_uids = self.uids[batch_idx]  #

        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()


# 获取路径嵌入
def paths_feature(path_file, uids, kg_utils, topn_paths):
    # 获取路径
    results = pickle.load(open(path_file, 'rb'))

    # 存路径
    user_paths = {}  # {uid: [path,...], ...}
    user_num = 0
    item_num = 0
    # 遍历全部路径,把有效路径存入字典
    for path, probs, rewards in zip(results['paths'], results['probs'], results['rewards']):
        # 去掉无效路径（最后一个实体不是目标域的用户或项的路径）
        node_id = path[-1][1]
        # if kg_utils.get_entity_info(node_id).get('domain_name') != TARGET:
        if not kg_utils.check_entity_domain(node_id, TARGET):
            continue
        skip = True
        if kg_utils.get_entity_info(node_id).get('type') == USER or kg_utils.get_entity_info(node_id).get(
                'type') == PRODUCT:
            # if kg_utils.get_entity_info(node_id).get('type') == PRODUCT:
            skip = False
            if kg_utils.get_entity_info(node_id).get('type') == USER:
                user_num +=1
            if kg_utils.get_entity_info(node_id).get('type') == PRODUCT:
                item_num +=1

        if skip:
            continue

        uid = path[0][1]
        # 用字典列表存起来（可能用到路径数量、路径长度、奖励、概率）
        if uid not in user_paths:
            user_paths[uid] = []
        # 计算路径概率
        path_len = len(path)
        if path_len == 1:
            continue
        elif path_len == 2:
            path_prob = probs[1]
            path_rewards = rewards[1]
        else:
            path_prob = reduce(lambda x, y: x * y, probs[1:])
            path_rewards = reduce(lambda x, y: x + y, rewards)
        user_paths[uid].append([path, path_prob, path_len, path_rewards])

    print("user_num:" + str(user_num))
    print("item_num:" + str(item_num))
    # 选择前n条路径，计算路径嵌入
    n = topn_paths
    sorted_paths = {uid: [] for uid in user_paths}

    path_entity_embeds = {uid: [] for uid in user_paths}
    for uid in user_paths:
        # 判断每个用户路径条数、长度、概率# 概率降序# 长度升序reverse = False升序（默认）
        # sort_paths = sorted(user_paths[uid], key=lambda x: x[3], reverse=True)
        sort_paths = sorted(user_paths[uid], key=lambda x: (x[3], x[1]), reverse=True)
        sort_paths = sorted(sort_paths, key=lambda x: x[2], reverse=False)

        for p, _, _, _ in sort_paths:
            if len(sorted_paths[uid]) >= n:
                break
            if p not in sorted_paths[uid]:
                sorted_paths[uid].append(p)

        for sorted_path in sorted_paths[uid]:
            path_entity_embed = []
            for node in sorted_path:
                eid = node[1]
                path_entity_embed.append(kg_utils.get_entity_info(eid).get('embed'))
                info = kg_utils.get_entity_info(eid)
                # print(info)
            path_entity_embeds[uid].append(path_entity_embed)

    for uid in uids:
        path_entity_embed = []
        if uid not in user_paths:
            path_entity_embed.append(kg_utils.get_entity_info(uid).get('embed'))
            path_entity_embeds[uid] = [path_entity_embed]

    return path_entity_embeds


def train(args):
    kg_utils = KGUtils(GRAPH_PATH, Entities_EMBED_PATH, RELATIONS_EMBED_PATH)
    # 获取训练用户数据集 uid, socre
    train_datas = load_attention_network_train_data(ATTENTION_NETWORK_TRAIN_PATH)  # ATTENTION_NETWORK_TRAIN_PATH
    train_uids = list(train_datas.keys())

    # 引入注意力网络
    attDataLoader = AttentionDataLoader(train_uids, args.att_batch_size)
    model = AttentionNetwork(args.entity_embedding_size, args.attention_hidden_size).to(args.device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.att_lr)
    lr = 1
    # 记录全部损失
    total_losses = []
    # 记录参数更新次数
    step = 0

    pred_paths = paths_feature(ATTENTION_TRAIN_PATH_FILE, train_uids, kg_utils, args.topn_paths)
    model.train()
    for epoch in range(1, args.epochs + 1):
        # 显示进度
        bar = tqdm(total=len(train_uids), desc='Epoch {}:'.format(epoch))
        batch_size = args.att_batch_size
        attDataLoader.reset()

        while attDataLoader.has_next():
            # 一批的用户id，评分，预测评分，用户嵌入
            batch_uids = attDataLoader.get_batch()
            batch_scores = []

            for uid in batch_uids:
                train_scores = []
                train_product_embeds = []

                for sample in train_datas[uid]:
                    for train_data in train_datas[uid][sample]:
                        # 获取每个用户交互过的项和评分
                        pid = train_data['product']
                        score = train_data['score']

                        # 存储评分、交互过项的嵌入
                        train_scores.append(score)
                        train_product_embeds.append(kg_utils.get_entity_info(pid).get(EMBED))


                # 获取路径嵌入
                if pred_paths.get(uid, -1) == -1:
                    continue
                else:
                    paths_nodes_embeds = pred_paths[uid]
                # 存储用户嵌入、评分、预测评分
                pre_scores = model.get_pred_scores(paths_nodes_embeds, train_product_embeds)
                batch_scores.append(train_scores)

            lr = args.att_lr * max(1e-2, 1.0 - float(step) / (args.epochs * len(train_uids) / args.att_batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            att_loss = model.update(optimizer, batch_scores, args.device)
            total_losses.append(att_loss)
            step += 1
            bar.update(batch_size)

            if step > 0 and step % 5 == 0:
                avg_loss = np.mean(total_losses)
                total_losses = []
                logger.info(
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss))

        # 保留模型参数
        # 每5轮保存一次
        if epoch % 10 == 0:
            path_reasoning_hops = eval_utils.get_path_resoning_hops(ATTENTION_TRAIN_PATH_FILE)
            att_file = '{}/2att_2mlp_model_time_{}_pathReason_{}_topPaths_{}_acts_{}_rewardW_{}_epoch_{}.ckpt'.format(args.log_dir, curr_time, path_reasoning_hops, args.topn_paths, MAX_ACTS, REWARD_WEIGHTS, epoch)
            logger.info("Save model to " + att_file)
            # model.state_dict()获取模型全部参数
            torch.save(model.state_dict(), att_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=EXAMPLE, help='')
    parser.add_argument('--name', type=str, default='train_att', help='')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--att_batch_size', type=int, default=64, help='batch size of AttentionNetwork train')
    parser.add_argument('--attention_hidden_size', type=int, default=128, help='hidden size of attention network')
    parser.add_argument('--entity_embedding_size', type=int, default=ENTITY_EMBEDDING_SIZE, help='entity emddding size')
    parser.add_argument('--att_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs of Train att')
    parser.add_argument('--topn_paths', type=int, default=30, help='Top N paths')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    global device
    device = args.device

    global curr_time
    curr_time = time()

    # 训练日志/tmp/dataset/train_att
    args.log_dir = '{}/{}/{}'.format(TMP_DIR[args.dataset], args.name, '2att_2mlp')
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_2att_2mlp_time_{}_top_{}_acts_{}_rewardW_{}_log.txt'.format(curr_time, args.topn_paths, MAX_ACTS, REWARD_WEIGHTS))
    # 保存超参
    logger.info(args)
    # 保存训练的路径文件
    logger.info("ATTENTION_TRAIN_PATH_FILE: " + ATTENTION_TRAIN_PATH_FILE)

    global logger_score
    logger_score = get_logger(args.log_dir + '/train_2att_2mlp_time_{}_top_{}_acts_{}_rewardW_{}_score.txt'.format(curr_time, args.topn_paths, MAX_ACTS, REWARD_WEIGHTS))
    # 开启训练
    train(args)


if __name__ == "__main__":
    for iter_time in range(0, 5):
        print("-------------------The time of training: {}----------------------".format(iter_time+1))
        main()
