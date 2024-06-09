
import argparse
import os

from torch.optim import Optimizer
from tqdm import tqdm

from knowledge_graph.data_utils import load_meta_network_train_data
from utils import *
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from knowledge_graph.kg_utils import *
from utils import *

# 日志, 全局变量
logger = None
logger_reward = None

class MetaNetwork(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim, att_hidden_dim):
        super().__init__()
        self.attnet = torch.nn.Sequential(torch.nn.Linear(emb_dim, att_hidden_dim), torch.nn.ReLU(),
                                          torch.nn.Linear(att_hidden_dim, 1))#.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.att_softmax = torch.nn.Softmax(dim=0)

        self.metanet = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim))#.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, user_entities_emb):
        att_scores = self.attnet(user_entities_emb)

        # 归一化处理
        atts = self.att_softmax(att_scores)
        fea_emb = torch.sum(atts * user_entities_emb, 0)

        weights = self.metanet(fea_emb)
        return weights


class RewardFunction(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim, att_hidden_dim):
        super().__init__()
        self.metaNetwork = MetaNetwork(emb_dim, meta_dim, att_hidden_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.emb_dim = emb_dim
        self.save_rewards = []

    def forward(self, src_entities_emb, batch_entity_emb, device):
        batch_entity_emb = torch.FloatTensor(np.array(batch_entity_emb)).to(device)
        src_entities_emb = torch.FloatTensor(np.array(src_entities_emb)).to(device)
        weight = self.metaNetwork.forward(src_entities_emb).unsqueeze(1).view(self.emb_dim, -1)
        rewards = torch.mm(batch_entity_emb, weight).squeeze(1)
        rewards_re = self.sigmoid(rewards) * 5
        self.save_rewards.append(rewards_re)

        return rewards_re.detach().cpu().numpy().tolist()

    def update(self, batch_entity_scores, optimizer, device):
        user_num = len(batch_entity_scores)
        if user_num <= 0:
            del self.save_rewards[:]
            return 0.0
        meta_loss = 0.0
        for i in range(user_num):
            rewards = self.save_rewards[i]
            entity_scores = torch.FloatTensor(batch_entity_scores[i]).to(device)
            # 归一化处理
            rewards = F.softmax(rewards, 0)
            entity_scores = F.softmax(entity_scores, 0)
            meta_loss += F.binary_cross_entropy(rewards, entity_scores)

        meta_loss = torch.div(meta_loss, float(user_num))
        optimizer.zero_grad()
        # meta_loss.requires_grad_(True)
        meta_loss.backward()
        optimizer.step()

        del self.save_rewards[:]

        return meta_loss.item()


class MetaDataLoader(object):
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


def train(args):

    def get_inter_items_embed(node_id):
        neigh_list = kg_utils.get_entity_neighbors(node_id)
        item_embed_list = []
        for relation_entity in neigh_list:
            entity_info = relation_entity.get(ENTITY)
            if kg_utils.check_entity_domain(entity_info.get(ID), SOURCE) and entity_info.get(TYPE) == PRODUCT:
                item_embed_list.append(entity_info.get(EMBED))
        return item_embed_list

    def get_train_data():
        data_dict = load_meta_network_train_data(META_NETWORK_TRAIN_PATH)
        uids = list(data_dict.keys())
        # 记录有交互的用户
        valid_uids = []
        uid_iids_dict = {}
        for uid in uids:
            iids_embed = []
            iids_score = []
            entity_score_list = []
            positive_entity_score_list = data_dict.get(uid).get(POSITIVE)
            negative_entity_score_list = data_dict.get(uid).get(NEGATIVE)
            # 合并正负样本
            entity_score_list.extend(positive_entity_score_list)
            entity_score_list.extend(negative_entity_score_list)
            # 判断用户在目标域中是否有交互
            if len(entity_score_list) <= 0:
                continue
            else:
                valid_uids.append(uid)

            # 获取样本信息
            for entity_score_dict in entity_score_list:
                entity_id = entity_score_dict.get('product')
                score = entity_score_dict.get('score')
                iids_embed.append(kg_utils.get_entity_info(entity_id).get(EMBED))
                iids_score.append(score)

            iids_info = [iids_embed, iids_score]
            uid_iids_dict[uid] = iids_info

        return valid_uids, uid_iids_dict

    kg_utils = KGUtils(GRAPH_PATH, Entities_EMBED_PATH, RELATIONS_EMBED_PATH)
    # 获取训练用户数据集 uid, socre
    train_uids, uis_iids_dict = get_train_data()

    metaDataLoader = MetaDataLoader(train_uids, args.meta_batch_size)
    model = RewardFunction(args.entity_embedding_size, args.meta_hidden_size, args.attention_hidden_size).to(args.device)
    logger.info('Meta Parameters:' + str([i[0] for i in model.named_parameters()]))
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
    # 记录全部损失
    total_losses = []
    # 记录参数更新次数
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        # 显示进度
        bar = tqdm(total=len(train_uids), desc='Epoch {}:'.format(epoch))
        batch_size = args.meta_batch_size
        metaDataLoader.reset()
        while metaDataLoader.has_next():
            batch_uids = metaDataLoader.get_batch()
            batch_rewards = []
            batch_entity_scores = []

            for uid in batch_uids:
                # 获取交互过的项的嵌入集合
                src_entities_emb = get_inter_items_embed(uid)
                # 包括用户本身
                src_entities_emb.append(kg_utils.get_entity_info(uid).get(EMBED))
                # 获取训练集中用户对应的项的嵌入
                tgt_entity_emb = uis_iids_dict[uid][0]
                # 获取项对应的评分=、
                entity_scores = uis_iids_dict[uid][1]
                # 计算奖励
                rewards = model.forward(src_entities_emb, tgt_entity_emb, args.device)
                # 存储奖励与评分
                batch_rewards.append(rewards)
                batch_entity_scores.append(entity_scores)

            for i in range(len(batch_rewards)):
                msg = "rewards:"+str(batch_rewards[i])+"----scores:"+str(batch_entity_scores[i])
                # print(msg)
                logger_reward.info('epoch:' + str(epoch) + '----rewards:' + str(batch_rewards[i]) + '----scores' + str(
                    batch_entity_scores[i]))

            meta_loss = model.update(batch_entity_scores, optimizer, args.device)

            total_losses.append(meta_loss)
            step += 1
            bar.update(batch_size)

            if step > 0 and step % 1 == 0:
                avg_loss = np.mean(total_losses)
                total_losses = []
                logger.info(
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss))

        # 保留模型参数
        # '/tmp/dataset/train_agent/{}/meta_model_epoch_{}.ckpt'.format(args.log_dir, epoch)
        meta_file = '{}/meta_model_epoch_{}.ckpt'.format(args.log_dir, epoch)
        logger.info("Save model to " + meta_file)
        # model.state_dict()获取模型全部参数
        torch.save(model.state_dict(), meta_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=EXAMPLE, help='')
    parser.add_argument('--name', type=str, default='train_meta', help='')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--meta_batch_size', type=int, default=64, help='batch size of metaNetwork train')
    parser.add_argument('--meta_hidden_size', type=int, default=256, help='hidden size of meta network')
    parser.add_argument('--attention_hidden_size', type=int, default=128, help='hidden size of attention network')
    parser.add_argument('--entity_embedding_size', type=int, default=ENTITY_EMBEDDING_SIZE, help='entity emddding size')
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    # 训练日志/tmp/dataset/train_meta
    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    # 打开训练日志 /tmp/dataset/train_agent/train_meta.txt
    logger = get_logger(args.log_dir + '/train_meta_log.txt')
    # 保存超参
    logger.info(args)
    # # 设置随机种子
    # set_random_seed(args.seed)

    global logger_reward
    logger_reward = get_logger(args.log_dir + '/train_meta_reward_log.txt')

    # 开启训练
    train(args)


if __name__ == "__main__":
    main()
