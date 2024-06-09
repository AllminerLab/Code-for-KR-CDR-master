import argparse
import os
from utils import *
import torch
from train_meta import *

# 日志，全局变量
logger = None


class TerminalReward(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default=EXAMPLE, help='')
        parser.add_argument('--train_name', type=str, default='train_meta', help='')
        parser.add_argument('--name', type=str, default='test_meta', help='')
        parser.add_argument('--seed', type=int, default=2022, help='random seed')
        parser.add_argument('--gpu', type=str, default='0', help='gpu device')
        parser.add_argument('--meta_batch_size', type=int, default=None, help='batch size of metaNetwork train')
        parser.add_argument('--meta_hidden_size', type=int, default=256, help='hidden size of meta network')
        parser.add_argument('--attention_hidden_size', type=int, default=128, help='hidden size of attention network')
        parser.add_argument('--entity_embedding_size', type=int, default=ENTITY_EMBEDDING_SIZE, help='entity emddding size')
        parser.add_argument('--meta_lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs')
        parser.add_argument('--is_train', type=int, default=1, help='transition of train and test')
        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

        # 测试日志/tmp/dataset/test_meta
        args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)

        # 训练日志目录
        self.train_log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.train_name)

        # 打开测试日志 /tmp/dataset/test_meta/test_meta.txt
        logger = get_logger(args.log_dir + '/test_meta.txt')
        # 保存超参
        logger.info(args)
        # 设置随机种子
        set_random_seed(args.seed)
        self.args = args
        # 引入元网络
        self.model = RewardFunction(args.entity_embedding_size, args.meta_hidden_size, args.attention_hidden_size).to(args.device)
        # 导入训练好的参数
        meta_file = self.train_log_dir + '/meta_model_epoch_{}.ckpt'.format(args.epochs)
        pretrained_sd = torch.load(meta_file)
        model_sd = self.model.state_dict()
        model_sd.update(pretrained_sd)
        self.model.load_state_dict(model_sd)

    def get_target_reward(self, src_entities_emb, entity_emb):
        if len(src_entities_emb) <= 0:
            return 0

        # 不更新参数
        self.model.eval()
        with torch.no_grad():
            reward_list = self.model.forward(src_entities_emb, entity_emb, self.args.device)

        return reward_list[0]

