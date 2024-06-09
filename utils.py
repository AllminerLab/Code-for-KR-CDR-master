import logging
import logging.handlers
import os
import random
import sys
import numpy as np
import torch
from knowledge_graph import kg_utils

# Entity Embedding Size
ENTITY_EMBEDDING_SIZE = 128

MAX_ACTS = 250
REWARD_WEIGHTS = [0.7, 0.2, 0.1]
# Dataset names.
EXAMPLE = 'FASHION_Software'

version = 'v4'

# Model result directories
TMP_DIR = {
    EXAMPLE: './tmp/' + EXAMPLE + '/' + version

}
KG_DIR_DICT = {
    'FASHION_Software': './data/FASHION_Software/'
}
KG_DIR = KG_DIR_DICT[EXAMPLE]
GRAPH_PATH_DICT = {
    'FASHION_Software': KG_DIR + 'kg/FASHION-Software-graph.pkl',


}
Entities_EMBED_DICT = {
    'FASHION_Software': {
        # version 1  add_embedding
        'v1': KG_DIR + 'kg/FASHION_Software-add-entities-embedding.npy',
        # version 2   source
        'v2': KG_DIR + 'kg/FASHION_Software-source-entities-embedding.npy',
        # version 3   add-no-user
        'v4': KG_DIR + 'kg/FASHION_Software-add-no-user-entities-embedding.npy',
        # version 4   target
        'v3': KG_DIR + 'kg/FASHION_Software-target-entities-embedding.npy',
    }
}
RELATIONS_EMBED_DICT = {
    'FASHION_Software': {
        # version 1  add_embedding
        'v1': KG_DIR + 'kg/FASHION_Software-add-relations-embedding.npy',
        # version 2   source
        'v2': KG_DIR + 'kg/FASHION_Software-source-relations-embedding.npy',
        # version 3   add-no-user
        'v4': KG_DIR + 'kg/FASHION_Software-add-no-user-relations-embedding.npy',
        # version 4   target
        'v3': KG_DIR + 'kg/FASHION_Software-target-relations-embedding.npy'
    }
}
GRAPH_PATH = GRAPH_PATH_DICT[EXAMPLE]
Entities_EMBED_PATH = Entities_EMBED_DICT[EXAMPLE][version]
RELATIONS_EMBED_PATH = RELATIONS_EMBED_DICT[EXAMPLE][version]
#
#
# Train or Test Data Path
TEST_DATA_PATH = KG_DIR + 'test_data.pkl'
TEST_NEGATIVE_DATA_PATH = KG_DIR + 'test_negative_data.pkl'
ATTENTION_NETWORK_TRAIN_PATH = KG_DIR + 'attention_network_train_data.pkl'
META_NETWORK_TRAIN_PATH = KG_DIR + 'meta_network_train_data.pkl'

# Path File
ATTENTION_TRAIN_PATH_FILE = TMP_DIR[EXAMPLE]+'/path_reasoning/train_policy_paths_pathLen_5_topK_[25, 5, 1]_acts_{}_rewar_{}_epoch_20.pkl'.format(MAX_ACTS, REWARD_WEIGHTS)
TEST_PATH_FILE = TMP_DIR[EXAMPLE] + '/path_reasoning/test_policy_paths_pathLen_5_topK_[25, 5, 1]_acts_{}_rewar_{}_epoch_20.pkl'.format(MAX_ACTS, REWARD_WEIGHTS)

ATTENTION_TRAIN_PATH_FILE_RANDOM = TMP_DIR[EXAMPLE]+'/path_reasoning/train_random_paths_pathLen_5_topK_[25, 5, 1]_epoch_20.pkl'
TEST_PATH_FILE_RANDOM = TMP_DIR[EXAMPLE] + '/path_reasoning/test_random_paths_pathLen_5_topK_[25, 5, 1]_epoch_20.pkl'

#
#
#
# Entities
USER = "USER"
PRODUCT = "PRODUCT"
WORD = "WORD"
BRAND = "BRAND"
CATEGORY = "CATEGORY"

# Relations
ALSO_BOUGHT = "ALSO_BOUGHT"
BELONG_TO = "BELONG_TO"
BOUGHT_TOGETHER = "BOUGHT_TOGETHER"
DESCRIBED_AS = "DESCRIBED_AS"
MENTION = "MENTION"
PRODUCED_BY = "PRODUCED_BY"
PURCHASE = "PURCHASE"
ALSO_VIEWED = "ALSO_VIEWED"
SAME_USER = 'SAME_USER'

# KnowledgeGraph dictionary key
ENTITY = 'entity'
RELATION = 'relation'
NAME = 'name'
TYPE = 'type'
DOMAIN_NAME = 'domain_name'
EMBED = 'embed'
ID = 'id'
SOURCE = 'source'
TARGET = 'target'
SAME = 'same'
POSITIVE = 'positive'
NEGATIVE = 'negative'
SCORE = 'score'


# 实体间关系
# KG_RELATION = {}


# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#获取日志文件 logname 日志文件，比如/tmp/dataset/train_agent/train_agent.txt
def get_logger(logname):
    logger = logging.getLogger(logname)
    # 日志等级
    logger.setLevel(logging.DEBUG)
    #log输出格式
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    #输出到控制台
    ch = logging.StreamHandler(sys.stdout)
    # 设置日志输出格式
    ch.setFormatter(formatter)
    # 添加到logger对象里
    logger.addHandler(ch)

    # 输出到文件
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    # 设置日志输出格式
    fh.setFormatter(formatter)
    # 添加到logger对象里
    logger.addHandler(fh)

    return logger


def make_dir(direction):
    if not os.path.isdir(direction):
        os.makedirs(direction)
    return direction
