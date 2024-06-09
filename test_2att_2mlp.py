import argparse
import os

from torch.optim import Optimizer
from tqdm import tqdm

import eval_utils
from knowledge_graph.kg_utils import KGUtils
from knowledge_graph.data_utils import load_test_data
from train_2att_2mlp import AttentionNetwork, paths_feature
from utils import *
import torch

import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pickle
from scipy import spatial
from functools import reduce
from math import log
import time
from tqdm import tqdm


def scores_calculate(user_feas_embed, train_product_embeds, embed_size) :
    pred_scores = []
    for train_product_embed in train_product_embeds:
        # pred_score = np.dot(user_feas_embed, train_product_embed)
        pred_score = 0
        for j in range(embed_size):
            pred_score += user_feas_embed[j] * train_product_embed[j]
        pred_scores.append(pred_score)
    pred_scores = np.maximum(0, pred_scores)
    return pred_scores


def test(args, att_file, kg_utils, path_file, epoch):
    # 测试集
    test_datas = load_test_data(TEST_DATA_PATH)
    test_nega_sample_datas = load_test_data(TEST_NEGATIVE_DATA_PATH)
    # 引入注意力网络
    model = AttentionNetwork(args.entity_embedding_size, args.attention_hidden_size)
    logger.info('epoch {} Test。。。'.format(epoch))

    pretrained_sd = torch.load(att_file)
    model_sd = model.state_dict()
    model_sd.update(pretrained_sd)
    model.load_state_dict(model_sd)

    test_uids = list(test_datas.keys())
    #              2058, 2059]
    test_scores = {uid: [] for uid in test_datas}
    test_sort_scores = {uid: [] for uid in test_datas}
    test_user_preds_scores = {uid: [] for uid in test_datas}
    test_pids = []
    test_product_embeds = []
    test_user_product_embeds = {uid: [] for uid in test_datas}
    test_user_pids = {uid: [] for uid in test_datas}
    test_sort_user_pids = {uid: [] for uid in test_datas}
    test_sample_user_product_embeds = {uid: [] for uid in test_datas}
    test_sample_user_pids = {uid: [] for uid in test_datas}
    pred_user_pids = {uid: [] for uid in test_datas}
    pred_user_all_pids = {uid: [] for uid in test_datas}
    total_sample_user_pids = {uid: [] for uid in test_datas}
    total_sample_user_scores = {uid: [] for uid in test_datas}
    # 目标域用户特征嵌入集合
    test_user_feas_embeds = []


    pred_paths = paths_feature(TEST_PATH_FILE, test_uids, kg_utils, args.topn_paths)

    print('Predicting Scores:')
    for uid in test_uids:
        for test_data in test_datas[uid]:
            # 获取每个用户交互过的项和评分，所有的项
            pid = test_data['product']
            score = test_data['score']
            # 存储用户嵌入、评分、预测评分
            if pid not in test_pids:
                test_pids.append(pid)
                test_product_embeds.append(kg_utils.get_entity_info(pid).get('embed'))
            test_user_pids[uid].append(pid)
            test_scores[uid].append(score)
            test_user_product_embeds[uid].append(kg_utils.get_entity_info(pid).get('embed'))
            test_sample_user_pids[uid].append(pid)
            test_sample_user_product_embeds[uid].append(kg_utils.get_entity_info(pid).get('embed'))

        # 获取负样本
        for test_nega_sample_data in test_nega_sample_datas[uid]:
            pid = test_nega_sample_data.get('product')
            if len(test_sample_user_pids[uid]) >= args.sample_size:
                break
            test_sample_user_pids[uid].append(pid)
            test_sample_user_product_embeds[uid].append(kg_utils.get_entity_info(pid).get('embed'))


        # 获取路径嵌入
        if pred_paths.get(uid, -1) == -1:
            continue
        path_embeds = pred_paths[uid]
        # 获取并存储目标域用户特征嵌入
        model.eval()
        with torch.no_grad():
            test_user_preds_scores[uid] = model.get_pred_scores(path_embeds, test_user_product_embeds[uid]).tolist()

    # 预测评分
    print('Predicting Topn items:')
    for i, uid in enumerate(test_uids):
        # 获取用户和样本里的项的预测评分
        if pred_paths.get(uid, -1) == -1:
            continue

        model.eval()
        with torch.no_grad():
            test_preds_scores = model.get_pred_scores(pred_paths[uid],  test_sample_user_product_embeds[uid])
        # 选择每个用户评分Topk的项和评分
        idxs = np.argsort(test_preds_scores)
        topn_idxs = idxs[::-1][:args.topk]
        for topn_idx in topn_idxs:
            pred_user_pids[uid].append(test_sample_user_pids[uid][topn_idx])
        # 存储每个用户的抽样的项及其对应的评分
        for idx in idxs[::-1]:
            total_sample_user_pids[uid].append(test_sample_user_pids[uid][idx])
            total_sample_user_scores[uid].append(test_preds_scores[idx])

        # 分数升序
        for idx in idxs:
            pred_user_all_pids[uid].append(test_sample_user_pids[uid][idx])
        # 给用户交互过的项按真实评分倒序排序
        idxs = np.argsort(test_scores[uid])
        for idx in idxs[::-1]:
            test_sort_user_pids[uid].append(test_user_pids[uid][idx])
            test_sort_scores[uid].append(test_scores[uid][idx])

    # 计算评价指标precisions, recalls, ndcgs, hits, f1_scores
    ndcgs, recalls, precisions, hits, f1_scores, accs = eval_utils.evaluate(pred_user_pids, test_sort_user_pids, test_sort_scores, args.sample_size, args.topk)
    # 计算评价指标auc
    aucs = eval_utils.eval_auc(pred_user_pids, test_sort_user_pids, args.topk)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)
    avg_hit = np.mean(hits)
    avg_auc = np.mean(aucs)
    avg_acc = np.mean(accs)
    avg_f1_score = np.mean(f1_scores)

    logger.info(
        'NDCG={:.4f} |  Recall={:.4f} | HR={:.4f} | Precision={:.4f} | AUC={:.4f} | ACC={:.4f} | F1_Score={:.4f}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, avg_auc, avg_acc, avg_f1_score))

    # 保留评价指标
    logger.info(
        ' | ndcgs={' + str(ndcgs) + '}' +
        ' | recalls={' + str(recalls) + '}' +
        ' | precisions={' + str(precisions) + '}' +
        ' | hits={' + str(hits) + '}' +
        ' | aucs={' + str(aucs) + '}' +
        ' | accs={' + str(accs) + '}' +
        ' | f1_scores={' + str(f1_scores) + '}' +
        ' | avg_ndcg={:.4f}'.format(avg_ndcg) +
        ' | avg_recall={:.4f}'.format(avg_recall) +
        ' | avg_precision={:.4f}'.format(avg_precision) +
        ' | avg_hit={:.4f}'.format(avg_hit) +
        ' | avg_auc={:.4f}'.format(avg_auc) +
        ' | avg_acc={:.4f}'.format(avg_acc) +
        ' | avg_f1_score={:.4f}'.format(avg_f1_score))

    eval_utils.save_predict_data(path_file, total_sample_user_pids, total_sample_user_scores, test_sort_user_pids, test_sort_scores)
    logger.info("Data have saved to"+path_file)

    # 返回数据
    return avg_ndcg, avg_recall, avg_hit, avg_precision, avg_auc, avg_acc, avg_f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=EXAMPLE, help='')
    parser.add_argument('--name', type=str, default='test_att', help='')
    parser.add_argument('--train_name', type=str, default='train_att', help='DIR')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--att_batch_size', type=int, default=None, help='batch size of AttentionNetwork train')
    # parser.add_argument('--att_hidden_size', type=int, default=None, help='hidden size of att network')
    parser.add_argument('--attention_hidden_size', type=int, default=128, help='hidden size of attention network')
    parser.add_argument('--entity_embedding_size', type=int, default=ENTITY_EMBEDDING_SIZE, help='entity embeding size')
    parser.add_argument('--att_lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs of train att')
    parser.add_argument('--test_epochs', type=int, default=5, help='Max number of epochs of Test att')
    parser.add_argument('--topn_paths', type=int, default=30, help='')
    parser.add_argument('--train_topn_paths', type=int, default=30, help='')
    parser.add_argument('--sample_size', type=int, default=100, help='sample_size')
    parser.add_argument('--topk', type=int, default=10, help='topk')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    path_reasoning_hops = eval_utils.get_path_resoning_hops(TEST_PATH_FILE)
    # 测试日志/tmp/dataset/test_att
    args.log_dir = '{}/{}/{}/{}'.format(TMP_DIR[args.dataset], args.name,  '2att_2mlp', 'pathReason_' + str(path_reasoning_hops) + '_topPaths_' + str(args.topn_paths)
                                        + '_acts_{}_rewardW_{}'.format(MAX_ACTS, REWARD_WEIGHTS))
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global statistics_logger
    # 存储最终统计结果
    statistics_logger = get_logger(
        make_dir(args.log_dir+'/statistic') + '/test_2att_2mlp_Ep{}_traPa{}_testPa{}_act{}_rew{}_tpI{}_sta.txt'.format(
            args.epochs, args.train_topn_paths, args.topn_paths, MAX_ACTS, REWARD_WEIGHTS, args.topk))
    # 保存超参
    statistics_logger.info(args)
    # 保存测试的路径文件路径
    statistics_logger.info("TEST_PATH_FILE: " + TEST_PATH_FILE)
    # 设置随机种子
    set_random_seed(args.seed)

    # 训练日志目录
    train_log_dir = '{}/{}/{}'.format(TMP_DIR[args.dataset], args.train_name, '2att_2mlp')
    # 获取多个存放注意力网络参数的目标文件
    # att_file = train_log_dir + '/2att_relu_model_top_{}_epoch_{}.ckpt'.format(args.train_topn_paths, args.epochs)
    train_att_file_names = eval_utils.find_target_att_file(train_log_dir, 'pathReason_{}_topPaths_{}_acts_{}_rewardW_{}_epoch_{}'.format(path_reasoning_hops, args.train_topn_paths, MAX_ACTS, REWARD_WEIGHTS, args.epochs))
    assert len(train_att_file_names) >= args.test_epochs
    train_att_file_names = train_att_file_names[:args.test_epochs]
    statistics_logger.info('train_att_file_names:' + str(train_att_file_names))


    global logger
    kg_utils = KGUtils(GRAPH_PATH, Entities_EMBED_PATH, RELATIONS_EMBED_PATH)
    # 执行多次tests
    ndcgs, recalls, hits, precisions, aucs, accs, f1_scores = [], [], [], [], [], [], []
    for epoch in range(1, args.test_epochs + 1):
        # 打开测试日志
        logger = get_logger(
            make_dir(args.log_dir+'/data_storage') + '/test_2att_2mlp_attEp{}_trP{}_topP{}_tpI{}_act{}_r{}_ep_{}.txt'.format(
                args.epochs, args.train_topn_paths, args.topn_paths, args.topk, MAX_ACTS, REWARD_WEIGHTS, epoch))
        # 存放实验数据的日志
        path_file = make_dir(args.log_dir+'/data_storage') + '/test_2att_2mlp_atEp{}_traP{}_tP{}_sam{}_ac{}_rew{}_ep{}.csv'.format(
            args.epochs, args.train_topn_paths, args.topn_paths, args.sample_size, MAX_ACTS, REWARD_WEIGHTS, epoch)
        # 导入训练好的参数的文件
        att_file = train_log_dir + '/' + train_att_file_names[epoch-1]
        # 开启测试
        avg_ndcg, avg_recall, avg_hit, avg_precision, avg_auc, avg_acc, avg_f1_score = test(args, att_file, kg_utils, path_file, epoch)
        ndcgs.append(avg_ndcg)
        recalls.append(avg_recall)
        hits.append(avg_hit)
        precisions.append(avg_precision)
        aucs.append(avg_auc)
        accs.append(avg_acc)
        f1_scores.append(avg_f1_score)

    # 处理保存的多轮数据
    # 取平均值
    ndcg = np.mean(ndcgs)
    recall = np.mean(recalls)
    hit = np.mean(hits)
    precision = np.mean(precisions)
    auc = np.mean(aucs)
    acc = np.mean(accs)
    f1_score = np.mean(f1_scores)
    # 保存数据
    statistics_logger.info("Here are mean values:")
    statistics_logger.info(
        'NDCG={:.4f} |  Recall={:.4f} | HR={:.4f} | Precision={:.4f} | AUC={:.4f} | ACC={:.4f} | F1_Score={:.4f}'.format(
            ndcg, recall, hit, precision, auc, acc, f1_score))

    # 计算standrad deviation标准差
    std_ndcg = np.std(ndcgs)
    std_recall = np.std(recalls)
    std_hit = np.std(hits)
    std_precision = np.std(precisions)
    std_auc = np.std(aucs)
    std_acc = np.std(accs)
    std_f1_score = np.std(f1_scores)
    statistics_logger.info("Here are standrad deviations:")
    statistics_logger.info(
        'NDCG={:.4f} |  Recall={:.4f} | HR={:.4f} | Precision={:.4f} | AUC={:.4f} | ACC={:.4f} | F1_Score={:.4f}'.format(
            std_ndcg, std_recall, std_hit, std_precision, std_auc, std_acc, std_f1_score))

    # 取最优值
    max_ndcg = max(ndcgs)
    max_recall = max(recalls)
    max_hit = max(hits)
    max_precision = max(precisions)
    max_auc = max(aucs)
    max_acc = max(accs)
    max_f1_score = max(f1_scores)
    # 保存数据
    statistics_logger.info("Here are optimal values:")
    statistics_logger.info(
        'NDCG={:.4f} |  Recall={:.4f} | HR={:.4f} | Precision={:.4f} | AUC={:.4f} | ACC={:.4f} | F1_Score={:.4f}'.format(
            max_ndcg, max_recall, max_hit, max_precision, max_auc, max_acc, max_f1_score))

    # 计算 variance方差
    var_ndcg = np.var(ndcgs)
    var_recall = np.var(recalls)
    var_hit = np.var(hits)
    var_precision = np.var(precisions)
    var_auc = np.var(aucs)
    var_acc = np.var(accs)
    var_f1_score = np.var(f1_scores)
    statistics_logger.info("Here are variances:")
    statistics_logger.info(
        'NDCG={:.4f} |  Recall={:.4f} | HR={:.4f} | Precision={:.4f} | AUC={:.4f} | ACC={:.4f} | F1_Score={:.4f}'.format(
            var_ndcg, var_recall, var_hit, var_precision, var_auc, var_acc, var_f1_score))

    # 保存每轮数据
    statistics_logger.info("This are the data per epoch:")
    for epoch in range(args.test_epochs):
        statistics_logger.info(
            'epoch:{} |  NDCG={} |  Recall={} | HR={} | Precision={} | AUC={} | ACC={} | F1_Score={}'.format(
                epoch + 1, ndcgs[epoch], recalls[epoch], hits[epoch], precisions[epoch], aucs[epoch], accs[epoch],
                f1_scores[epoch]))

    # 采用csv格式保存均值和标准差
    csv_path = args.log_dir + '/test_2att_2mlp_attEpo_{}_traTopPa_{}_testTopPa_{}_act{}_rewar{}_topItems{}_Sta.csv'.format(
        args.epochs, args.train_topn_paths, args.topn_paths, MAX_ACTS, REWARD_WEIGHTS, args.topk)
    ndcg, recall, hit, precision, auc, acc, f1_score = eval_utils.keep_decimal(ndcg, recall, hit, precision, auc, acc,
                                                                                           f1_score, 4)
    std_ndcg, std_recall, std_hit, std_precision, std_auc, std_acc, std_f1_score = eval_utils.keep_decimal(
        std_ndcg, std_recall, std_hit, std_precision, std_auc, std_acc, std_f1_score, 4)
    mean_dict = {'NDCG': ndcg, 'Recall': recall, 'HR': hit, 'Precision': precision, 'AUC': auc, 'ACC': acc,
                 'F1_Score': f1_score}
    std_dict = {'NDCG': std_ndcg, 'Recall': std_recall, 'HR': std_hit, 'Precision': std_precision, 'AUC': std_auc,
                'ACC': std_acc, 'F1_Score': std_f1_score}
    save_result = {'mean': mean_dict, 'std_dev': std_dict}
    eval_utils.save_eval_data(csv_path, save_result)


if __name__ == "__main__":
    main()