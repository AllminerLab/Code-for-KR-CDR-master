import copy
import os
import pickle
from math import log
import pandas as pd

import torch
import torch.nn.functional as F


# 计算auc
def eval_auc(pred_matches, test_user_products, topk):
    print('Evaluating AUC:')
    aucs = []
    for uid in list(test_user_products.keys()):
        # 如果没预测该用户，或预测的推荐项的少于10个
        if uid not in pred_matches or len(pred_matches[uid]) < topk:
            continue
        size_post, rank = 0, 0
        for post_sample in test_user_products[uid]:
            if post_sample in pred_matches[uid]:
                size_post += 1
                rank += topk - pred_matches[uid].index(post_sample)
        size_nega = topk-size_post
        if size_post == 0:
            aucs.append(0)
            continue
        if size_nega == 0:
            aucs.append(1)
            continue
        auc = (rank - size_post*(size_post+1)/2)/(size_nega*size_post)
        aucs.append(auc)
    return aucs



# 计算评价指标
def evaluate(topk_matches, test_user_products, test_user_scores, sample_size, topk):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    print('Evaluating NDCG, Recall, HR, Precision, F1_Score:')
    precisions, recalls, ndcgs, hits, f1_scores, accs = [], [], [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        # 如果没预测该用户，或预测的推荐项的少于10个
        if uid not in topk_matches or len(topk_matches[uid]) < topk:
            # invalid_users.append(uid)
            continue
        pred_list, rel_pids = topk_matches[uid], test_user_products[uid]
        # if len(pred_list) == 0:
        #     continue
        dcg, idcg, hit_num = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_pids:
                index = rel_pids.index(pred_list[i])
                rel = test_user_scores[uid][index]
                dcg += rel / log(i + 2)
                hit_num += 1
        iindex = 0
        for i in range(len(rel_pids)):
            if rel_pids[i] in pred_list:
                rel = test_user_scores[uid][i]
                idcg += rel / log(iindex + 2)
                iindex += 1

        ndcg = dcg / idcg if idcg != 0.0 else 0.0
        recall = hit_num / len(rel_pids)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0
        f1_score = 2*precision*recall/(precision+recall+float("1e-8"))
        tn = (sample_size - len(test_user_products[uid])) - (topk - hit_num)
        acc = (hit_num + tn) / sample_size

        accs.append(acc)
        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)
        f1_scores.append(f1_score)

    return ndcgs, recalls, precisions, hits, f1_scores, accs


# 计算acc
def eval_acc(topk_matches, test_user_products, sample_size, topk):
    print('Evaluating ACC:')
    accs = []
    for uid in list(test_user_products.keys()):
        # 如果没预测该用户，或预测的推荐项的少于topk个
        if uid not in topk_matches or len(topk_matches[uid]) < topk:
            continue
        tp = 0
        # 如果项同时在测试集和预测集中
        for pid in topk_matches[uid]:
            if pid in test_user_products[uid]:
                tp += 1
        tn = (sample_size - len(test_user_products[uid])) - (topk - tp)
        acc = (tp + tn)/sample_size
        accs.append(acc)
    return accs

def save_predict_data(path_file, topk_users_pre_iids, topk_users_pre_scores, users_truth_iids, users_truth_scores):
    uids = list(topk_users_pre_iids.keys())
    results = []
    for uid in uids:
        results.append([uid, topk_users_pre_iids[uid], topk_users_pre_scores[uid], users_truth_iids[uid], users_truth_scores[uid]])
    results = pd.DataFrame(results, columns=['uid', 'pre_iids', 'pre_scores', 'truth_iids', 'truth_scores'])
    results.to_csv(path_file, index=0)


def read_predict_data(path_file):
    results = pd.read_csv(path_file)
    for uid, pre_iids, pre_scores, truth_iids, truth_scores in zip(results.uids, results.pre_iids, results.pre_scores, results.truth_iids, results.truth_scores):
        re= str(uid)+"--"+str(pre_iids)+"--"+str(pre_scores)+"--"+str(truth_iids)+"--"+str(truth_scores)
        print(re)


def find_target_att_file(file_dir, condition):
    target_file_names = []
    for file_name in os.listdir(file_dir):
        if file_name.find(condition) != -1:
            target_file_names.append(file_name)
    target_file_names = sorted(target_file_names, key=lambda x: x, reverse=True)

    return target_file_names


def get_path_resoning_hops(str1):
    str = copy.deepcopy(str1)
    start = str.find('[')
    end = str.find(']')
    res = str[start:end + 1]
    return res


def save_eval_data(path_file, model_eval_data):
    results = []
    for name in list(model_eval_data.keys()):
        evals = model_eval_data[name]
        results.append([name, evals['NDCG'], evals['Recall'],  evals['HR'], evals['Precision'], evals['AUC'], evals['ACC'], evals['F1_Score']])
    results = pd.DataFrame(results, columns=['Name', 'NDCG', 'Recall', 'HR', 'Precision', 'AUC', 'ACC', 'F1_Score'])
    results.to_csv(path_file, index=0)


def read_eval_data(path_file):
    results = pd.read_csv(path_file)
    model_evals = {name: {} for name in list(results['Name'])}
    for name,  ndcg, recall, hr, precision, auc, acc, f1_Score in zip(results.Name, results.NDCG, results.Recall, results.HR, results.Precision, results.AUC,  results.ACC, results.F1_Score):
        model_evals[name]['NDCG'] = ndcg
        model_evals[name]['Recall'] = recall
        model_evals[name]['HR'] = hr
        model_evals[name]['Precision'] = precision
        model_evals[name]['AUC'] = auc
        model_evals[name]['ACC'] = acc
        model_evals[name]['F1_Score'] = f1_Score
    return model_evals


def keep_decimal(ndcg, recal, hit, precision, auc, acc, f1_score, number):
    ndcg = round(ndcg, number)
    recal = round(recal, number)
    hit = round(hit, number)
    precision = round(precision, number)
    auc = round(auc, number)
    acc = round(acc, number)
    f1_score = round(f1_score, number)
    return ndcg, recal, hit, precision, auc, acc, f1_score

