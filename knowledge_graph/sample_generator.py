
import os
import pickle
import random
import sys

import pandas as pd
from tqdm import tqdm
from utils import *

class SampleGenerator(object):

    def __init__(self, source_domain_graph_path: str, target_domain_graph_path: str):
        self.source_domain_graph_path = source_domain_graph_path
        self.target_domain_graph_path = target_domain_graph_path
        self.common_entity_dict = {}
        self.common_entity_name_dict = {}

        # 测试集
        self.test_user_list = []
        self.test_user_id_list = []
        self.test_user_names = None
        # 训练集共同用户ID
        self.train_user_id_list = []

        # 存放训练集 for personalprompt
        self.train_data_pp = []
        # 存放测试集 for personalprompt
        self.test_data_pp = []

        # 更新目标域关系
        self.tmp_relations = {
                'PURCHASE': [],
                'MENTION': [],
                'DESCRIBED_AS': [],
                'PRODUCED_BY': [],
            }
        self.relation_num = 0
        self.user_item_inters_num = 0

        self._init_domain_data()

        self.test_data_dict = {}
        self.test_neg_data_dict = {}
        self.meta_train_data = {}
        self.attention_train_data = {}


    def _init_domain_data(self):
        if not os.path.exists(self.source_domain_graph_path):
            raise Exception('Source domain path is not exists.')
        if not os.path.exists(self.target_domain_graph_path):
            raise Exception('Target domain path is not exists.')
        self.source_domain = pickle.load(open(self.source_domain_graph_path, 'rb'))
        self.target_domain = pickle.load(open(self.target_domain_graph_path, 'rb'))

    def find_common_entity(self):
        source_domain_entity_name_dict = {}
        target_domain_entity_name_dict = {}
        for entity_type in self.target_domain['entities'].keys():
            if entity_type != 'PRODUCT':
                target_domain_entity_name_dict[entity_type] = []
                for entity in self.target_domain['entities'][entity_type]:
                    target_domain_entity_name_dict[entity_type].append(entity['name'])

                source_domain_entity_name_dict[entity_type] = []
                for entity in self.source_domain['entities'][entity_type]:
                    source_domain_entity_name_dict[entity_type].append(entity['name'])

        # 寻找除项实体外的共同实体
        for entity_type in target_domain_entity_name_dict.keys():
            tmp_list = list(
                set(target_domain_entity_name_dict[entity_type]) & set(source_domain_entity_name_dict[entity_type]))
            if len(tmp_list) > 0:
                self.common_entity_name_dict[entity_type] = tmp_list

            # 找到目标域共同实体的ID
        for entity_type in self.common_entity_name_dict.keys():
            self.common_entity_dict[entity_type] = {}
            for entity in self.target_domain['entities'][entity_type]:
                if entity['name'] in self.common_entity_name_dict[entity_type]:
                    if entity['name'] not in self.common_entity_dict[entity_type]:
                        self.common_entity_dict[entity_type][entity['name']] = {'target_id': entity['id']}
                    else:
                        self.common_entity_dict[entity_type][entity['name']]['target_id'] = entity['id']

            for entity in self.source_domain['entities'][entity_type]:
                if entity['name'] in self.common_entity_name_dict[entity_type]:
                    if entity['name'] not in self.common_entity_dict[entity_type]:
                        self.common_entity_dict[entity_type][entity['name']] = {'source_id': entity['id']}
                    else:
                        self.common_entity_dict[entity_type][entity['name']]['source_id'] = entity['id']

        print('The size of common entity is {}'.format(len(list(self.common_entity_dict.keys()))))

    def generate_sample(self, not_prune_graph, test_ratio: float = 0.2,
                                         test_data_path=None, test_neg_data_path=None, meta_network_train_path=None, attention_network_train_path=None):
        print('Extracting common entity...')
        self.find_common_entity()
        print('Extracting common entity done.')

        purchase_relations = {}
        for relation in tqdm(not_prune_graph['relations']['PURCHASE'], desc=' - Purchase relations to dict: ', position=0):
            if purchase_relations.get(relation['start']) is None:
                purchase_relations[relation['start']] = []
            purchase_relations[relation['start']].append({'item': relation['end'], 'rating': relation['weight']})

        # 随机筛选目标域20%的共同用户作为测试集
        common_user_list = self.common_entity_name_dict[USER]
        self.test_user_names = random.sample(common_user_list, round(len(common_user_list) * test_ratio))

        for user_name in self.test_user_names:
            self.test_user_id_list.append(self.common_entity_dict[USER][user_name]['target_id'])

        all_items_list = [entity['id'] for entity in not_prune_graph['entities']['PRODUCT']]
        # 生成测试集，正样本+负样本
        print('Extracting test data...')
        for user in self.test_user_id_list:
            if user not in self.test_data_dict:
                self.test_data_dict[user] = []
            if user not in self.test_neg_data_dict:
                self.test_neg_data_dict[user] = []
            tmp_user_pucharse = []

            # 正样本
            for item in purchase_relations[user]:
                self.test_data_dict[user].append({'product': item['item'], 'score': item['rating']})
                tmp_user_pucharse.append(item['item'])
            # 负样本
            tmp_items = [item for item in all_items_list if item not in tmp_user_pucharse]
            test_neg_items = random.sample(tmp_items, 100)
            for item in test_neg_items:
                self.test_neg_data_dict[user].append({'product': item, 'score': 0})

        pickle.dump(self.test_data_dict, open(test_data_path, 'wb'))
        pickle.dump(self.test_neg_data_dict, open(test_neg_data_path, 'wb'))

        print('Extracting test data done.')
        # -------------------------------------------------------
        # 提取元网络训练集
        train_user_names = [user_name for user_name in self.common_entity_name_dict[USER]
                            if user_name not in self.test_user_names]
        for user_name in train_user_names:
            self.train_user_id_list.append(self.common_entity_dict[USER][user_name]['target_id'])

        train_meta_user_list = random.sample(self.train_user_id_list, round(len(common_user_list) * test_ratio))

        for user in train_meta_user_list:
            if user not in self.meta_train_data:
                self.meta_train_data[user] = {'positive': [], 'negative': []}
            tmp_user_pucharse = []
            # 正样本
            for item in purchase_relations[user]:
                self.meta_train_data[user]['positive'].append({'product': item['item'], 'score': item['rating']})
                tmp_user_pucharse.append(item['item'])
            # 负样本
            tmp_items = [item for item in all_items_list if item not in tmp_user_pucharse]
            train_meta_neg_items = random.sample(tmp_items, len(tmp_user_pucharse)*4)
            for item in train_meta_neg_items:
                self.meta_train_data[user]['negative'].append({'product': item, 'score': 0})

        pickle.dump(self.meta_train_data, open(meta_network_train_path, 'wb'))
        # -------------------------------------------------------
        # 生成推理注意力网络训练集，正样本+负样本
        print('Extracting train data..')
        for user in self.train_user_id_list:
            if user not in train_meta_user_list:
                if user not in self.attention_train_data:
                    self.attention_train_data[user] = {'positive': [], 'negative': []}
                tmp_user_pucharse = []
                # 正样本
                for item in purchase_relations[user]:
                    self.attention_train_data[user]['positive'].append({'product': item['item'], 'score': item['rating']})
                    tmp_user_pucharse.append(item['item'])
                # 负样本
                tmp_items = [item for item in all_items_list if item not in tmp_user_pucharse]
                train_att_neg_items = random.sample(tmp_items, len(tmp_user_pucharse)*4)
                for item in train_att_neg_items:
                    self.attention_train_data[user]['negative'].append({'product': item, 'score': 0})

        pickle.dump(self.attention_train_data, open(attention_network_train_path, 'wb'))
        print('Extracting train data done.')

    def update_target_domain_graph(self, target_domain_save_path=None):
        print('Updating graph...')

        def _extrac_relation(relation_type: str, start_id: int, end_id: int, weight=0.0):
            if relation_type not in self.tmp_relations: return
            self.tmp_relations[relation_type].append({
                'id': self.relation_num,
                'start': start_id,
                'end': end_id,
                'weight': weight
            })
            self.relation_num += 1
            if relation_type == 'PURCHASE':
                self.user_item_inters_num += 1

        # 更细目标域知识图谱并生成数据集
        for relation_type in tqdm(
                self.target_domain['relations'],
                desc=' - Update target domain relations: ',
                file=sys.stdout,
                position=0
        ):
            for relation in self.target_domain['relations'][relation_type]:
                if relation_type == PURCHASE:
                    if relation['start'] not in self.test_user_id_list:
                        _extrac_relation(relation_type, relation['start'], relation['end'], relation['weight'])
                else:
                    _extrac_relation(relation_type, relation['start'], relation['end'], relation['weight'])

        self.target_domain['relations'] = self.tmp_relations
        self.target_domain['num_relations'] = self.relation_num
        self.target_domain['num_user_item_inters'] = self.user_item_inters_num
        pickle.dump(self.target_domain, open(target_domain_save_path, 'wb'))
        print('Graph saved to {}.'.format(target_domain_save_path))
        return self.target_domain

    def get_source_domain(self):
        return self.source_domain

    def export_data_fro_dglke(self, kg_dglke_dir=None, domain_type=None, graph=None, name=None):
        # 为每个知识图谱构建训练嵌入需要的文件
        print('Exporting entities data...')
        entities_dict = []
        for entity_key in graph['entities'].keys():
            for entity in graph['entities'][entity_key]:
                entities_dict.append(str(entity['id']) + ',"' + str(entity['id']) + '"')
        print("entities_dict {}".format(len(entities_dict)))
        print('Exporting entities data done.')
        print('Exporting relations data...')
        relations = graph['relations']
        train_data = []
        relations_dict = ['"PURCHASE",0', '"MENTION",1', '"DESCRIBED_AS",2', '"PRODUCED_BY",3']
        relation_types = {'PURCHASE': '0', 'MENTION': '1', 'DESCRIBED_AS': '2', 'PRODUCED_BY': '3'}
        for key in tqdm(relations.keys(), desc=' - Exporting data: ', position=0):
            for relation in relations[key]:
                train_data.append(
                    str(relation['start']) + ',' + relation_types[key] + ',' + str(relation['end']))
        print('Exporting relations data done.')

        dglke_entity_dict_save_path = '/'.join([kg_dglke_dir, '{}_entities.dict'.format(name)])
        dglke_relation_dict_save_path = '/'.join([kg_dglke_dir, '{}_relations.dict'.format(name)])
        dglke_train_save_path = '/'.join([kg_dglke_dir, '{}_train.tsv'.format(name)])

        with open(dglke_train_save_path, 'w') as f:
            f.writelines([line + '\n' for line in train_data])
        with open(dglke_entity_dict_save_path, 'w') as f:
            f.writelines([line + '\n' for line in entities_dict])
        with open(dglke_relation_dict_save_path, 'w') as f:
            f.writelines([line + '\n' for line in relations_dict])

        print('train_data for kg save to {}'.format(dglke_train_save_path))
        print('entities dict for kg save to {}'.format(dglke_entity_dict_save_path))
        print('relations dict for kg save to {}'.format(dglke_relation_dict_save_path))



    def main(self):
        pass