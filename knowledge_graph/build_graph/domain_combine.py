import pandas as pd
from gevent import monkey

monkey.patch_all()
import gevent
from gevent.pool import Pool

import os
import sys
import gzip
import json
import pickle

import numpy as np
from tqdm import tqdm


# noinspection PyUnresolvedReferences
class DomainCombine(object):

    def __init__(self, source_domain_path: str, target_domain_path: str, save_path: str,
                 emb_combine_method: str = 'target',
                 source_domain_kg_entity_embedding_save_path: str = None,
                 source_domain_kg_relation_embedding_save_path: str = None,
                 target_domain_kg_entity_embedding_save_path: str = None,
                 target_domain_kg_relation_embedding_save_path: str = None):
        """
        :param source_domain_kg_entity_embedding_save_path:
        :param source_domain_kg_relation_embedding_save_path:
        :param target_domain_kg_entity_embedding_save_path:
        :param target_domain_kg_relation_embedding_save_path:
        :param source_domain_path: source domain path
        :param target_domain_path: target domain path
        :param save_path: save path
        """

        self.source_domain_path = source_domain_path
        self.target_domain_path = target_domain_path
        self.save_path = save_path
        self.emb_combine_method = emb_combine_method
        self.source_domain_kg_entity_embedding_save_path = source_domain_kg_entity_embedding_save_path
        self.source_domain_kg_relation_embedding_save_path = source_domain_kg_relation_embedding_save_path
        self.target_domain_kg_entity_embedding_save_path = target_domain_kg_entity_embedding_save_path
        self.target_domain_kg_relation_embedding_save_path = target_domain_kg_relation_embedding_save_path

    def _init_domain_data(self):
        if not os.path.exists(self.source_domain_path):
            raise Exception('Source domain path is not exists.')
        if not os.path.exists(self.target_domain_path):
            raise Exception('Target domain path is not exists.')
        self.source_domain = pickle.load(open(self.source_domain_path, 'rb'))
        self.target_domain = pickle.load(open(self.target_domain_path, 'rb'))
        self.source_domain_embedding = {
            'entities': np.load(self.source_domain_kg_entity_embedding_save_path),
            'relations': np.load(self.source_domain_kg_relation_embedding_save_path)
        }
        self.target_domain_embedding = {
            'entities': np.load(self.target_domain_kg_entity_embedding_save_path),
            'relations': np.load(self.target_domain_kg_relation_embedding_save_path)
        }
        self.graph = {
            'entities': {},
            'relations': {},
            'num_entities': self.source_domain['num_entities'] + self.target_domain['num_entities'],
            'num_relations': self.source_domain['num_relations'] + self.target_domain['num_relations'],
            'domain_info': {
                'source': {
                    'num_entities': self.source_domain['num_entities'],
                    'num_relations': self.source_domain['num_relations'],
                    'domain_name': self.source_domain['domain_name'],
                    'entity_start_id': 0,
                    'relation_start_id': 0
                },
                'target': {
                    'num_entities': self.target_domain['num_entities'],
                    'num_relations': self.target_domain['num_relations'],
                    'domain_name': self.target_domain['domain_name'],
                    'entity_start_id': self.source_domain['num_entities'],
                    'relation_start_id': self.source_domain['num_relations']
                }
            }
        }
        relation_types = {'PURCHASE': '0', 'MENTION': '1', 'DESCRIBED_AS': '2', 'PRODUCED_BY': '3'}
        self.graph_embedding = {
            'entities': np.zeros((self.graph['num_entities'], self.source_domain_embedding['entities'].shape[1])),
            'relations': np.zeros((len(relation_types), self.source_domain_embedding['relations'].shape[1]))
        }
        for key in relation_types.keys():
            if self.emb_combine_method == 'target':
                self.graph_embedding['relations'][int(relation_types[key])] = self.target_domain_embedding['relations'][
                    int(relation_types[key])]
            elif self.emb_combine_method == 'source':
                self.graph_embedding['relations'][int(relation_types[key])] = self.source_domain_embedding['relations'][
                    int(relation_types[key])]
            elif self.emb_combine_method == 'add':
                self.graph_embedding['relations'][int(relation_types[key])] = (
                                                                                      self.source_domain_embedding[
                                                                                          'relations'][
                                                                                          int(relation_types[key])] +
                                                                                      self.target_domain_embedding[
                                                                                          'relations'][
                                                                                          int(relation_types[key])]
                                                                              ) / 2
            else:
                self.graph_embedding['relations'][int(relation_types[key])] = (
                                                                                      self.source_domain_embedding[
                                                                                          'relations'][
                                                                                          int(relation_types[key])] +
                                                                                      self.target_domain_embedding[
                                                                                          'relations'][
                                                                                          int(relation_types[key])]
                                                                              ) / 2
        self.same_items = {'target': {}, 'source': {}, 'total_remove': 0}
        self.target_domain_entity_id_map = {}

    def _find_same_item(self):

        for entity_type in self.source_domain['entities'].keys():
            self.same_items['target'][entity_type] = {}
            self.same_items['source'][entity_type] = {}
            if entity_type != 'PRODUCT':
                source_domain_items = [entity['name'] for entity in self.source_domain['entities'][entity_type]]
                target_domain_items = [entity['name'] for entity in self.target_domain['entities'][entity_type]]
                source_domain_items = frozenset(source_domain_items)
                target_domain_items = frozenset(target_domain_items)
                same_items = source_domain_items & target_domain_items

                self.same_items['total_remove'] += len(same_items)
                self.graph['num_entities'] -= len(same_items)

                source_domain_item_id_map = {
                    entity['name']: entity['id']
                    for entity in self.source_domain['entities'][entity_type]
                }
                target_domain_item_id_map = {
                    entity['name']: entity['id']
                    for entity in self.target_domain['entities'][entity_type]
                }

                for item_name in same_items:
                    source_id = source_domain_item_id_map[item_name]
                    target_id = target_domain_item_id_map[item_name]
                    self.same_items['target'][entity_type][target_id] = source_id
                    self.same_items['source'][entity_type][source_id] = target_id

            print(' - {} same {} found.'.format(entity_type, len(self.same_items['target'][entity_type]), entity_type))

    def _combine_entities(self):
        target_domain_entities = {}
        for key in tqdm(
                self.target_domain['entities'].keys(),
                desc=' - Updating target domain entities id: ',
                file=sys.stdout,
                position=0
        ):
            target_domain_entities[key] = []
            target_new_id = self.source_domain['num_entities']

            for entity in self.target_domain['entities'][key]:

                if entity['id'] in self.same_items['target'][key]:
                    self.target_domain_entity_id_map[entity['id']] = self.same_items['target'][key][entity['id']]
                    continue

                # entity_id = entity['id'] + self.source_domain['num_entities'] - self.same_items['total_remove']
                entity_id = target_new_id
                target_new_id += 1

                target_domain_entities[key].append({
                    'id': entity_id, 'name': entity['name'], 'domain': self.target_domain['domain_name']
                })
                self.graph_embedding['entities'][entity_id] = self.target_domain_embedding['entities'][entity['id']]
                self.target_domain_entity_id_map[entity['id']] = entity_id

                if entity_id >= self.graph['num_entities']:
                    raise ValueError('entity id should not be greater than {}'.format(self.graph['num_entities']))

        for key in target_domain_entities.keys():
            for i in range(len(self.source_domain['entities'][key])):
                self.source_domain['entities'][key][i]['domain'] = self.source_domain['domain_name']
                if self.source_domain['entities'][key][i]['id'] in self.same_items['source'][key]:
                    self.source_domain['entities'][key][i]['domain'] = 'same'

                # 2024 need modification
                if self.emb_combine_method == 'target' and self.source_domain['entities'][key][i]['domain'] == 'same':
                    self.graph_embedding['entities'][self.source_domain['entities'][key][i]['id']] = \
                        self.target_domain_embedding['entities'][
                            self.same_items['source'][key][self.source_domain['entities'][key][i]['id']]]
                elif self.emb_combine_method == 'add' and self.source_domain['entities'][key][i]['domain'] == 'same':
                    self.graph_embedding['entities'][self.source_domain['entities'][key][i]['id']] += \
                        self.source_domain_embedding['entities'][self.source_domain['entities'][key][i]['id']]
                    self.graph_embedding['entities'][self.source_domain['entities'][key][i]['id']] /= 2
                elif self.emb_combine_method == 'add-no-user' and self.source_domain['entities'][key][i][
                    'domain'] == 'same':
                    if key == 'USER':
                        self.graph_embedding['entities'][self.source_domain['entities'][key][i]['id']] = \
                            self.source_domain_embedding['entities'][self.source_domain['entities'][key][i]['id']]
                    else:
                        self.graph_embedding['entities'][self.source_domain['entities'][key][i]['id']] += \
                            self.source_domain_embedding['entities'][self.source_domain['entities'][key][i]['id']]
                        self.graph_embedding['entities'][self.source_domain['entities'][key][i]['id']] /= 2
                else:
                    self.graph_embedding['entities'][self.source_domain['entities'][key][i]['id']] = \
                        self.source_domain_embedding['entities'][self.source_domain['entities'][key][i]['id']]

        self.graph['entities'] = self.source_domain['entities']
        for key in tqdm(
                self.target_domain['entities'].keys(),
                desc=' - Combining entities: ',
                file=sys.stdout,
                position=0
        ):
            print('\n - {} {} entities in source domain.'.format(len(self.source_domain['entities'][key]), key))
            print(' - {} {} entities in target domain'.format(len(self.target_domain['entities'][key]), key))
            print(' - {} {} same entities found.'.format(len(self.same_items['target'][key]), key))
            self.graph['entities'][key].extend(target_domain_entities[key])
            print(' - {} {} entities.'.format(len(self.graph['entities'][key]), key))

    def _combine_relations(self):
        relation_types = {
            'PURCHASE': ['USER', 'PRODUCT'],
            'MENTION': ['USER', 'WORD'],
            'DESCRIBED_AS': ['PRODUCT', 'WORD'],
            'PRODUCED_BY': ['PRODUCT', 'BRAND'],
        }
        for key in tqdm(
                self.target_domain['relations'].keys(),
                desc=' - Updating target domain relations id: ',
                file=sys.stdout,
                position=0
        ):
            relation_type = relation_types[key]
            for i in range(len(self.target_domain['relations'][key])):
                self.target_domain['relations'][key][i]['id'] += self.source_domain['num_relations']
                start_id, start_type = self.target_domain['relations'][key][i]['start'], relation_type[0]
                end_id, end_type = self.target_domain['relations'][key][i]['end'], relation_type[1]

                self.target_domain['relations'][key][i]['start'] = self.target_domain_entity_id_map[start_id]
                self.target_domain['relations'][key][i]['end'] = self.target_domain_entity_id_map[end_id]

                if self.target_domain['relations'][key][i]['start'] >= self.graph['num_entities']:
                    raise ValueError('entity id should not be greater than {}'.format(self.graph['num_entities']))
                if self.target_domain['relations'][key][i]['end'] >= self.graph['num_entities']:
                    raise ValueError('entity id should not be greater than {}'.format(self.graph['num_entities']))

        self.graph['relations'] = self.source_domain['relations']
        for key in tqdm(
                self.target_domain['relations'].keys(),
                desc=' - Combining relations: ',
                file=sys.stdout,
                position=0
        ):
            self.graph['relations'][key].extend(self.target_domain['relations'][key])

    def update_train_and_test_data(self, test_data_path, test_neg_data_path,
                                   attention_network_train_path, meta_network_train_path):
        test_data_dict = pd.read_pickle(test_data_path)
        test_neg_data_dict = pd.read_pickle(test_neg_data_path)
        meta_train_data = pd.read_pickle(meta_network_train_path)
        attention_train_data = pd.read_pickle(attention_network_train_path)

        new_test_data_dict = {}
        new_test_neg_data_dict = {}
        new_meta_train_data = {}
        new_attention_train_data = {}

        for user in test_data_dict.keys():
            new_user = self.target_domain_entity_id_map[user]
            if new_user not in new_test_data_dict:
                new_test_data_dict[new_user] = []
            for item in test_data_dict[user]:
                new_item_id = self.target_domain_entity_id_map[item['product']]
                new_test_data_dict[new_user].append({'product': new_item_id, 'score': item['score']})

        for user in test_neg_data_dict.keys():
            new_user = self.target_domain_entity_id_map[user]
            if new_user not in new_test_neg_data_dict:
                new_test_neg_data_dict[new_user] = []
            for item in test_neg_data_dict[user]:
                new_item_id = self.target_domain_entity_id_map[item['product']]
                new_test_neg_data_dict[new_user].append({'product': new_item_id, 'score': item['score']})

        for user in meta_train_data:
            new_user = self.target_domain_entity_id_map[user]
            if new_user not in new_meta_train_data:
                new_meta_train_data[new_user] = {'positive': [], 'negative': []}
            for item in meta_train_data[user]['positive']:
                new_item_id = self.target_domain_entity_id_map[item['product']]
                new_meta_train_data[new_user]['positive'].append({'product': new_item_id, 'score': item['score']})
            for item in meta_train_data[user]['negative']:
                new_item_id = self.target_domain_entity_id_map[item['product']]
                new_meta_train_data[new_user]['negative'].append({'product': new_item_id, 'score': item['score']})

        for user in attention_train_data:
            new_user = self.target_domain_entity_id_map[user]
            if new_user not in new_attention_train_data:
                new_attention_train_data[new_user] = {'positive': [], 'negative': []}
            for item in attention_train_data[user]['positive']:
                new_item_id = self.target_domain_entity_id_map[item['product']]
                new_attention_train_data[new_user]['positive'].append({'product': new_item_id, 'score': item['score']})
            for item in attention_train_data[user]['negative']:
                new_item_id = self.target_domain_entity_id_map[item['product']]
                new_attention_train_data[new_user]['negative'].append({'product': new_item_id, 'score': item['score']})
        pickle.dump(new_test_data_dict, open(test_data_path, 'wb'))
        pickle.dump(new_test_neg_data_dict, open(test_neg_data_path, 'wb'))
        pickle.dump(new_meta_train_data, open(meta_network_train_path, 'wb'))
        pickle.dump(new_attention_train_data, open(attention_network_train_path, 'wb'))

    def combine(self):
        # if not os.path.exists(self.save_path):
        #     os.makedirs(self.save_path)

        print('Loading domain data...')
        self._init_domain_data()
        print('Loading domain data done.')

        print('Find same item...')
        self._find_same_item()
        print('Find same item done.')

        print('Combine entities...')
        self._combine_entities()
        print('Combine entities done. Entities num: {}'.format(self.graph['num_entities']))

        print('Combine relations...')
        self._combine_relations()
        print('Combine relations done. Relations num: {}'.format(self.graph['num_relations']))

        print('Saving graph...')
        pickle.dump(self.graph, open(os.path.join(self.save_path, '{}_{}-graph.pkl'.format(
            self.source_domain['domain_name'], self.target_domain['domain_name']
        )), 'wb'))
        print('Graph saved to {}'.format(os.path.join(self.save_path, '{}_{}-graph.pkl'.format(
            self.source_domain['domain_name'], self.target_domain['domain_name']
        ))))

        # remove zeros entities embedding and save with numpy
        self.graph_embedding['entities'] = self.graph_embedding['entities'][:self.graph['num_entities']]
        print('Saving graph embedding...')
        np.save(os.path.join(self.save_path, '{}_{}-{}-entities-embedding.npy'.format(
            self.source_domain['domain_name'], self.target_domain['domain_name'], self.emb_combine_method
        )), self.graph_embedding['entities'])
        np.save(os.path.join(self.save_path, '{}_{}-{}-relations-embedding.npy'.format(
            self.source_domain['domain_name'], self.target_domain['domain_name'], self.emb_combine_method
        )), self.graph_embedding['relations'])
        print('Graph embedding saved to {}, size: {}'.format(
            os.path.join(self.save_path, '{}_{}-{}-entities-embedding.npy'.format(
                self.source_domain['domain_name'], self.target_domain['domain_name'], self.emb_combine_method
            )), self.graph_embedding['entities'].shape))
        print('Graph embedding saved to {}, size: {}'.format(
            os.path.join(self.save_path, '{}_{}-{}-relations-embedding.npy'.format(
                self.source_domain['domain_name'], self.target_domain['domain_name'], self.emb_combine_method
            )), self.graph_embedding['relations'].shape))


def load_reviews_data(data_path, reviews_name):
    reviews = []
    if not os.path.exists(os.path.join(data_path, reviews_name)):
        raise FileNotFoundError('File not found: {}'.format(os.path.join(data_path, reviews_name)))
    with gzip.open(os.path.join(data_path, reviews_name), 'rb') as f:
        for line in f:
            reviews.append(json.loads(line))
    return reviews


def find_same_users(source_domain_raw_data_path, source_reviews_name, target_domain_raw_data_path, target_reviews_name):
    sorce_domain_reviews = load_reviews_data(source_domain_raw_data_path, source_reviews_name)
    target_domain_reviews = load_reviews_data(target_domain_raw_data_path, target_reviews_name)
    source_domain_users = frozenset(
        [review['reviewerID'] for review in tqdm(sorce_domain_reviews, desc=' - Get source users: ')])
    target_domain_users = frozenset(
        [review['reviewerID'] for review in tqdm(target_domain_reviews, desc=' - Get target users: ')])
    same_users = list(source_domain_users & target_domain_users)
    return same_users
