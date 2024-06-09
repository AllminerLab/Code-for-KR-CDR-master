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
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


class SingleDomainToGraph(object):

    def __init__(self,
                 raw_data_root: str,
                 reviews_name: str,
                 metadata_name: str,
                 domain_name: str,
                 save_path: str,
                 train_ratio: float = 0.6,
                 user_threshold: int = 3,
                 same_users=None,
                 domain_type: str = None):
        """
        :param raw_data_root: the root path of raw data
        :param reviews_name: the name of reviews file
        :param metadata_name: the name of metadata file
        :param domain_name: the name of domain
        :param save_path: the path to save the extracted graph
        :param train_ratio: the ratio of training data
        """
        self.raw_data_root = raw_data_root
        self.reviews_name = reviews_name
        self.metadata_name = metadata_name
        self.domain_name = domain_name
        self.save_path = save_path
        self.train_ratio = train_ratio
        self.user_threshold = 3
        self.same_users = same_users
        self.domain_type = domain_type

    def _init_domain_data(self):
        self.reviews, self.metadata = self._load_raw_data()
        self.graph = {
            'entities': {
                'USER': [],
                'PRODUCT': [],
                'BRAND': [],
                'WORD': []
            },
            'relations': {
                'PURCHASE': [],
                'MENTION': [],
                'DESCRIBED_AS': [],
                'PRODUCED_BY': []
            },
            'num_entities': 0,
            'num_relations': 0,
            'num_user_item_inters': 0,
            'domain_name': self.domain_name,
            'domain_type': self.domain_type
        }
        self.tmp_graph = {
            'entities': {},
            'relations': {}
        }
        self.train_data = self.reviews[:int(len(self.reviews) * self.train_ratio)]

        self.user_dict = {}
        for i in tqdm(
                range(len(self.train_data)),
                desc=' - Remove users with less than {} reviews'.format(self.user_threshold),
                file=sys.stdout,
                position=0
        ):
            # noinspection PyTypeChecker
            user_id = self.train_data[i]['reviewerID']
            if user_id not in self.user_dict:
                self.user_dict[user_id] = 1
            else:
                self.user_dict[user_id] += 1

    def _load_raw_data(self) -> (list, list):
        reviews, metadata = [], []
        for file_name in [self.reviews_name, self.metadata_name]:
            if not os.path.exists(os.path.join(self.raw_data_root, file_name)):
                raise FileNotFoundError('File not found: {}'.format(file_name))
            with gzip.open(os.path.join(self.raw_data_root, file_name), 'rb') as f:
                for line in f:
                    if file_name == self.reviews_name:
                        reviews.append(json.loads(line))
                    else:
                        metadata.append(json.loads(line))
        return reviews, metadata

    # noinspection PyTypeChecker
    def _extract_entities(self):

        # key_bert_model = KeyBERT()
        # 需下载 all-MiniLM-L6-v2
        keybert_model_path = './data/raw/keybert/all-MiniLM-L6-v2'
        kw_model = SentenceTransformer(keybert_model_path)
        key_bert_model = KeyBERT(model=kw_model)
        entity_list = {'USER': [], 'PRODUCT': [], 'WORD': [], 'BRAND': []}

        def _extract_keywords(review: str) -> list:
            return key_bert_model.extract_keywords(
                review,
                keyphrase_ngram_range=(1, 1),
                top_n=5,
                stop_words='english',
                use_mmr=True, diversity=0.7
            )

        def _extract_entity(entity_type: str, entity_id: str):
            entity_list[entity_type].append(entity_id)

        pool = Pool(1000)
        theads = []
        for i in tqdm(
                range(len(self.train_data)),
                desc=' - Extracting USER, PRODUCT, WORD entities: ',
                file=sys.stdout,
                position=0
        ):
            data = self.train_data[i]
            if data['reviewerID'] in self.same_users or self.user_dict[data['reviewerID']] >= self.user_threshold:
                theads.append(pool.spawn(_extract_entity, 'USER', data['reviewerID']))
                theads.append(pool.spawn(_extract_entity, 'PRODUCT', data['asin']))
                if 'reviewText' in data:
                    keywords = _extract_keywords(data['reviewText'])
                    self.train_data[i]['reviewText'] = keywords
                    for keyword in keywords:
                        theads.append(pool.spawn(_extract_entity, 'WORD', keyword[0]))

        gevent.joinall(theads)

        theads = []
        for data in tqdm(
                self.metadata,
                desc=' - Extracting BRAND entities: ',
                file=sys.stdout,
                position=0
        ):
            if data['asin'] in entity_list['PRODUCT']:
                theads.append(pool.spawn(_extract_entity, 'PRODUCT', data['asin']))
                brand = data.get('brand', '')
                if brand:
                    theads.append(pool.spawn(_extract_entity, 'BRAND', brand))
        gevent.joinall(theads)

        for key in entity_list.keys():
            entity_list[key] = list(set(entity_list[key]))
            for name in entity_list[key]:
                tmp_key = key + '-' + name
                entity_info = {'id': self.graph['num_entities'], 'name': name}
                self.graph['num_entities'] += 1
                self.tmp_graph['entities'][tmp_key] = entity_info
                self.graph['entities'][key].append(entity_info)

    def _extract_relations(self):

        # noinspection PyShadowingNames
        def _extrac_relation(relation_type: str, start: dict, end: dict, weight=0.0):
            start_id = self.tmp_graph['entities'][start['type'] + '-' + start['id']]['id']
            end_id = self.tmp_graph['entities'][end['type'] + '-' + end['id']]['id']
            self.graph['relations'][relation_type].append({
                'id': self.graph['num_relations'],
                'start': start_id,
                'end': end_id,
                'weight': weight
            })
            self.graph['num_relations'] += 1
            if relation_type == 'PURCHASE':
                self.graph['num_user_item_inters'] += 1

        pool = Pool(1000)
        theads = []
        for data in tqdm(
                self.train_data,
                desc=' - Extracting PURCHASE, MENTION, DESCRIBED_AS relations: ',
                file=sys.stdout,
                position=0
        ):
            if data['reviewerID'] in self.same_users or self.user_dict[data['reviewerID']] >= self.user_threshold:
                start = {'type': 'USER', 'id': data['reviewerID']}
                end = {'type': 'PRODUCT', 'id': data['asin']}
                theads.append(pool.spawn(_extrac_relation, 'PURCHASE', start, end, data['overall']))

                if 'reviewText' in data:
                    for keywords in data['reviewText']:
                        theads.append(
                            pool.spawn(_extrac_relation, 'MENTION', start, {'type': 'WORD', 'id': keywords[0]},
                                       keywords[1]))
                        theads.append(
                            pool.spawn(_extrac_relation, 'DESCRIBED_AS', end, {'type': 'WORD', 'id': keywords[0]},
                                       keywords[1]))
        gevent.joinall(theads)

        theads = []
        for data in tqdm(
                self.metadata,
                desc=' - Extracting PRODUCED_BY relations: ',
                file=sys.stdout,
                position=0
        ):
            if 'PRODUCT' + '-' + data['asin'] in self.tmp_graph['entities']:
                start = {'type': 'PRODUCT', 'id': data['asin']}
                brand = data.get('brand', '')
                if brand:
                    theads.append(pool.spawn(_extrac_relation, 'PRODUCED_BY', start, {'type': 'BRAND', 'id': brand}))


    def extract(self):
        print('Loading raw data from {} ...'.format(self.raw_data_root))
        self._init_domain_data()
        print('Loading raw data done.')

        print('Extracting entities...')
        self._extract_entities()
        print('Extracting entities done.')

        print('Extracting relations...')
        self._extract_relations()
        print('Extracting relations done.')

        print('Saving graph...')
        print('Graph size: {} entities, {} relations, {} user-item-interations'.format(
            self.graph['num_entities'], self.graph['num_relations'], self.graph['num_user_item_inters']))
        # print('Raw train data size: {}'.format(len(self.train_data)))
        # print('Raw test data size: {}'.format(len(self.test_data)))
        pickle.dump(self.graph, open(self.save_path, 'wb'))
        # raw_train_data, raw_test_data = self._extract_raw_train_and_test_data()
        # pickle.dump(raw_train_data, open(os.path.join(self.save_path, 'raw_train.pkl'), 'wb'))
        # pickle.dump(raw_test_data, open(os.path.join(self.save_path, 'raw_test.pkl'), 'wb'))
        print('Graph saved to {}.'.format(self.save_path))
        # print('Raw train data saved to {}.'.format(os.path.join(self.save_path, 'raw_train.pkl')))
        # print('Raw test data saved to {}.'.format(os.path.join(self.save_path, 'raw_test.pkl')))


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

