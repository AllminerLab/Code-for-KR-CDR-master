from gevent import monkey
monkey.patch_all()
import gevent
from gevent.pool import Pool

import os
import sys
import gzip
import json
import pickle
import argparse
from tqdm import tqdm
from keybert import KeyBERT


class SingleDomainToGraph(object):

    def __init__(self, raw_data_root: str, reviews_name: str, metadata_name: str, domain_name: str, save_path: str, train_ratio: float = 0.8):
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

    def _init_domain_data(self):
        self.reviews, self.metadata = self._load_raw_data()
        self.graph = {
            'entities': {
                'USER': [],
                'PRODUCT': [],
                'CATEGORY': [],
                'BRAND': [],
                'WORD': []
            },
            'relations': {
                'PURCHASE': [],
                'MENTION': [],
                'DESCRIBED_AS': [],
                'ALSO_BOUGHT': [],
                'ALSO_VIEWED': [],
                'BELONG_TO': [],
                'PRODUCED_BY': [],
            },
            'num_entities': 0,
            'num_relations': 0,
            'domain_name': self.domain_name
        }
        self.tmp_graph = {
            'entities': {},
            'relations': {}
        }
        self.train_data = self.reviews[:int(len(self.reviews) * self.train_ratio)]
        self.test_data = self.reviews[int(len(self.reviews) * self.train_ratio):]

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

        key_bert_model = KeyBERT()
        entity_list = {'USER': [], 'PRODUCT': [], 'WORD': [], 'CATEGORY': [], 'BRAND': []}

        def _extract_keywords(review: str) -> list:
            return key_bert_model.extract_keywords(
                review,
                keyphrase_ngram_range=(1, 1),
                top_n=int(len(review) * 0.1),
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
                desc=' - Extracting CATEGORY, BRAND entities: ',
                file=sys.stdout,
                position=0
        ):
            theads.append(pool.spawn(_extract_entity, 'PRODUCT', data['asin']))
            also_buy_products = data.get('also_buy', [])
            for product in also_buy_products:
                theads.append(pool.spawn(_extract_entity, 'PRODUCT', product))
            also_view_products = data.get('also_view', [])
            for product in also_view_products:
                theads.append(pool.spawn(_extract_entity, 'PRODUCT', product))
            categories = data.get('category', [])
            for category in categories:
                theads.append(pool.spawn(_extract_entity, 'CATEGORY', category))
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

        pool = Pool(1000)
        theads = []
        for data in tqdm(
                self.train_data,
                desc=' - Extracting PURCHASE, MENTION, DESCRIBED_AS relations: ',
                file=sys.stdout,
                position=0
        ):
            start = {'type': 'USER', 'id': data['reviewerID']}
            end = {'type': 'PRODUCT', 'id': data['asin']}
            theads.append(pool.spawn(_extrac_relation, 'PURCHASE', start, end, data['overall']))

            if 'reviewText' in data:
                for keywords in data['reviewText']:
                    theads.append(pool.spawn(_extrac_relation, 'MENTION', start, {'type': 'WORD', 'id': keywords[0]}, keywords[1]))
                    theads.append(pool.spawn(_extrac_relation, 'DESCRIBED_AS', end, {'type': 'WORD', 'id': keywords[0]}, keywords[1]))
        gevent.joinall(theads)

        theads = []
        for data in tqdm(
                self.metadata,
                desc=' - Extracting BELONG_TO, PRODUCED_BY relations: ',
                file=sys.stdout,
                position=0
        ):
            start = {'type': 'PRODUCT', 'id': data['asin']}
            also_buy_products = data.get('also_buy', [])
            for product in also_buy_products:
                theads.append(pool.spawn(_extrac_relation, 'ALSO_BOUGHT', start, {'type': 'PRODUCT', 'id': product}))
            also_view_products = data.get('also_view', [])
            for product in also_view_products:
                theads.append(pool.spawn(_extrac_relation, 'ALSO_VIEWED', start, {'type': 'PRODUCT', 'id': product}))
            brand = data.get('brand', '')
            if brand:
                theads.append(pool.spawn(_extrac_relation, 'PRODUCED_BY', start, {'type': 'BRAND', 'id': brand}))

    def _extract_raw_train_and_test_data(self) -> (list, list):
        raw_train_data = [
            {'user_id': data['reviewerID'], 'product_id': data['asin'], 'overall': data['overall']}
            for data in self.train_data
        ]
        raw_test_data = [
            {'user_id': data['reviewerID'], 'product_id': data['asin'], 'overall': data['overall']}
            for data in self.test_data
        ]
        return raw_train_data, raw_test_data

    def extract(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        print('Loading raw data from {} ...'.format(self.raw_data_root))
        self._init_domain_data()
        print('Loading raw data done.')

        print('Extracting entities...')
        self._extract_entities()
        print('Extracting entities done.')

        print('Extracting relations...')
        self._extract_relations()
        print('Extracting relations done.')

        print('Saving graph, train data and test data...')
        print('Graph size: {} entities, {} relations'.format(self.graph['num_entities'], self.graph['num_relations']))
        print('Raw train data size: {}'.format(len(self.train_data)))
        print('Raw test data size: {}'.format(len(self.test_data)))
        pickle.dump(self.graph, open(os.path.join(self.save_path, 'amazon_beauty-amazon_appliances-graph.pkl'), 'wb'))
        raw_train_data, raw_test_data = self._extract_raw_train_and_test_data()
        pickle.dump(raw_train_data, open(os.path.join(self.save_path, 'raw_train.pkl'), 'wb'))
        pickle.dump(raw_test_data, open(os.path.join(self.save_path, 'raw_test.pkl'), 'wb'))
        print('Graph saved to {}.'.format(os.path.join(self.save_path, 'amazon_beauty-amazon_appliances-graph.pkl')))
        print('Raw train data saved to {}.'.format(os.path.join(self.save_path, 'raw_train.pkl')))
        print('Raw test data saved to {}.'.format(os.path.join(self.save_path, 'raw_test.pkl')))


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_root', type=str)
    parser.add_argument('--reviews_name', type=str)
    parser.add_argument('--metadata_name', type=str)
    parser.add_argument('--domain_name', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    return parser.parse_args()


if __name__ == '__main__':
    args = make_args()

    extractor = SingleDomainToGraph(
        raw_data_root=args.raw_data_root,
        reviews_name=args.reviews_name,
        metadata_name=args.metadata_name,
        domain_name=args.domain_name,
        save_path=args.save_path,
        train_ratio=args.train_ratio
    )
    extractor.extract()

# python single_domain_to_graph.py --raw_data_root ../data/raw/amazon_beauty --reviews_name All_Beauty.json.gz --metadata_name meta_All_Beauty.json.gz --domain_name amazon_beauty --save_path ../data/processed/amazon_beauty
# python single_domain_to_graph.py --raw_data_root ../data/raw/amazon_appliances --reviews_name Appliances.json.gz --metadata_name meta_Appliances.json.gz --domain_name amazon_appliances --save_path ../data/processed/amazon_appliances
# python single_domain_to_graph.py --raw_data_root ../data/raw/amazon_prime_pantry --reviews_name Prime_Pantry.json.gz --metadata_name meta_Prime_Pantry.json.gz --domain_name amazon_prime_pantry --save_path ../data/processed/amazon_prime_pantry