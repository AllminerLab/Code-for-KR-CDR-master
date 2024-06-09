import os
import time
import pickle
import numpy as np


class KGUtils(object):

    def __init__(self, graph_path: str, entities_embed_path: str, relations_embed_path: str):
        print('Initializing KGUtils...')
        self._load_graph_data(graph_path, entities_embed_path, relations_embed_path)
        print(' - Building adjacency list...')
        start_time = time.perf_counter()
        self.entities_id_map = self._get_entities_id_map()
        self.relations_id_map = self._get_relations_id_map()
        self.adjacency_list = self._graph_to_adjacency_list()
        print(' - Adjacency list built in {:.2f} seconds'.format(time.perf_counter() - start_time))
        print('KGUtils initialized')

    def _load_graph_data(self, graph_path: str, entities_embed_path: str, relations_embed_path: str):
        print(' - Loading graph data...')
        start_time = time.perf_counter()
        self.graph = pickle.load(open(graph_path, 'rb'))
        self.entities_embed = np.load(entities_embed_path)
        self.relations_embed = np.load(relations_embed_path)

        self.source_domain = self.graph['domain_info']['source']
        self.target_domain = self.graph['domain_info']['target']

        self.user_data = {'source': [], 'target': [], 'same': []}

        for entity in self.graph['entities']['USER']:
            if entity['domain'] == 'same':
                self.user_data['same'].append(entity['id'])
        self.user_data['same'] = list(set(self.user_data['same']))

        for entity in self.graph['entities']['USER']:
            if entity['domain'] == self.source_domain['domain_name']:
                self.user_data['source'].append(entity['id'])
            elif entity['domain'] == self.target_domain['domain_name']:
                self.user_data['target'].append(entity['id'])

        print(' - Graph data loaded in {:.2f} seconds'.format(time.perf_counter() - start_time))

    def _check_entity_id(self, entity_id: int):
        """
        Check if entity_id is valid.
        :param entity_id:
        :return:
        """
        if entity_id not in self.entities_id_map:
            raise ValueError('Entity id not found')

    def _check_relation_id(self, relation_id=None, relation_type=None):
        """
        Check if relation_id or relation_type is valid.
        :param relation_id:
        :return:
        """
        if relation_id is not None:
            if relation_id not in self.relations_id_map:
                raise ValueError('Relation id not found')
        if relation_type is not None:
            if relation_type not in self.relations_id_map:
                raise ValueError('Relation type not found')

    def _get_entities_id_map(self) -> dict:
        entities_id_map = {}
        for key in self.graph['entities'].keys():
            for entity in self.graph['entities'][key]:
                entities_id_map[entity['id']] = {
                    'id': entity['id'],
                    'name': entity['name'],
                    'type': key,
                    'domain_name': entity['domain'],
                    'embed': self.entities_embed[entity['id']]
                }
        return entities_id_map

    def _get_relations_id_map(self) -> dict:
        relations = ['PURCHASE', 'MENTION', 'DESCRIBED_AS', 'PRODUCED_BY']
        relations_id_map = {}
        for i, relation in enumerate(relations):
            relations_id_map[i] = {'id': i, 'type': relation, 'embed': self.relations_embed[i]}
            relations_id_map[relation] = {'id': i, 'type': relation, 'embed': self.relations_embed[i]}
        return relations_id_map

    def _graph_to_adjacency_list(self):
        adjacency_list = {}
        for key in self.graph['relations'].keys():
            for relation in self.graph['relations'][key]:
                start = relation['start']
                end = relation['end']
                if adjacency_list.get(start) is None:
                    adjacency_list[start] = []
                if adjacency_list.get(end) is None:
                    adjacency_list[end] = []
                adjacency_list[start].append({
                    'entity': self.entities_id_map[end],
                    'relation': self.relations_id_map[key]
                })
                adjacency_list[end].append({
                    'entity': self.entities_id_map[start],
                    'relation': self.relations_id_map[key]
                })
        return adjacency_list

    def get_entity_info(self, entity_id: int) -> dict:
        """
        Get entity information by entity id
        :param entity_id:
        :return: {'id': id, 'name': name, 'type': type, 'domain_name': domain_name, 'embed': embed}
        """
        self._check_entity_id(entity_id)
        return self.entities_id_map[entity_id]

    def get_relation_info(self, relation_id=None, relation_type=None) -> dict:
        """
        Get relation information by relation id or relation type
        :param relation_id:
        :param relation_type:
        :return: {'id': id, 'type': type, 'embed': embed}
        """
        self._check_relation_id(relation_id, relation_type)
        if relation_id is not None:
            return self.relations_id_map[relation_id]
        return self.relations_id_map[relation_type]

    def check_entity_domain(self, entity_id: int, domain='source'):
        """
        Check if an entity belongs to a domain.
        :param entity_id:
        :param domain: 'source' or 'target' or 'same'
        :return: True or False
        """
        entity_info = self.get_entity_info(entity_id)
        if entity_info['domain_name'] == self.source_domain['domain_name']:
            return domain == 'source'
        elif entity_info['domain_name'] == self.target_domain['domain_name']:
            return domain == 'target'
        return domain == 'same'

    def check_user_domain(self, user_id: int, domain='source'):
        """
        Check if a user belongs to a domain.
        :param user_id:
        :param domain: 'source' or 'target' or 'same'
        :return: True or False
        """
        if domain not in self.user_data.keys():
            raise ValueError('Domain not found')
        return user_id in self.user_data[domain]

    def get_users(self, domain='source'):
        """
        Get all users in a domain.
        :param domain: 'source' or 'target' or 'same'
        :return: list of user ids
        """
        if domain not in self.user_data.keys():
            raise ValueError('Domain not found')
        return self.user_data[domain]

    def get_entity_all_neighbors(self, entity_id: int) -> list:
        """
        Get all neighbors of an entity.
        :param entity_id:
        :return: [{'entity': entity_info, 'relation': relation_info}, ...]
        """
        self._check_entity_id(entity_id)
        if self.adjacency_list.get(entity_id) is None:
            return []
        return self.adjacency_list[entity_id]

    def get_entity_neighbors(self, entity_id: int) -> list:
        """
        Get neighbors of an entity with constraints.
        :param entity_id:
        :return: [{'entity': entity_info, 'relation': relation_info}, ...]
        """
        self._check_entity_id(entity_id)
        if self.adjacency_list.get(entity_id) is None:
            return []
        neighbors = []
        all_neighbors = self.adjacency_list[entity_id]
        entity = self.get_entity_info(entity_id)

        # if the entity belongs to the target domain
        if entity['domain_name'] == self.target_domain['domain_name']:
            for neighbor in all_neighbors:
                # if the neighbor's type is 'USER' and belongs to the 'same' domain, don't return it.
                if neighbor['entity']['type'] == 'USER':
                    if neighbor['entity']['id'] in self.user_data['same']:
                        continue
                neighbors.append(neighbor)
        else:
            # if the entity's type is 'USER' and belongs to the 'same' domain
            if entity['type'] == 'USER' and entity['domain_name'] == 'same':
                for neighbor in all_neighbors:
                    # if the neighbor belongs to target domain, don't return it.
                    if neighbor['entity']['type'] == 'PRODUCT':
                        if neighbor['entity']['domain_name'] == self.target_domain['domain_name']:
                            continue
                    neighbors.append(neighbor)
            else:
                neighbors = all_neighbors

        return neighbors

    @staticmethod
    def query_next_entity_type(entity_type: str, relation_type: str) -> str:
        """
        Query next entity type by current entity type and relation type.
        :param entity_type:
        :param relation_type:
        :return: next entity type
        """
        relation_types = {
            'PURCHASE': ['USER', 'PRODUCT'],
            'MENTION': ['USER', 'WORD'],
            'DESCRIBED_AS': ['PRODUCT', 'WORD'],
            'PRODUCED_BY': ['PRODUCT', 'BRAND']
        }
        if relation_type not in relation_types.keys():
            raise ValueError('Relation type not found')
        if entity_type not in relation_types[relation_type]:
            raise ValueError('Entity type not found')
        if relation_types[relation_type][0] == entity_type:
            return relation_types[relation_type][1]
        return relation_types[relation_type][0]

    def get_entities_by_type(self, entity_type: str) -> list:
        """
        Get all entities by type.
        :param entity_type:
        :return: list of entites info
        """
        if entity_type not in self.graph['entities'].keys():
            raise ValueError('Entity type not found')
        entities = []
        for entity in self.graph['entities'][entity_type]:
            entities.append(self.get_entity_info(entity['id']))
        return entities

