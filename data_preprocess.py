
from knowledge_graph.domain_construction import SingleDomainToGraph, find_same_users
from knowledge_graph.sample_generator import SampleGenerator
from utils import *
from knowledge_graph.build_graph.domain_combine import DomainCombine

source_domain_raw_data_path = './data/raw'
target_domain_raw_data_path = './data/raw'

source_reviews_name = 'FASHION.json.gz'
target_reviews_name = 'Software.json.gz'

source_metadata_name = 'meta_FASHION.json.gz'
target_metadata_name = 'meta_Software.json.gz'

source_domain_name = 'FASHION'
target_domain_name = 'Software'

source_domain_save_path = './data/FASHION_Software/kg/FASHION_graph.pkl'
target_domain_tmp_save_path = './data/FASHION_Software/kg/Software_not_prune_graph.pkl'
target_domain_save_path = './data/FASHION_Software/kg/Software_graph.pkl'

test_data_path = './data/FASHION_Software/test_data.pkl'
test_neg_data_path = './data/FASHION_Software/test_negative_data.pkl'
attention_network_train_path = './data/FASHION_Software/attention_network_train_data.pkl'
meta_network_train_path = './data/FASHION_Software/meta_network_train_data.pkl'

kg_dglke_dir = './data/FASHION_Software/dglke'

kg_save_path = './data/FASHION_Software/kg'

# 嵌入文件
source_domain_kg_entity_embedding_save_path = './data/FASHION_Software/dglke/FASHION_128/TransE_own_kg_0/own_kg_TransE_entity.npy'
source_domain_kg_relation_embedding_save_path = './data/FASHION_Software/dglke/FASHION_128/TransE_own_kg_0/own_kg_TransE_relation.npy'
target_domain_kg_entity_embedding_save_path = './data/FASHION_Software/dglke/Software_128/TransE_own_kg_0/own_kg_TransE_entity.npy'
target_domain_kg_relation_embedding_save_path = './data/FASHION_Software/dglke/Software_128/TransE_own_kg_0/own_kg_TransE_relation.npy'

def domain_construction():
    print('Find same users...')
    same_users = find_same_users(
        source_domain_raw_data_path=source_domain_raw_data_path,
        source_reviews_name=source_reviews_name,
        target_domain_raw_data_path=target_domain_raw_data_path,
        target_reviews_name=target_reviews_name
    )
    print('Find same users done.')

    print('Same users num: {}'.format(len(same_users)))
    print('Source domain ({}) data processing...'.format(source_domain_name))
    extractor = SingleDomainToGraph(
        raw_data_root=source_domain_raw_data_path,
        reviews_name=source_reviews_name,
        metadata_name=source_metadata_name,
        domain_name=source_domain_name,
        save_path=source_domain_save_path,
        same_users=same_users,
        domain_type='source domain'
    )
    extractor.extract()
    print('Source domain data processing done.')
    print('Target domain ({}) data processing...'.format(target_domain_name))
    extractor = SingleDomainToGraph(
        raw_data_root=target_domain_raw_data_path,
        reviews_name=target_reviews_name,
        metadata_name=target_metadata_name,
        domain_name=target_domain_name,
        save_path=target_domain_tmp_save_path,
        same_users=same_users,
        domain_type='target domain'
    )
    extractor.extract()
    print('Target domain data processing done.')


# 样本生成
def sample_generator():
    generator = SampleGenerator(source_domain_graph_path=source_domain_save_path,
                                target_domain_graph_path=target_domain_tmp_save_path)

    # 生成测试集和训练集
    generator.generate_sample(not_prune_graph=generator.target_domain,
                              test_ratio=0.2,
                              test_data_path=test_data_path,
                              test_neg_data_path=test_neg_data_path,
                              meta_network_train_path=meta_network_train_path,
                              attention_network_train_path=attention_network_train_path)
    # 更新目标域知识图谱，剔除测试集的信息，方便后面训练
    target_domain = generator.update_target_domain_graph(target_domain_save_path=target_domain_save_path)
    # 生成目标域知识图谱嵌入训练所需要的文件
    generator.export_data_fro_dglke(kg_dglke_dir=kg_dglke_dir, domain_type='target_domain', graph=target_domain, name=target_domain_name)
    # 生成源域知识图谱嵌入训练所需要的文件
    source_domain = generator.get_source_domain()
    generator.export_data_fro_dglke(kg_dglke_dir=kg_dglke_dir, domain_type='source_domain', graph=source_domain, name=source_domain_name)

    # 安装并使用dglke训练嵌入--绝对路径训练
    #dglke_train --model_name TransE --data_path  ***/data/kg/FASHION_Software/dglke --data_files FASHION_entities.dict FASHION_relations.dict FASHION_train.tsv --format udd_hrt --dataset own_kg --save_path ***/data/kg/FASHION_Software/dglke/FASHION_128 --batch_size 1000 --log_interval 1000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 128 --gamma 19.9 --lr 0.25 -adv --gpu 0 --max_step 50000 --delimiter ,

    #dglke_train --model_name TransE --data_path  ***/data/kg/FASHION_Software/dglke --data_files Software_entities.dict Software_relations.dict Software_train.tsv --format udd_hrt --dataset own_kg --save_path ***/data/kg/FASHION_Software/dglke/Software_128 --batch_size 1000 --log_interval 1000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 128 --gamma 19.9 --lr 0.25 -adv --gpu 0 --max_step 50000 --delimiter ,



def main():
    # 知识图谱构建
    domain_construction()
    # 样本生成
    sample_generator()
    # 合并知识图谱

    domain_combine = DomainCombine(
        source_domain_path=source_domain_save_path,
        target_domain_path=target_domain_save_path,
        source_domain_kg_entity_embedding_save_path=source_domain_kg_entity_embedding_save_path,
        source_domain_kg_relation_embedding_save_path=source_domain_kg_relation_embedding_save_path,
        target_domain_kg_entity_embedding_save_path=target_domain_kg_entity_embedding_save_path,
        target_domain_kg_relation_embedding_save_path=target_domain_kg_relation_embedding_save_path,
        save_path=kg_save_path,
        emb_combine_method='target'
    )
    domain_combine.combine()
    # 更新测试集和训练集的ID
    domain_combine.update_train_and_test_data(test_data_path=test_data_path,
                                              test_neg_data_path=test_neg_data_path,
                                              meta_network_train_path=meta_network_train_path,
                                              attention_network_train_path=attention_network_train_path)

if __name__ == '__main__':
    main()

