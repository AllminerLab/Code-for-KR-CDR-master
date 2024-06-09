
# KR-CDR

# Reference:
Ling Huang, Xiao-Dong Huang, Han Zou, Yuefang Gao, Chang-Dong Wang, and Philip S. Yu.  "Knowledge-Reinforced Cross-Domain Recommendation". 2024.

# Usage
python main.py

# Requirements
1. python==3.8  
2. Requires installation of all-MiniLM-L6-v2 in './data/raw/keybert/all-MiniLM-L6-v2'.  
3. Requires download of AMAZON Dataset in './data/raw/'.    
  e.g. FASHION.json.gz  meta_FASHION.json.gz meta_Software.json.gz Software.json.gz
4. Requires TransE pre-training with dglke.   
  e.g.  (*** indicates an absolute directory)     
  (1) dglke_train --model_name TransE --data_path  ***/data/kg/FASHION_Software/dglke --data_files FASHION_entities.dict FASHION_relations.dict FASHION_train.tsv --format udd_hrt --dataset own_kg --save_path ***/data/kg/FASHION_Software/dglke/FASHION_128 --batch_size 1000 --log_interval 1000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 128 --gamma 19.9 --lr 0.25 -adv --gpu 0 --max_step 50000 --delimiter ,  
  (2) dglke_train --model_name TransE --data_path  ***/data/kg/FASHION_Software/dglke --data_files Software_entities.dict Software_relations.dict Software_train.tsv --format udd_hrt --dataset own_kg --save_path ***/data/kg/FASHION_Software/dglke/Software_128 --batch_size 1000 --log_interval 1000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 128 --gamma 19.9 --lr 0.25 -adv --gpu 0 --max_step 50000 --delimiter ,  

