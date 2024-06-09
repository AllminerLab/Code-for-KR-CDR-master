from data_preprocess import main as data_preprocess_main
from train_meta import main as train_meta_main
from train_agent import  main as train_agent_main
from path_reasoning import main as path_reasoning_main
from train_2att_2mlp import main as train_2att_2mlp_main
from test_2att_2mlp import main as test_2att_2mlp_main


# 数据预处理
data_preprocess_main()
# 训练元网络
train_meta_main()
# 训练智能体
train_agent_main()
# 路径推理
# 需更改is_train_task参数，一次为True, 一次为False
path_reasoning_main()
# 训练注意力网络
for iter_time in range(0, 5):
    print("-------------------The time of training: {}----------------------".format(iter_time + 1))
    train_2att_2mlp_main()
# 测试
test_2att_2mlp_main()