#参数
from stagetwo.Reward import PromptedRelationExtractReward
from stagetwo.Train import train_agent

base_path = "../data"
dataset = "triple"
num_episodes=100#训练轮数

# 生成参数
num_repeats =1
num_samples: int = 1
num_bootstraps: int = 1
compute_zscore: bool = True

# 奖励权重参数
exact_weight: float = 0.7
fuzzy_weight: float = 0.3

# similarity_model: str = "/hy-tmp/cui/mpnet_base_v2"
# token_model="/hy-tmp/cui/opprompt/bert_base_chinese"
mode="train"

similarity_model: str = "sentence-transformers/all-mpnet-base-v2"
token_model="google-bert/bert-base-chinese"


#创建类对象
from stageone.data_loder.dataset import make_relation_extract_dataset
from stagetwo.Agent import REINFORCE,StateModel


if __name__ == '__main__':
    agent = REINFORCE()
    train_dataset, val_dataset, test_dataset = make_relation_extract_dataset(base_path, dataset)
    reward=PromptedRelationExtractReward(similarity_model,num_repeats,num_samples,num_bootstraps,compute_zscore,
                exact_weight,
                fuzzy_weight
                )
    state_model=StateModel(token_model)

    #训练
    train_agent(train_dataset, val_dataset, test_dataset,num_episodes,agent, reward,state_model,mode)