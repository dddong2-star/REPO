import time

import numpy as np
import  os
import torch
import re
import itertools
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)
from stageone.data_loder.dataset import make_relation_extract_dataset

from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

from stagetwo.promptPredict import PromptPredict
import pymysql
IP = ""
MYSQLPWD = ''
DB = ''

# 生成参数
num_repeats =1
num_samples: int = 2
num_bootstraps: int = 2
compute_zscore: bool = True

# 奖励权重参数
exact_weight: float = 0.5
fuzzy_weight: float = 0.3
similarity_model: str = "/hy-tmp/cui/mpnet_base_v2"
base_path = "../data"
dataset = "triple"
class PromptedRelationExtractReward():
    def __init__(
            self,
            # 核心模型组件
            similarity_model,
            # 生成参数
            num_repeats:int =1,
            num_samples: int = 2,
            num_bootstraps: int = 2,
            compute_zscore: bool = True,
            exact_weight=0.7,
            fuzzy_weight=0.3,
            # similarity_model: str = "sentence-transformers/all-mpnet-base-v2",
            # cache_folder ='/hy-tmp/cui/opprompt/mpnet_base_v2'

    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化生成模型
        # self.tokenizer = AutoTokenizer.from_pretrained(task_lm,trust_remote_code=True)
        self.generator = PromptPredict()
        self.sim_encoder = SentenceTransformer(similarity_model)
        # 配置参数
        self.num_samples = num_samples
        self.num_bootstraps = num_bootstraps
        self.weights = {
            'exact': exact_weight,
            'fuzzy': fuzzy_weight,
        }
        # 其他参数
        self.num_repeats = num_repeats
        self.compute_zscore = compute_zscore
        self._counter = 0

    def compute_reward(
            self,
            source_texts: List[str],
            re_labels: List[str],  # 标准三元组列表
            new_prompt,
            to_tensor: bool,
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        all_f1=0
        all_p=0
        all_r=0
        all_reward=0
        all_similary=0

        num_repeats=len(source_texts)
        new_prompt= self._repeat_texts(new_prompt,num_repeats)
        rewards = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for i, (prompt, src, label) in enumerate(zip(new_prompt, source_texts, re_labels)):
            # 生成候选文本-
            hypos = self.generator.yuce_generate(
                prompt, src,
                self.num_samples * self.num_bootstraps
            )

            # print("======预测结果=======")
            # print(hypos)
            #去除引号
            def normalize_text(text):
                return re.sub(r'[“”"\']', '', text)  # 去除中文引号、英文引号
            def normalize_triples(list_triple):
                return [normalize_text(item) for item in list_triple]

            hypos=normalize_triples(hypos)
            # 三元组解析
            generated_triples =self._extract_triples(hypos)

            # 计算各维度奖励
            p,r,f1, fuzzy_scores = \
                self._compute_reward_components(src, generated_triples, label)

            # 综合奖励计算
            combined_rewards =5* f1 + 1 * fuzzy_scores[0]

            # Bootstrap处理-随机采样
            # bootstrapped_rewards = _bootstrap_rewards(combined_rewards,num_bootstraps)
            # reward = torch.mean(bootstrapped_rewards)

            # Bootstrap处理-切割列表再采样
            # bootstrap_max_rewards: List[float] = \
            #     self._boostrap_max_rewards_k_times(combined_rewards, self.num_samples)
            # reward = torch.Tensor(bootstrap_max_rewards).float().mean()
            #
            # input_rewards[src] += bootstrap_max_rewards
            #
            # # 要打印的最大值
            # max_reward = max(bootstrap_max_rewards)
            # top_index = combined_rewards.index(max_reward)
            all_p+=p
            all_r+=r
            all_f1+=f1
            all_reward+=combined_rewards
            all_similary+=fuzzy_scores[0]
            # self.save_hypos(src,label,generated_triples,p,r,f1)
            # print(generated_triples)
            # print(label)
            print(f"reward_step:{i}|"
                  f"precision: {p:.2f}| "
                  f"recall: {r:.2f} | "
                  f"f1: {f1:.2f} | "
                  f"similarity: {fuzzy_scores[0]:.2f}|"
                  f"reward_score:{combined_rewards:.2f}")
        mean_p=all_p/num_repeats
        mean_r=all_r/num_repeats
        mean_f1=all_f1/num_repeats
        mean_similary=all_similary/num_repeats
        mean_reward=all_reward/num_repeats
        # 日志记录
        quantities_to_log['mean_p'] = mean_p
        quantities_to_log['mean_r'] = mean_r
        quantities_to_log['mean_f1'] = mean_f1
        quantities_to_log['mean_similary'] = mean_similary
        quantities_to_log['mean_reward'] = mean_reward
        rewards = [mean_reward]
        # 后处理
        rewards = [torch.tensor(r, dtype=torch.float32) for r in rewards]
        rewards_tensor = torch.stack(rewards)

        # rewards_tensor = torch.stack(rewards)
        # rewards_tensor = self._compute_reward_zscores(rewards_tensor, source_texts,
        #                                              input_rewards)
        rewards_log = dict(
            (reward_key, reward_vals)
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            print( rewards_tensor, rewards_log)
            return rewards_tensor, rewards_log

        else:
            return rewards_tensor, rewards_log

    def _extract_triples(self,texts: List[str]) -> List[List[Tuple]]:
        """三元组正则匹配"""
        triples = []
        for text in texts:
            rule_triples = self._rule_based_extraction(text)
            triples.append(rule_triples)
        return triples

    def _rule_based_extraction(self,text: str) -> List[Tuple]:
        """正则表达式匹配 (实体1, 关系, 实体2) 模式"""
        found = []
        pattern = re.compile(r'[（(]\s*([^,，]+)\s*[,，]\s*([^,，]+)\s*[,，]\s*([^)）]+)\s*[）)]')
        matches = pattern.findall(text)
        for match in matches:
            found.append(tuple(match))
        return found

    def _compute_reward_components(self,src: str, generated: List[List[Tuple]],
                                   golden: List[Tuple]

                                   ) -> Tuple:
        # f1值作为评分
        p,r,f1 = self._calc_exact_scores(golden, generated)
        # 模糊匹配
        fuzzy_scores = self._calc_fuzzy_score(golden, generated, src)

        return p,r,f1, fuzzy_scores


    def _calc_exact_scores(self,golden, generated):
        if len(generated)==0:
            return 0.00,0.00,0.00
        def normalize_text(text):
            return re.sub(r'[“”"\']', '', text)  # 去除中文引号、英文引号

        def str2list(golden: str):
            # 正则表达式匹配模式
            triple_pattern = re.compile(r'（(.*?)，(.*?)，(.*?)）')
            # 解析字符串，提取所有匹配的三元组
            triples = triple_pattern.findall(golden)
            # 输出转换后的三元组列表
            return triples
        # 抽取结果为空
        golden=normalize_text(golden)
        golden_list = str2list(golden)
        # 真实标签转化为集合
        true_triplets = set(golden_list)
        # 预测标签转换为集合
        predictions = set()
        predictions.update(generated[0])
        precision, recall, f1=self.calculate_metricss(predictions,true_triplets)

        return precision, recall, f1

    def fuzzy_match(self,str1, str2):
        # 检查两个字符串是否有模糊匹配
        return bool(re.search(re.escape(str1), str2)) or bool(re.search(re.escape(str2), str1))

    def calculate_metricss(self,predictions, truths):
        if len(predictions)==0:
            return 0.00,0.00,0.00
        # print("=========calculate_metricss======")
        # print(predictions)
        # print(truths)
        true_positives = 0
        for pred in predictions:
            for truth in truths:
                if pred[1] == truth[1] and (self.fuzzy_match(pred[0], truth[0]) and self.fuzzy_match(pred[2], truth[2])):
                    true_positives += 1
                    break

        precision = true_positives / len(predictions) if len(predictions) > 0 else 0
        recall = true_positives / len(truths) if len(truths) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def _calc_fuzzy_score(self,golden, generated, src):
        """计算单个三元组的模糊匹配得分"""
        if len(generated)==0:
            return [0.0]
        def str2list(golden: str):
            # 正则表达式匹配模式
            triple_pattern = re.compile(r'（(.*?)，(.*?)，(.*?)）')

            # 解析字符串，提取所有匹配的三元组
            triples = triple_pattern.findall(golden)

            # 输出转换后的三元组列表
            return triples

        fuzzy_scores = []
        if all(len(pred) == 0 for pred in generated):
            for _ in generated:
                fuzzy_scores.append(0)
            return fuzzy_scores

        golden_list = str2list(golden)
        # 预处理所有golden关系嵌入（提高计算效率）
        golden_relations = [t[1] for t in golden_list] if golden_list else []
        gold_rel_embeds = self.sim_encoder.encode(golden_relations) if golden_relations else None
        gold_rel_embeds_tensor=torch.tensor(gold_rel_embeds)

        for gen in generated:
            score = 0.0
            num = 0
            for g in gen:
                # 实体边界验证
                e1_in_src = 1.0 if g[0] in src else 0
                e2_in_src = 1.0 if g[2] in src else 0
                entity_valid = 0.5 * (e1_in_src + e2_in_src)

                if gold_rel_embeds is not None:
                    # 关系语义相似度
                    rel_embed = self.sim_encoder.encode(g[1])
                    # rel_embed_tensor 是单个关系的嵌入，gold_rel_embeds_tensor 是所有黄金关系的嵌入
                    # 需要将 rel_embed_tensor 扩展为二维 (1, D)，这样可以计算与所有黄金关系的相似度
                    rel_embed = self.sim_encoder.encode(g[1])
                    rel_embed_tensor = torch.tensor(rel_embed)
                    rel_embed_tensor = rel_embed_tensor.unsqueeze(0)  # 转换为 (1, D)
                    similarities = cosine_similarity(rel_embed_tensor, gold_rel_embeds_tensor)  # 结果是 (1, N)
                    # 获取与黄金关系中最相似的一个关系的相似度
                    rel_sim = similarities.max().item()  # 获取最大相似度值
                    # rel_embed_tensor = torch.tensor(rel_embed)
                    # gold_rel_embeds_tensor = torch.tensor(gold_rel_embeds)
                    # similarities = cosine_similarity(rel_embed_tensor, gold_rel_embeds_tensor)[0]
                    # rel_sim = torch.max(torch.tensor(similarities)).item()
                    #     # .clone().detach().item()
                else:
                    rel_sim = 0
                triple_score = 0.7 * entity_valid + 0.3 * rel_sim
                score += triple_score
                num += 1
            if num == 0:
                f_score = 0.0
            else:
                f_score = score / num
            fuzzy_scores.append(f_score)

        return fuzzy_scores

    def _calc_structure_score(self,src: str, triples):
        # 计算三元组中实体在原句中的位置合理性
        s_score = []
        for triple in triples:
            scores = 0.0
            num = 0
            for t in triple:
                e1, rel, e2 = t
                e1_start = src.find(e1)
                e2_start = src.find(e2)
                if e1_start != -1 and e2_start != -1:
                    position_diff = abs(e1_start - e2_start)
                    scores += 1.0 - (position_diff / len(src))
                else:
                    scores += 0.0
                num += 1
            if num == 0:
                s_score.append(0.0)
            else:
                s_score.append(scores / num)
        return s_score



    def _bootstrap_rewards(self, rewards: List[float],num_bootstraps) -> torch.Tensor:
        # 将Python列表转换为PyTorch张量
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # 生成随机索引（形状为 [num_bootstraps, len(rewards)]）
        indices = torch.randint(0, len(rewards), (num_bootstraps, len(rewards)))

        # 对每个自助样本计算最大值
        return torch.stack([torch.max(rewards[idx]) for idx in indices])


    def _boostrap_max_rewards_k_times(
        self,
        rewards: List[float],
        k: int
    ) -> List[float]:
        # Segment list rewards into k equal sub-lists
        l = len(rewards)
        assert l % k == 0, f'l={l}, k={k}'
        segmented_rewards = [rewards[i*l//k:(i+1)*l//k]
                             for i in range(k)]  # [k, l/k]
        # We use different rewards for each bootstrap for now
        bootstrap_rewards = segmented_rewards

        # For each sub-list, take the max as the sub-reward
        values, indices = (torch.tensor(bootstrap_rewards)
                           .float().max(axis=1))
        # Take numbers from the original list to avoid numerical issues
        bootstrap_max_rewards = [bootstrap_rewards[i][index]
                                 for i, index in enumerate(indices)]

        return bootstrap_max_rewards


    def _compute_reward_zscores(
            self,
            rewards_tensor: torch.Tensor,
            input_texts,
            input_rewards: Dict[str, List[float]],
            eps: float = 1e-4
    ) -> torch.Tensor:
        # input_texts=self._repeat_texts(input_texts)
        input_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
        input_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
        idx_means = torch.tensor([input_reward_means[s] for s in input_texts])
        idx_stds = torch.tensor([input_reward_stds[s] for s in input_texts])
        # print(idx_means)
        # print(idx_stds)
        return (rewards_tensor - idx_means.float()) / (idx_stds.float() + eps)








    def _repeat_texts(
        self,
        texts: List[str],
        num_repeats
    ) -> List[str]:
        return [texts for _ in range(num_repeats)]

    def evaluate_score(self,text, gold_triples, prompt):
        hypos = self.generator.val_generate(
            prompt, text
        )
        # print("==========evaluate_hypos====")
        # print(gold_triples)
        # print(hypos)
        def str2list(golden: str):
            # 正则表达式匹配模式
            triple_pattern = re.compile(r'（(.*?)，(.*?)，(.*?)）')
            # 解析字符串，提取所有匹配的三元组
            triples = triple_pattern.findall(golden)
            # 输出转换后的三元组列表
            return triples

        def normalize_text(text):
            return re.sub(r'[“”"\']', '', text)  # 去除中文引号、英文引号

        def normalize_triples(list_triple):
            return [normalize_text(item) for item in list_triple]

        hypos = normalize_triples(hypos)
        # 三元组解析
        generated_triples = self._extract_triples(hypos)
        golden = normalize_text(gold_triples)
        golden_list = str2list(golden)
        # 真实标签转化为集合
        true_triplets = set(golden_list)
        # 预测标签转换为集合
        predictions = set()
        if len(generated_triples)==0:
            predictions.update([])
        else:
            predictions.update(generated_triples[0])

        return self.calculate_metricss(predictions,true_triplets)
        # print("======验证结果=======")
        # print("f1:"+str(f1))

    def save_hypos(self,text,true_label,predicted_label,precision,recall,f1_score):
        max_retries = 3
        retry_delay = 5  # 每次重试间隔5秒

        # 连接数据库并重试
        for attempt in range(max_retries):
            try:
                conn = pymysql.connect(host=IP, user='root', passwd=MYSQLPWD, db=DB)
                break  # 成功连接，跳出循环
            except pymysql.err.OperationalError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    conn = None  # 确保 conn 变量存在，即使连接失败
        if conn:
            cursor = conn.cursor()
            sql="INSERT INTO hypos_logs_chatgpt_138(content, true_label, predicted_label, precision_score, recall_score, f1_score) VALUES (%s,%s,%s,%s,%s,%s)"
            cursor.execute(sql,(
                str(text),
                str(true_label),
                str(predicted_label),
                precision,
                recall,
                f1_score
            ))
            conn.commit()
            print("预测结果保存成功")

if __name__ == '__main__':
    if not os.path.exists(similarity_model):
        raise FileNotFoundError(f"Model path does not exist: {similarity_model}")
    print(similarity_model)
    r = PromptedRelationExtractReward( similarity_model,num_repeats, num_samples, num_bootstraps, compute_zscore,
                                           )

    train_dataset, val_dataset, test_dataset = make_relation_extract_dataset(base_path, dataset)

    new_prompt="""1. 从以下文本中提取出“制造”关系三元组：(实体，制造，实体)。
2. 请从给定的资料中找出哪些实体之间存在“供应”关系，并以三元组形式输出：(实体，供应，实体)。
3. 从这段内容中提取出涉及“构成”关系的三元组：(实体，构成，实体)。
4. 请识别文本中的制造关系，返回三元组：(实体，制造，实体)。
5. 根据提供的资料，找出哪些实体之间有供应关系，输出为三元组：(实体，供应，实体)。
6. 以三元组的形式识别构成关系：(实体，构成，实体)。
7. 请提取出制造关系的三元组：(实体，制造，实体)，并输出。
8. 结合以下信息，提取供应关系的三元组：(实体，供应，实体)。
9. 从文本中识别出构成关系并输出：(实体，构成，实体)。
10. 请提取出文本中的制造、供应或构成三元组并输出。"""
    text = test_dataset.source_texts
    gold_labels = test_dataset.re_labels
    mode="train"
    reward = r.compute_reward(text, gold_labels, new_prompt, True, mode)
