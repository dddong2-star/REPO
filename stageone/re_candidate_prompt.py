import random
import re
import time
import traceback

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

from stageone.data_loder.dataset import make_relation_extract_dataset


class ReCandidatePrompt:
    def __init__(self):
        # self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1",trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1",trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(self.device)

    #total为生成提示总数
    #data为数据集
    def llm_generate(self,total,data):
        res=[]
        """"
        逻辑：
        1. 设计引导任务提示
        2. 修改温度采样增加多样性
        3. 约束生成格式
        """
        examples = "\n".join([f"示例输入：{item['source_texts']} 示例输出：{item['re_labels']}" for item in data])

        num=0
        while num<total:
            if num < total / 3:
                temperature = 1.5
            else:
                temperature = 0.7

            prompt = f"""
                    参考示例输入，经过大模型处理后得到示例输出。理解任务，根据任务生成一个关系抽取提示，使大模型下一次能完成这个任务。要求：
                    1. 仅返回提示
                    2. 覆盖病因、药物治疗、临床表现三类关系。
                    {examples}"""
            client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                base_url="https://api.xty.app/v1",
                api_key="sk-AgLrjpbbfCEFEi3wuP1Zn03FOy38TcIZbWWpZLvvRNmhbHVx",
            )
            completion = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=temperature,
            )
            r = completion.choices[0].message.content
            print("===============第"+str(num)+"提示===================")
            print(r)
            if "病因" in r or "药物治疗" in r or "临床表现" in r:
                res.append(r)
                num+=1
        return res

    #prompt为提示集合
    #test_data为测试数据集
    #返回数据 为字典
    def llm_evluate(self,prompt,test_data):
        good_candidate=[]
        text,labels=test_data.source_texts,test_data.re_labels
        num=len(text)
        #计算这个提示在整个数据集上的平均得分
        for p in prompt:
            grade=0
            for s,l in zip(text,labels):
                content=p+s
                print(content)
                for i in range(1, 3):
                    try:
                        client = OpenAI(
                            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                            base_url="https://api.xty.app/v1",
                            api_key="sk-AgLrjpbbfCEFEi3wuP1Zn03FOy38TcIZbWWpZLvvRNmhbHVx",
                        )
                        completion = client.chat.completions.create(
                            model='gpt-3.5-turbo',
                            messages=[
                                {'role': 'system', 'content': ''},
                                {'role': 'user', 'content': content}
                            ],
                            temperature=0,
                        )
                        r = completion.choices[0].message.content
                        # 去掉引号
                        def normalize_text(text):
                            return re.sub(r'[“”"\']', '', text)
                        r=normalize_text(r)
                        #标签变成元组列表
                        def str2list(golden: str):
                            # 正则表达式匹配模式
                            pattern = r"（([^，]+)，“([^”]+)”，([^）]+)）"
                            # 提取三元组并转换为元组列表
                            triplets = [tuple(match) for match in re.findall(pattern, golden)]
                            # 输出结果
                            return triplets
                        l=str2list(l)
                        #列表去掉引号
                        def normalize_triples(triples):
                            return {tuple(normalize_text(item) for item in triple) for triple in triples}
                        l=normalize_triples(l)


                        generated_triples = self.extract_triples(r)

                        f1=self.calc_exact_scores(generated_triples,l)
                        print("=====当前p的准确率、召回率、f1值=======")
                        # print(p)
                        # print(precision)
                        # print(recall)
                        print(f1)
                        print("====================================")
                        fuzzy_scores =self.calc_fuzzy_scores (s,generated_triples, l)
                        combined_rewards =0.7 * f1 +0.25 * fuzzy_scores
                        grade+=combined_rewards
                    except Exception as e:
                        print(f"错误信息：{e}")
                        print("=== 详细堆栈信息 ===")
                        traceback.print_exc()
                        time.sleep(5)
                    else:
                        break
            print("=======平均得分======")
            print(grade/num)
            print("=============")
            dic={'prompt':p,"grade":grade}
            good_candidate.append(dic)
        return good_candidate

    def extract_triples(self, text):
        found = []
        pattern = re.compile(r'[（(]\s*([^,，]+)\s*[,，]\s*([^,，]+)\s*[,，]\s*([^)）]+)\s*[）)]')
        matches = pattern.findall(text)
        for match in matches:
            found.append(tuple(match))
        return found

    def extract_triples(self, text):
        if not isinstance(text, str):
            return set()
        found = []
        # 支持英文/中文括号，忽略空白符
        pattern = re.compile(r'[（(]\s*([^),;]+?)\s*[，,]\s*([^),;]+?)\s*[，,]\s*([^)）;]+?)\s*[）)]', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            found.append(tuple(match))
        return set(found)
    #计算抽取结果和真实标签的准确率
    def calc_exact_scores(self, pred, golden):
        true_triplets = set(golden)
        if len(pred) == 0:
            return 0
        predictions = set()
        predictions.update(pred)
        # 计算精准度（Precision）和召回率（Recall）
        intersection = true_triplets & predictions  # 找到真实标签与预测结果的交集
        precision = len(intersection) / len(predictions) if predictions else 0
        recall = len(intersection) / len(true_triplets) if true_triplets else 0
        # 计算F1得分
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1
    #计算相似度得分-所有标签的相似度得分的平均值
    def calc_fuzzy_scores(self, src, pred, golden):
        if len(pred) == 0:
            return 0
        # 预处理所有golden关系嵌入（提高计算效率）
        golden_relations = [t[1] for t in golden] if golden else []
        gold_rel_embeds = self.sim_encoder.encode(golden_relations) if golden_relations else None
        gold_rel_embeds_tensor = torch.tensor(gold_rel_embeds)
        score = 0.0
        num = 0
        for g in pred:
            e1_in_src = 1.0 if g[0] in src else 0
            e2_in_src = 1.0 if g[2] in src else 0
            entity_valid = 0.25 * (e1_in_src + e2_in_src)
            if gold_rel_embeds is not None:
                # 关系语义相似度
                rel_embed = self.sim_encoder.encode(g[1])
                rel_embed_tensor = torch.tensor(rel_embed)
                rel_embed_tensor = rel_embed_tensor.unsqueeze(0)  # 转换为 (1, D)
                similarities = cosine_similarity(rel_embed_tensor, gold_rel_embeds_tensor)  # 结果是 (1, N)
                # 获取与黄金关系中最相似的一个关系的相似度
                rel_sim = similarities.max().item()/2  # 获取最大相似度值
            else:
                rel_sim = 0
            triple_score = 0.7 * entity_valid + 0.3 * rel_sim
            score += triple_score
            num += 1
        if num == 0:
            f_score = 0.0
        else:
            f_score = score / num
        return f_score

    # LLM重写提示
    def rewrite_prompts(self,prompts):
        new_prompts = []
        tishi="""假设你是一名提示工程师，请你重写这个关系抽取任务的提示。
                 要求：
                    1. 仅返回提示
                    2. 覆盖病因、药物治疗、临床表现这三类关系    
                    3.字数在50词左右
                    """
        #// 生产、供应、构成三类关系
        for p in prompts:
            i=0
            client = OpenAI(
                base_url="https://api.xty.app/v1",
                api_key="sk-AgLrjpbbfCEFEi3wuP1Zn03FOy38TcIZbWWpZLvvRNmhbHVx",
            )
            messages = [
                {'role': 'system', 'content': '你是一个提示工程师，请按照要求书写提示'}]
            while i<2:
                messages.append({'role': 'user', 'content':tishi+p })
                try:
                    completion = client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=messages,
                        temperature=0.7,
                    )

                    ai_response = completion.choices[0].message.content
                    print(f"AI: {ai_response}")

                    messages.append({'role': 'assistant', 'content': ai_response})
                    i=i+1
                    if i==2:
                        new_prompts.append(ai_response)
                except Exception as e:
                    print(f"Error: {e}")
        return new_prompts

    def candidate_prompt_construction(self,
                                      initial_nums,
                                      train_dataset,
                                      test_dataset,
                                      max_rounds=6,
                                      k_percent=50,
                                      threshold=0.01):
        # Step 1: 生成初始提示集合
        subset_size = min(10, int(0.1 * len(train_dataset)))
        print("=====sub_size："+str(subset_size)+"===============")
        Z_train = random.sample(list(train_dataset), subset_size)
        U = self.llm_generate(initial_nums,Z_train)
        #评估分数
        scored_U= self.llm_evluate(U, test_dataset)
        top_k = max(1, int(len(U) * k_percent / 100))
        top_prompts = [item["prompt"] for item in sorted(scored_U, key=lambda x: x["grade"], reverse=True)[:top_k]]
        #选择种子提示Z_seed
        Z_seed = [p for p in top_prompts]
        #初始评分
        prev_avg = sum(item["grade"] for item in sorted(scored_U, key=lambda x: x["grade"], reverse=True)[:top_k]) / top_k
        print("初始平均分："+str(prev_avg))
        global_best = {
            "prompt": None,
            "grade": -float('inf')
        }
        U_K=[]
        stable_count=0
        for r in range(1,max_rounds):
            # Step 4: 打乱并重写提示
            random.shuffle(Z_seed)
            U_prime = self.rewrite_prompts(Z_seed)
            #合并历史高质量提示
            merged_U = U_prime + Z_seed
            # 评估新提示
            scored_merged = self.llm_evluate(merged_U,test_dataset)
            # 更新全局最优
            current_best = max(scored_merged, key=lambda x: x["grade"])
            print("第"+str(r)+"轮的当前最高分"+str(current_best))
            if current_best["grade"] > global_best["grade"]:
                global_best = current_best
            # 筛选Top-k%
            top_k = max(1, int(len(scored_merged) * k_percent / 100))
            top_prompts = [item["prompt"] for item in sorted(scored_merged, key=lambda x: x["grade"], reverse=True)[:top_k]]
            U_K = [p for p in top_prompts]
            current_avg = sum(item["grade"] for item in sorted(scored_merged, key=lambda x: x["grade"], reverse=True)[:top_k]) / top_k
            # Step 7-10: 每5轮重采样
            if r % 5 == 0:
                resampled_prompts = self.diversity_resampling(
                    prompts_with_scores=scored_merged,
                    sample_ratio=0.8
                )
                U_k = [p["prompt"] for p in resampled_prompts]
                current_avg = sum(p["grade"] for p in resampled_prompts) / len(resampled_prompts)
            # 更新最优提示


            # MODIFIED: 增强收敛判断（需连续3轮稳定）
            if abs(current_avg - prev_avg) < threshold:
                stable_count += 1
                if stable_count >= 3:
                    break
            else:
                stable_count = 0
            prev_avg = current_avg
            Z_seed = U_K  # 更新种子集合
        return global_best["prompt"]

    def bert_encode(self, text):
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    def diversity_resampling(self, prompts_with_scores, sample_ratio=0.8):
        from sklearn.cluster import KMeans
        import numpy as np

        # 特征提取
        embeddings = [self.bert_encode(p["prompt"]) for p in prompts_with_scores]
        # 动态聚类
        n_clusters = int(np.sqrt(len(prompts_with_scores)))
        clusters = KMeans(n_clusters=n_clusters).fit_predict(embeddings)
        print("=======clusters=======")
        print(clusters)
        # 计算每个簇的权重
        cluster_weights = {}
        for idx, cluster_id in enumerate(clusters):
            score = prompts_with_scores[idx]["grade"]
            if cluster_id not in cluster_weights:
                cluster_weights[cluster_id] = {"total": 0.0, "count": 0}
            cluster_weights[cluster_id]["total"] += score
            cluster_weights[cluster_id]["count"] += 1

        # 分配采样名额
        total_weight = sum(v["total"] * v["count"] for v in cluster_weights.values())
        sample_num = int(len(prompts_with_scores) * sample_ratio)

        selected = []
        for cluster_id in cluster_weights:
            # 计算该簇应选数量
            weight = (cluster_weights[cluster_id]["total"] *
                      cluster_weights[cluster_id]["count"]) / total_weight
            n_select = max(1, int(sample_num * weight))

            # 从该簇选择Top-N
            cluster_items = [p for idx, p in enumerate(prompts_with_scores)
                             if clusters[idx] == cluster_id]
            cluster_sorted = sorted(cluster_items, key=lambda x: x["grade"], reverse=True)
            selected.extend(cluster_sorted[:n_select])

        # 确保数量正确
        return selected[:sample_num]


if __name__ == '__main__':
    base_path = "../data"
    dataset = "triple_CMeIE"
    initial_nums=10
    (train_dataset, val_dataset, test_dataset) = \
        make_relation_extract_dataset(base_path, dataset)
    obj=ReCandidatePrompt()
    res=obj.candidate_prompt_construction(10,train_dataset,
                                          test_dataset)


    def save_best_prompt(best_prompt, file_path):
        with open(file_path, 'w') as f:
            f.write(best_prompt)
    file_path="./candidate_best_prompt.txt"
    save_best_prompt(res,file_path)
