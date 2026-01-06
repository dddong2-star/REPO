import random
import re

import openai
import torch
from openai import OpenAI
from sentence_transformers import util
from sympy.parsing.sympy_parser import null
from transformers import AutoTokenizer, pipeline, BertModel

class StateModel():
    def __init__(self,
                 token_model: str ,):
        def bert_encode(self, text):
            inputs = self.tokenizer.encode_plus(
                text,
                max_length=512,  # 确保最大长度不超过 512
                truncation=True,  # 过长时自动截断
                padding="max_length",  # 保证长度一致
                return_tensors="pt"
            )
            return self.bert_model(**inputs).last_hidden_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(token_model ,trust_remote_code=True)
        # ,
        # pad_token='<pad>')
        self.model= BertModel.from_pretrained(token_model ,trust_remote_code=True)
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=token_model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False

    def get_state(
            self,
            prompt: str):

        inputs = self.tokenizer(text=prompt, return_tensors='pt', max_length=512,
                truncation=True,
                padding="max_length")
        with torch.no_grad():  # 关闭梯度计算，节省内存
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # 获取最后一层的隐藏状态
        # 取均值 (在 token 维度上进行平均)
        state=last_hidden_state
        # prompt_emb = last_hidden_state.mean(dim=1)  # [1, 768]
        # # 拼接文本编码历史奖励
        # state = torch.cat([
        #     prompt_emb
        #     # torch.tensor([history_rewards[-5:].mean()]).unsqueeze(0),
        # ], dim=-1)
        return state


class REINFORCE:

    def take_action(self,action_id, prompt,val_dataset=None):
        """
        执行动作
        输入:
          - action_id: 动作编号 (0~10)
          - prompt: 当前提示模板
        输出: 修改后的提示模板
        """
        if action_id == 0:  # 避免优化过度 - 保持原提示
            return prompt

        elif action_id == 1:  # 改变句式
            return self._modify_sentence_structure(prompt)

        elif action_id == 2:  # 同义词替换
            return self._replace_with_synonyms(prompt)

        elif action_id == 3:  # 增加实体类型
            return self._add_entity_types(prompt)

        elif action_id == 4:  # 增加实体描述
            return self._add_entity_description(prompt)

        elif action_id == 5:  # 增强实体位置
            return self._enhance_entity_position(prompt)

        elif action_id == 6:  # 增加关系描述
            return self._add_relation_description(prompt)

        elif action_id == 7:  # 增强关系位置
            return self._enhance_relation_position(prompt)

        elif action_id == 8:  # 增加输入/输出对
            return self._add_input_output_pairs(prompt,val_dataset)

        elif action_id == 9:  # 增加句子模板
            return self._add_sentence_templates(prompt)

        elif action_id == 10:  # 增强输出格式
            return self._enhance_output_format(prompt)

        else:
            raise ValueError(f"未知的动作编号: {action_id}")


    """**提示：**
    
    请根据输入信息抽取以下关系：
    
    1. **生产**关系：提取哪些公司生产了哪些产品或技术。例如：“公司A”，“生产”，“产品B”。
    2. **供应商**关系：提取公司是否为某一产品或行业的供应商。例如：“公司A”，“供应商”，“产品B”。
    3. **构成**关系：提取某一产品的构成部分。例如：“产品A”，“构成”，“部分B”。
    4. **合作**关系：提取公司之间的合作关系。例如：“公司A”，“合作”，“公司B”。
    5. **应用领域**关系：提取产品或技术的应用领域。例如：“产品A”，“应用于”，“领域B”。
    
    ---
    
    **示例结构：**
    
    - （公司A，“生产”，产品B）
    - （产品A，“我构成了”，部分B）
    - （公司A，“供应商”，行业B）
    - （公司A，“合作”，公司B）
    - （产品A，“应用于”，领域B）
    
    ---
    
    请从提供的输入中提取并输出符合上述关系的语句。
    
    """
    def _modify_sentence_structure(self, prompt):
        """改变句式，调整提示的语法结构"""
        responses = self.llm_query(
            f"请用不同句式改写以下提示，保持语义不变:\n{prompt}",
        )
        # 选择与原文意最接近的变体
        return responses

    def _replace_with_synonyms(self, prompt):
        """对提示中的关键词进行正则化"""
        prompt = re.sub(r'(生产了|生产出|制造了|制造出)', '生产', prompt)
        # 构成关系的正则替换
        prompt = re.sub(r'(构成了|包含|组成)', '构成', prompt)
        # 供应商关系的正则替换
        prompt = re.sub(r'(供应商|提供|供货)', '供应商', prompt)
        # 向量库检索
        return prompt

    def _add_entity_types(self, prompt):
        """增加实体类型信息"""
        str=""
        if "公司"  in prompt:
            str+="‌公司是经济实体,包括有限责任公司和股份有限公司。"
        if "产品" in prompt:
            str+="产品是指被人们使用和消费，并能满足人们某种需求的任何东西，包括有形的物品、无形的服务、组织、观念或它们的组合。"
        return prompt

    def _add_entity_description(self, prompt):
        """增加实体的详细描述"""
        str=""
        if "公司" in prompt:
            str += "‌公司是经济实体。公司名称通常包括有限责任公司或者股份有限公司。" \
                   "在新闻中可能也会使用股票名称指代公司。比如：小米公司，小米科技有限责任公司等。"
        if "产品" in prompt:
            str += "产品是指被人们使用和消费，并能满足人们某种需求的任何东西，包括有形的物品、无形的服务、组织、观念或它们的组合。。"

        return prompt + str

    def _enhance_entity_position(self, prompt):
        """增强对实体位置的关注"""
        return prompt + " 注意识别实体位置，确保正确提取关系的起点和终点。"

    def _add_relation_description(self, prompt):
        """增加关系的详细描述"""
        relation_desc = """供应商关系：如果句子中描述了A公司向B公司提供产品，抽取结果(A公司,"供应商",B公司)。
        生产关系：如果句子中描述了A公司生产X产品，抽取结果(A公司,"供应商",X产品)。
        构成关系：如果句子中描述了P产品作为X产品的构成部分，抽取结果(P产品,"构成",X产品)"""
        return prompt + relation_desc

    def _enhance_relation_position(self, prompt):
        """增强对关系位置的关注"""
        return prompt + " 请确定关系的方向，明确关系的主体和客体。"

    def _add_input_output_pairs(self, prompt,val_dataset):

        data = random.sample(list(val_dataset),1)
        examples = "\n".join(f"示例输入：{item['source_texts']} 示例输出：{item['re_labels']}" for item in data)
        """增加示例输入/输出对"""
        return prompt+examples

    def _add_sentence_templates(self, prompt):
        """增加句子模板"""
        templates =["""A公司（为/向）B公司（提供/供应）P产品，则可以抽取出(A, "供应商", B)，(A, "生产", P)。""",
"""A公司提供P产品，应用于B公司的产品，则可以抽取出(A, "供应商", B)，(A, "生产", P)。""",
"""A公司是B公司的供应商，为B公司提供P产品，则可以抽取出(A, "供应商", B)，(A, "生产", P)。""",
"""B公司是A公司的客户，则可以抽取出(A, "供应商", B)""",
"""A公司供货B公司，则可以抽取出(A, "供应商", B)""",
"""A公司是B公司P产品的供应商，则可以抽取出(A, "供应商", B)，(A, "生产", P)。""",
"""A公司的P产品的客户是B公司，则可以抽取出(A, "供应商",B)，(A, "生产", P)。""",
"""A公司的P产品向B，则可以抽取出公司供货(A, "供应商", B)，(A, "生产", P)。""",
"""A公司与B公司合作，为X产品提供/交付P产品则可以抽取出(A, "供应商", B)，(A, "生产", P)，(B, "生产", X)，(P, "构成", X)。""",
"""A公司与B公司围绕p产品展开合作，则可以抽取出(A, "供应商", B)，(A, "生产", P)。""",
"""A公司的P产品进入B公司供应链，则可以抽取出(A, "供应商", B)，(A, "生产", P)。""",
]

        random_template = random.choice(templates)
        return prompt + random_template

    def _enhance_output_format(self, prompt):
        """增强输出格式要求"""
        format_req = " 请以三元组形式输出：(实体1, 关系, 实体2)。"
        return prompt + format_req

    def llm_query(self ,prompt, temperature=0.7, max_tokens=500):
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            base_url="https://api.xty.app/v1",
            api_key="sk-xxx",
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
        return r

