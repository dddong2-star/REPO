import random
import re
import time

import openai
import torch
from openai import OpenAI
from sentence_transformers import util
from sympy.parsing.sympy_parser import null
from transformers import AutoTokenizer, pipeline, BertModel
from stagetwo.ActionSpace import ActionSpace


class StateModel():
    def __init__(self,
                 token_model: str, ):
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
        self.tokenizer = AutoTokenizer.from_pretrained(token_model, trust_remote_code=True)
        # ,
        # pad_token='<pad>')
        self.model = BertModel.from_pretrained(token_model, trust_remote_code=True)
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
        state = last_hidden_state
        # prompt_emb = last_hidden_state.mean(dim=1)  # [1, 768]
        # # 拼接文本编码历史奖励
        # state = torch.cat([
        #     prompt_emb
        #     # torch.tensor([history_rewards[-5:].mean()]).unsqueeze(0),
        # ], dim=-1)
        return state


class REINFORCE:
    # 类级别缓存，确保在整个训练生命周期中只生成一次
    _cache_4 = None  # 存储实体描述字典 {"实体": "描述"}
    _cache_6 = None  # 存储关系描述字符串
    _cache_2 = None
    _cache_3 = None
    _cache_8 = None
    _action_tool = None  # 动作空间工具实例
    MAX_RETRIES = 3

    def take_action(self, action_id, prompt, val_dataset=None):
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
            return self._add_input_output_pairs(prompt, val_dataset)

        elif action_id == 9:  # 增加句子模板
            return self._add_sentence_templates(prompt)

        elif action_id == 10:  # 增强输出格式
            return self._enhance_output_format(prompt)

        else:
            raise ValueError(f"未知的动作编号: {action_id}")

    def _init_tool_if_needed(self):
        """仅在需要时初始化 ActionSpace 实例"""
        if self._action_tool is None:
            # 这里的领域和路径可以根据实际需求调整
            domain_name = "公司新闻"
            self.file_path = "../data/triple/LCN.tsv"
            self._action_tool = ActionSpace(domain=domain_name, llm_model="gpt-3.5-turbo")

    def _modify_sentence_structure(self, prompt):
        """改变句式，调整提示的语法结构1"""
        # 使用多行 f-string 构建强大的元提示词 (Meta-Prompt)
        meta_prompt = f"""
        # 角色设定
        你是一位资深的 AI 提示词工程师（Prompt Engineer）和自然语言处理（NLP）专家。你的擅长将冗长、混乱、重复的任务指令重构为结构清晰、逻辑严密、易于大模型理解的高质量提示词。
        # 任务目标
        我将提供一份关于“实体关系抽取（三元组）”的原始提示词草稿。这份草稿包含很多重复的指令、散乱的规则和过多的零碎示例。请你对其进行深度清洗和重构，输出一份可以直接用于生产环境的高质量提示词。

        # 重构核心原则与步骤
        1. **去重与精简**：删除所有重复的废话（如多次出现的“请确定关系的方向，明确关系的主体和客体”等）。
        2. **结构化定义**：用清晰的列表结构重写提示词涉及到的关系的核心定义。
        3. **提取句式规律（关键）**：深入分析原始草稿中的各种示例，总结出每个关系在自然语言中最常出现的几种特定句式（如：A向B提供P、B是A的客户等），并将这些规则作为明确的判别条件列出。
        4. **精简示例（Few-Shot 优化）**：不要罗列大量示例。请选出一、两个示例输入和示例输出。
        5. 不要修改原提示词各个句段的表述，只调整原提示每一句的顺序，格式参考原提示。
        6. **纯净输出（强制约束）**：直接输出重构后的提示词正文，**绝对不要**输出任何诸如“好的，我已经为您重写了”之类的解释性或寒暄性废话！
        
        # 输出格式要求
        请按照以下结构输出重构后的提示词：
        - 任务背景与输出格式要求
        - 关系分类与基础定义
        - 关系的句式判定规则
        - 示例输入和示例输出

        # 原始提示词草稿（输入）
        {prompt}
        """

        # 调用大模型接口进行请求
        responses = self.llm_query(meta_prompt)
        # 选择与原文意最接近的变体
        return responses

    def _replace_with_synonyms(self, prompt):
        """动作 2：自动化正则化处理"""
        self._init_tool_if_needed()

        if self._cache_2 is None:
            retries = 0
            while retries < self.MAX_RETRIES:
                print(f">>> 正在尝试生成关系近义词 (第 {retries + 1} 次)...")
                res = self._action_tool.generate_action(2, self.file_path)
                templates = res.get("generated_templates", [])

                temp_cache = {}
                for item in templates:
                    if ":" in item:
                        rel, syns = item.split(":", 1)
                        # 简单校验：近义词里是否包含 "|"
                        if "|" in syns:
                            temp_cache[rel.strip()] = syns.strip()

                if temp_cache:
                    self._cache_2 = temp_cache
                    break
                retries += 1

            if self._cache_2 is None:
                print("!!! 警告：动作 2 格式匹配失败，跳过正则化。")
                return prompt

        # 执行正则替换
        new_prompt = prompt
        for standard_rel, pattern in self._cache_2.items():
            # 将 (近义词1|近义词2) 替换为 标准关系名
            # 使用 re.sub 确保只匹配完整的词或特定模式
            regex_pattern = f"({pattern})"
            new_prompt = re.sub(regex_pattern, standard_rel, new_prompt)

        return new_prompt

    def _add_entity_types(self, prompt):
        """
        动作 3：自动化增加实体类型信息
        替代原有的 if "公司" in prompt 硬编码逻辑
        """
        self._init_tool_if_needed()

        # 1. 缓存与重试逻辑
        if self._cache_3 is None:
            retries = 0
            while retries < self.MAX_RETRIES:
                print(f">>> 正在尝试生成实体类型定义 (动作3, 第 {retries + 1} 次)...")
                res = self._action_tool.generate_action(3, self.file_path)
                templates = res.get("generated_templates", [])

                temp_dict = {}
                for item in templates:
                    for sep in [":", "："]:
                        if sep in item:
                            parts = item.split(sep, 1)
                            if len(parts) == 2:
                                temp_dict[parts[0].strip()] = parts[1].strip()
                                break

                if temp_dict:
                    self._cache_3 = temp_dict
                    print(temp_dict)
                    break
                retries += 1

            if self._cache_3 is None:
                print("!!! 警告：动作 3 生成失败，返回原 Prompt。")
                self._cache_3 = {}  # 设为空防止重复请求

        # 2. 自动化匹配逻辑 (替代原有的多个 if 语句)
        added_str = ""
        for entity_type, definition in self._cache_3.items():
            # 如果原 Prompt 中提到了该实体类型，则追加定义
            if entity_type in prompt:
                added_str += f"\n{entity_type}是{definition}"

        return prompt + added_str

    def _add_entity_description(self, prompt):
        """动作 4：增加实体描述，带有格式校验重试"""
        self._init_tool_if_needed()

        if self._cache_4 is None:
            retries = 0
            while retries < self.MAX_RETRIES:
                print(f">>> 正在尝试生成实体描述 (第 {retries + 1} 次)...")
                res = self._action_tool.generate_action(4, self.file_path)
                raw_templates = res.get("generated_templates", [])

                temp_dict = {}
                for item in raw_templates:
                    # 校验格式：必须包含中英文冒号，且分割后长度合理
                    for sep in [":", "："]:
                        if sep in item:
                            parts = item.split(sep, 1)
                            if len(parts) == 2 and len(parts[0].strip()) > 0:
                                temp_dict[parts[0].strip()] = parts[1].strip()
                                break

                # 校验：至少成功解析出 1 条才算成功
                if temp_dict:
                    self._cache_4 = temp_dict
                    print((temp_dict))
                    break
                else:
                    retries += 1
                    time.sleep(1)  # 短暂延迟，避免请求过快

            # 如果重试 3 次都失败了
            if self._cache_4 is None:
                print("!!! 警告：动作 4 生成格式持续不匹配，跳过描述注入。")
                self._cache_4 = {}  # 设为空字典防止再次调用 LLM

        # 匹配逻辑保持不变
        added_content = ""
        for entity_type, description in self._cache_4.items():
            if entity_type in prompt:
                added_content += f"\n{entity_type}：{description}"

        return prompt + added_content

    def _enhance_entity_position(self, prompt):
        """增强对实体位置的关注5"""
        return prompt + " 注意识别实体位置，确保正确提取关系的起点和终点。"

    def _add_relation_description(self, prompt):
        """动作 6：增加关系描述，带有格式校验重试"""
        self._init_tool_if_needed()

        if self._cache_6 is None:
            retries = 0
            while retries < self.MAX_RETRIES:
                print(f">>> 正在尝试生成关系描述 (第 {retries + 1} 次)...")
                res = self._action_tool.generate_action(6, self.file_path)
                templates = res.get("generated_templates", [])

                # 校验格式：确保返回的模板包含关系三元组的特征，如引号、括号或特定的“抽取结果”字样
                valid_templates = [t for t in templates if "抽取结果" in t or "(" in t or "（" in t]

                if valid_templates:
                    self._cache_6 = "\n".join(valid_templates)
                    print(valid_templates)
                    break
                else:
                    retries += 1

            if self._cache_6 is None:
                print("!!! 警告：动作 6 生成内容不符合规范，跳过注入。")
                self._cache_6 = ""

        return prompt + ("\n" + self._cache_6 if self._cache_6 else "")

    def _enhance_relation_position(self, prompt):
        """增强对关系位置的关注7"""
        return prompt + "\n" + " 请确定关系的方向，明确关系的主体和客体。"

    def _add_input_output_pairs(self, prompt, val_dataset):

        data = random.sample(list(val_dataset), 1)
        examples = "\n".join(f"示例输入：{item['source_texts']} 示例输出：{item['re_labels']}" for item in data)
        """增加示例输入/输出对"""
        return prompt + "\n" + examples

    def _add_sentence_templates(self, prompt):
        """
        动作 8：自动化增加句子模板
        动态生成领域模板库，每次随机挑选一句拼接
        """
        self._init_tool_if_needed()

        # 1. 缓存与重试逻辑
        if self._cache_8 is None:
            retries = 0
            while retries < self.MAX_RETRIES:
                print(f">>> 正在尝试生成句子模板库 (动作8, 第 {retries + 1} 次)...")
                res = self._action_tool.generate_action(8, self.file_path)
                templates = res.get("generated_templates", [])

                valid_templates = []
                for item in templates:
                    # 格式校验：模板中应包含 "抽取" 和括号等典型特征
                    if "抽取" in item and "(" in item and ")" in item:
                        valid_templates.append(item.strip())
                    elif "抽取" in item and "（" in item and "）" in item:
                        valid_templates.append(item.strip())

                # 校验：至少成功解析出 2 条以上模板才算构建成功，保证随机性
                if len(valid_templates) >= 2:
                    self._cache_8 = valid_templates
                    print(valid_templates)
                    break

                retries += 1

            if self._cache_8 is None:
                print("!!! 警告：动作 8 生成模板库失败，跳过添加。")
                self._cache_8 = []  # 设为空列表防止重复请求

        # 2. 全部加入逻辑 (由原来的 random.choice 修改为 join)
        if self._cache_8:
            # 将缓存的所有模板通过换行符拼接在一起
            all_templates = "\n".join(self._cache_8)
            return prompt + "\n" + all_templates
        else:
            return prompt

    def _enhance_output_format(self, prompt):
        """增强输出格式要求"""
        format_req = " 请以三元组形式输出：(实体1, 关系, 实体2)。"
        return prompt + format_req

    def llm_query(self, prompt, temperature=0.7, max_tokens=500):
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

