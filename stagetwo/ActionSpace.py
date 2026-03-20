import json
import re
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import pandas as pd


client = OpenAI(
    api_key="sk-AgLrjpbbfCEFEi3wuP1Zn03FOy38TcIZbWWpZLvvRNmhbHVx",  # 替换为你的API Key
    base_url="https://api.xty.app/v1"  # 如使用代理/本地化部署的LLM地址（可选）
)


class ActionSpace:
    """
    4: Add entity description - 从标注样本中提取实体属性生成领域化实体描述
    6: Add relation description - 从标注样本中提取关系上下文生成领域化关系描述
    """

    def __init__(self, domain: str, llm_model: str = "gpt-3.5-turbo"):
        """
        初始化动作空间
        :param domain: 目标领域
        :param llm_model: 调用的LLM模型名
        """
        self.domain = domain.lower()
        self.llm_model = llm_model
        # 动作空间的核心定义（4/6号）
        self.action_def = {
            4: {"name": "增加实体描述",
                "desc": "为实体添加符合目标领域的属性描述，基于标注样本的实体上下文和属性信息"},
            6: {"name": "增加关系描述",
                "desc": "为关系添加符合目标领域的语义描述，基于标注样本的关系上下文和实体关联信息"},
            2:{"name": "同义词替换",
                "desc": "对提示中的关键词进行正则化"},
            3: {"name": "增加实体类型",
                "desc": "增加实体类型信息"},
            8: {"name": "增加句子模板",
                "desc": "增加句子模板"},

        }
        self.domain_relation = self._get_demain_relation()


    def _get_demain_relation(self) ->str:
        demain_relations = {
            "医疗": "病因、临床表现、药物治疗三类关系",
            "金融": "合作、拥有两类关系",
            "法律": "贩卖（毒品）、贩卖（给人）、非法容留、持有四类关系",
            "公司新闻":"供应商、构成、生产三类关系"
        }
        return demain_relations.get(self.domain)

    def parse_annotation_samples_from_tsv(self, file_path: str) -> List[Dict]:
        """
        从all.tsv读取数据，随机抽取5条，并解析为关系抽取三元组样本
        :param file_path: tsv文件路径
        :return: 标准化样本列表
        """
        # 读取tsv
        df = pd.read_csv(file_path, sep="\t")
        # 随机抽取5条
        df = df.sample(n=5, random_state=None)
        parsed_samples = []
        for text, label_str in zip(df["text"], df["label"]):
            # 清洗文本
            context = re.sub(r"\s+", " ", str(text)).strip()
            # 提取三元组
            triples = re.findall(r'[（(](.*?)[,，]["“]?(.*?)["”]?[,，](.*?)[）)]', label_str)
            for h, r, t in triples:
                parsed_samples.append({
                    "context": context,
                    "head_entity": h.strip(),
                    "tail_entity": t.strip(),
                    "relation": r.strip()
                })

        return parsed_samples

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """
        封装LLM调用接口（适配openai 1.0.0+版本），生成动作内容
        :param prompt: 输入LLM的提示词
        :param temperature: 生成温度，0-1之间，越小越确定
        :return: LLM生成的结果
        """
        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
                stop=None
            )
            # 提取生成结果（属性路径也有变化）
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"LLM调用失败: {str(e)}")

    def generate_action(self, action_id: int, file_path: str) -> Dict:
        """
        核心方法：生成指定动作的内容（仅支持4/6号动作）
        :param action_id: 动作编号，4或6
        :param samples: 训练集标注样本，parse_annotation_samples_from_tsv
        :return: 动作生成结果，包含动作信息、领域、生成的描述模板、样本依据
        """
        # 解析并清洗标注样本
        parsed_samples = self.parse_annotation_samples_from_tsv(file_path)
        if not parsed_samples:
            raise ValueError("无有效标注样本，无法生成动作内容")

        domain_relation_1= self.domain_relation
        action_info = self.action_def[action_id]
        # 选取前5个样本作为依据（避免样本过多导致提示词过长）
        sample_demo = json.dumps(parsed_samples[:5], ensure_ascii=False, indent=2)
        # print("sample_demo:",sample_demo)
        # print("domain:",self.domain)
        # print("domain_relation_1:",domain_relation_1)
        if action_id == 4:
            # 4号动作：添加实体描述的提示词
            llm_prompt = f"""
            请参考标注样本，为关系抽取任务中的涉及到的目标实体类型（如公司、疾病、毒品等）添加精准、简洁的{self.domain}领域专属描述，描述需贴合该领域的专业定义，仅补充实体核心属性、特征或所属范畴，不冗余拓展。
            示例1：公司是经济实体,包括有限责任公司和股份有限公司。
            示例2：产品是指被人们使用和消费，并能满足人们某种需求的任何东西，包括有形的物品、无形的服务、组织、观念或它们的组合。。
            标注样本：
            {sample_demo}
            要求：1. 只生成对实体类型的描述，不输出对具体实体的描述；2. 单实体类型描述控制在 10-20 字 ；3.只选择2到3个和标注样本实体类型最相近的进行描述，不要太多；4.请以'模板n：实体类型:内容'的格式输出
            """
        elif action_id == 6:
            # 6号动作：添加关系描述的提示词
            llm_prompt = f"""
            为{self.domain}领域关系抽取中的{domain_relation_1}生成标准化描述，用于优化抽取prompt，描述需同时明确**文本触发条件**和**三元组抽取格式**。                      
            示例1：供应商关系：如果句子中描述了A公司向B公司提供产品，抽取结果(A公司,"供应商",B公司)。                                                                                         
            示例2：生产关系：如果句子中描述了A公司生产X产品，抽取结果(A公司,"供应商",X产品)。                                                        
            标注样本：                                             
            {sample_demo}                                     
            要求：1.只输出描述模板，不输出其他解释性内容；2. 描述控制在 10-20 字；3.每个描述必须只有一行，格式参考示例；4.请以'模板n：关系：内容和抽取形式'的格式输出"""
        elif action_id == 2:
            # 新增动作 2：生成关系近义词
            llm_prompt = f"""
                        请参考标注样本和{self.domain}领域的标准关系：{self.domain_relation}。
                        针对每个标准关系，列出在自然语言文本中可能出现的 3-4 个动词或短语（近义词），用于正则化处理。
                        标注样本：
                        {sample_demo}
                        要求：
                        1. 严格按照 '标准关系名:近义词1|近义词2|近义词3' 的格式输出；
                        2. 每个关系占一行；
                        3. 近义词应涵盖常见的变体（如：生产、制造、研制、产出）；
                        4. 只输出匹配结果，不要任何解释。
                        示例输出：
                        模板1：生产:生产|制造|加工|研发|研制
                        模板2：供应商:供应商|提供|供货|交付|配套
                        """
        elif action_id==3:
            llm_prompt = f"""
                        请参考标注样本，识别其中涉及的核心实体类型（如：公司、机构、症状、毒品名称等）,领域为{self.domain}。
                        为这些实体类型提供精准的、百科式的【类型定义】。
                        标注样本：
                        {sample_demo}
                        要求：
                        1. 格式严格按照 '模板n：实体类型:定义内容' 输出；
                        2. 定义内容需简洁专业，控制在 20-30 字以内；
                        3. 只针对样本中出现的 2-3 个核心实体类型进行定义；
                        4. 示例：公司:公司是依法设立的，以营利为目的的从事生产经营活动的经济组织。
                        """
        elif action_id  == 8:
            # 新增动作 8：增加句子模板
            llm_prompt = f"""
                        请参考标注样本和{self.domain}领域的（{domain_relation_1}），总结并生成 2-3 个该领域常见的关系抽取【句子模板】。
                        句子模板需使用抽象代词（如 A、B 代表实体主体，P、X 代表客体），说明典型的文本表达方式以及对应的三元组抽取结果。

                        标注样本：
                        {sample_demo}

                        要求：
                        1. 格式严格按照 '模板n：[句子模式]，则可以抽取出([实体1], "[关系]", [实体2])' 输出；
                        2. 每个模板占一行，包含文本触发词和抽取逻辑；
                        3. 请尽可能涵盖多重关系的联合抽取情况（如一个句子能抽出两个三元组）。
                        4. 只总结{domain_relation_1},必须忽略其他关系
                        示例输出：
                        模板1：A公司（为/向）B公司（提供/供应）P产品，则可以抽取出(A, "供应商", B)，(A, "生产", P)。
                        模板2：A与B围绕P展开合作，则可以抽取出(A, "合作", B)。
                        """
        # 调用LLM生成动作内容
        llm_result = self._call_llm(llm_prompt)
        # 解析LLM生成的模板（提取编号后的内容）
        templates = re.findall(r"模板\d+：(.*)", llm_result)
        templates = [t.strip() for t in templates if t.strip()]

        # 构造返回结果
        return {
            "action_id": action_id,
            "action_name": action_info["name"],
            "domain": self.domain,
            "generated_templates": templates if templates else [llm_result],  # 解析失败则返回原始结果
            "sample_count": len(parsed_samples),
            "action_description": action_info["desc"]
        }

    def batch_generate(self, file_path: str) -> Dict:
        """
        批量生成4和6号动作的内容
        :return: 包含两个动作生成结果的字典
        """
        return {
            4: self.generate_action(4, file_path),
            6: self.generate_action(6, file_path),
            2: self.generate_action(2, file_path),
            3: self.generate_action(3, file_path),
            8: self.generate_action(8, file_path)
        }


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 初始化动作空间
    myDomain = "LexEval"#换领域时需要更换
    base_path = "../data/triple/"
    file_type = ".tsv"
    file_path = base_path + myDomain + file_type
    print(file_path)
    action_space = ActionSpace(domain="法律", llm_model="gpt-3.5-turbo")#换领域时需要更换，金融、医疗、法律、公司新闻
    #  批量生成4和6号动作内容
    result = action_space.batch_generate(file_path)

    for act_id, act_result in result.items():
        print(f"===== 动作{act_id}：{act_result['action_name']} =====")
        print(f"领域：{act_result['domain']}")
        print(f"生成的描述模板：")
        for template in act_result['generated_templates']:
            print(template)
        print("-" * 80)