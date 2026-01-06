import re
from datetime import time

import pymysql
from openai import OpenAI
from typing import List, Tuple
IP = ""
MYSQLPWD = ''
DB = ''

def _extract_triples(texts: List[str]) -> List[List[Tuple]]:
    """三元组正则匹配"""
    triples = []
    for text in texts:
        rule_triples = _rule_based_extraction(text)
        triples.append(rule_triples)
    return triples


def _rule_based_extraction(text: str) -> List[Tuple]:
    """正则表达式匹配 (实体1, 关系, 实体2) 模式"""
    found = []
    pattern = re.compile(r'[（(]\s*([^,，]+)\s*[,，]\s*([^,，]+)\s*[,，]\s*([^)）]+)\s*[）)]')
    matches = pattern.findall(text)
    for match in matches:
        found.append(tuple(match))
    return found

def _calc_exact_scores(golden, generated):
    if len(generated) == 0:
        return 0.00, 0.00, 0.00

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
    golden = normalize_text(golden)
    golden_list = str2list(golden)
    # 真实标签转化为集合
    true_triplets = set(golden_list)
    # 预测标签转换为集合
    predictions = set()
    predictions.update(generated[0])
    precision, recall, f1 = calculate_metricss(predictions, true_triplets)

    return precision, recall, f1


def fuzzy_match(str1, str2):
    # 检查两个字符串是否有模糊匹配
    return bool(re.search(re.escape(str1), str2)) or bool(re.search(re.escape(str2), str1))


def calculate_metricss(predictions, truths):
    if len(predictions) == 0:
        return 0.00, 0.00, 0.00
    # print("=========calculate_metricss======")
    # print(predictions)
    # print(truths)
    true_positives = 0
    for pred in predictions:
        for truth in truths:
            if pred[1] == truth[1] and (fuzzy_match(pred[0], truth[0]) and fuzzy_match(pred[2], truth[2])):
                true_positives += 1
                break

    precision = true_positives / len(predictions) if len(predictions) > 0 else 0
    recall = true_positives / len(truths) if len(truths) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def yuce_generate(
        prompt: str,
        source_text: str,
        ) -> List[str]:
        formatted_template = f"{prompt}:{source_text}"
        #generated_texts = deepseek_extract(formatted_template)
        generated_texts=chatGPT_extract(formatted_template)
        return generated_texts

def chatGPT_extract(formatted_template,num_samples=1):
    res=[]
    for i in range(1,3):
        try:
            client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                base_url="https://api.xty.app/v1",
                api_key="sk-xxx",
            )
            completion = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': formatted_template}
                ],
                temperature=0.5,
            )
            # print(completion.choices[0].message.content)
            r = completion.choices[0].message.content
            res.append(r)
        except Exception as e:
            print(f"错误信息：{e}")
            time.sleep(15)
        else:
            break
    return res

def _repeat_texts(
    texts: List[str],
    num_repeats
) -> List[str]:
    return [texts for _ in range(num_repeats)]


def save_hypos(text,true_label,predicted_label,precision,recall,f1_score):
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
        sql="INSERT INTO hypos_logs_chatgpt_138_zhidai(content, true_label, predicted_label, precision_score, recall_score, f1_score) VALUES (%s,%s,%s,%s,%s,%s)"
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
    import pandas as pd

    df = pd.read_csv("data/original/all_zhidai.tsv", sep='\t')
    source_texts = df.text.tolist()
    re_labels = df.label.tolist()

    print(source_texts[:5])
    print(re_labels[:5])

    prompt="""根据下列新闻中的内容,总结新闻中的供应商关系,生产关系,构成关系。
                            以多个三元组的形式输出。没有此类关系则输出无。
                            关系描述如下:
                            1. 供应商关系:
                               如果句子中描述了A公司向B公司提供产品,可以抽取出(A, "供应商", B)。
                            2. 生产关系:
                               如果句子中描述了A公司生产某个产品P,可以抽取出 (A, "生产", P)。
                            3. 构成关系:
                               如果句子中描述了P产品作为X产品的构成部分,可以抽取出 (P, "构成", X)。
                            供应关系满足一下句式:
                            1.A公司（为/向）B公司（提供/供应）P产品,则可以抽取出(A, "供应商", B),(A, "生产", P)。
                            2.A公司提供P产品,应用于B公司的产品,则可以抽取出(A, "供应商", B),(A, "生产", P)。
                            3.A公司是B公司的供应商,为B公司提供P产品,则可以抽取出(A, "供应商", B),(A, "生产", P)。
                            4.B公司是A公司的客户,则可以抽取出(A, "供应商", B)
                            5.A公司供货B公司,则可以抽取出(A, "供应商", B)
                            示例输入:趣睡科技董秘回复||| 财联社4月1日电,有投资者问,小米汽车已上市,请问公司目前跟小米汽车有合作吗,公司之前回复称有跟汽车厂商合作能透露是哪几家不？趣睡科技在互动平台表示,公司作为一家专注于自有品牌科技创新家居产品的互联网零售公司,公司积极开发车载家居产品,公司新开发的车载遮阳帘等产品已陆续上线销售,并与部分国内汽车厂商开展合作,小米汽车作为公司重要客户之一。
                            输出:（趣睡科技,"供应商",小米）,（小米,"生产",小米汽车）,（趣睡科技,"生产",车载遮阳帘）,（车载遮阳帘,"构成",小米汽车） """
    all_p=0
    all_r=0
    all_f=0
    num_repeats = len(source_texts)
    new_prompt = _repeat_texts(prompt, num_repeats)
    for i, (prompt, src, label) in enumerate(zip(new_prompt,source_texts , re_labels)):
    # 生成候选文本-
        hypos = yuce_generate(
            prompt, src
        )
        print("======预测结果=======")
        print(hypos)
        # 去除引号
        def normalize_text(text):
            return re.sub(r'[“”"\']', '', text)  # 去除中文引号、英文引号


        def normalize_triples(list_triple):
            return [normalize_text(item) for item in list_triple]


        hypos = normalize_triples(hypos)
        # 三元组解析
        generated_triples = _extract_triples(hypos)

        p,r,f1 =_calc_exact_scores(label,generated_triples)
        print(f"num:{i}|"
          f"precision: {p:.2f}| "
          f"recall: {r:.2f} | "
          f"f1: {f1:.2f} | ")
       # save_hypos(src, label, generated_triples, p, r, f1)
        all_p+=p
        all_r+=r
        all_f+f1
    mean_p=all_p/139
    mean_r=all_r/139
    mean_f1=all_f/139
    print(f"precision: {mean_p:.2f}| "
          f"recall: {mean_r:.2f} | "
          f"f1: {mean_f1:.2f} | ")