import re
import time
import pymysql
from openai import OpenAI
from typing import List, Tuple

IP = ""
MYSQLPWD = ''
DB = ''

def _extract_triples(texts: List[str]) -> List[List[Tuple]]:
    triples = []
    for text in texts:
        rule_triples = _rule_based_extraction(text)
        triples.append(rule_triples)
    return triples

def _rule_based_extraction(text: str) -> List[Tuple]:
    pattern = re.compile(r'[（(]\s*([^,，]+)\s*[,，]\s*([^,，]+)\s*[,，]\s*([^)）]+)\s*[）)]')
    matches = pattern.findall(text)
    return [tuple(match) for match in matches]

def format_triples_to_string(triples: List[Tuple]) -> str:
    """将三元组列表转为字符串格式"""
    return ''.join([f"（{t[0]}，{t[1]}，{t[2]}）" for t in triples])

def chatGPT_extract(formatted_template):
    res = []
    for i in range(3):  # 最多尝试3次
        try:
            client = OpenAI(
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
            r = completion.choices[0].message.content.strip()
            res.append(r)
            break
        except Exception as e:
            print(f"调用失败，重试第{i+1}次，错误：{e}")
            time.sleep(15)
    return res

def yuce_generate(prompt: str, source_text: str) -> List[str]:
    formatted_template = f"{prompt}:{source_text}"
    generated_texts = chatGPT_extract(formatted_template)
    return generated_texts

def getdata() -> List[Tuple[int, str]]:
    connection = pymysql.connect(host=IP, user='root', passwd=MYSQLPWD, db=DB)
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT id, text FROM medical_set LIMIT 369, 3001")
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return [(item['id'], item['text']) for item in data]

def save_label_to_db(data_with_labels):
    conn = None
    for attempt in range(3):
        try:
            conn = pymysql.connect(host=IP, user='root', passwd=MYSQLPWD, db=DB)
            break
        except pymysql.err.OperationalError:
            time.sleep(5)

    if not conn:
        print("数据库连接失败！")
        return

    cursor = conn.cursor()
    sql = """
    INSERT INTO medical_set (id, extract_label) VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE extract_label = VALUES(extract_label)
    """
    cursor.executemany(sql, data_with_labels)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"已成功写入 {len(data_with_labels)} 条记录")

if __name__ == '__main__':
    prompt = """根据下列文本中的内容，总结文本中的治疗关系、引发关系、检查关系、症状关系。以多个三元组的形式输出。没有此类关系则输出无。​
    关系描述如下:
    治疗关系:
    如果句子中描述了治疗手段 T 用于治疗疾病 D，可以抽取出 (T, "治疗", D)。
    引发关系:
    如果句子中描述了因素 F 导致疾病 D 发生，可以抽取出 (F, "引发", D)。
    检查关系:
    如果句子中描述了检查方法 M 用于检测疾病 D，可以抽取出 (M, "检查", D)。
    症状关系:
    如果句子中描述了疾病 D 会出现症状 S，可以抽取出 (D, "症状", S)。
    关系满足以下句式:
    T 治疗手段（用于 / 针对）治疗 D 疾病，则可以抽取出 (T, "治疗", D)。
    F 因素（导致 / 引起）D 疾病发生，则可以抽取出 (F, "引发", D)。
    M 检查方法（用于 / 用以）检测 D 疾病，则可以抽取出 (M, "检查", D)。
    D 疾病（会出现 / 表现为）S 症状，则可以抽取出 (D, "症状", S)。
    示例输入：肺炎患者通常会出现咳嗽、发热的症状，医生会通过胸部 X 光检查来诊断肺炎，常用抗生素治疗肺炎。
    输出:（抗生素，"治疗", 肺炎）,（肺炎，"引发", 咳嗽）,（肺炎，"引发", 发热）,（胸部 X 光，"检查", 肺炎）,（肺炎，"症状", 咳嗽）,（肺炎，"症状", 发热） """

    data = getdata()
    results = []

    BATCH_SIZE = 10

    for idx, (id, text) in enumerate(data):
        print(f"\n处理第 {idx + 1}/3000 条数据，ID: {id}")
        hypo = yuce_generate(prompt, text)
        print("原始响应：", hypo)
        # 提取三元组
        predicted_triples = _extract_triples(hypo)[0]
        label_str = format_triples_to_string(predicted_triples)

        print("提取结果：", label_str)
        results.append((id, label_str))

        if len(results) >= BATCH_SIZE:
            print(f"写入第 {idx + 1 - BATCH_SIZE + 1} 到第 {idx + 1} 条记录")
            save_label_to_db(results)
            results.clear()  # 清空缓存

    if len(results) > 0:
        print(f"写入最后的 {len(results)} 条记录")
        save_label_to_db(results)
        results.clear()

    print("所有数据处理完成并已写入数据库。")