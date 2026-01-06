import pandas as pd
import openai
import time

# 设置 OpenAI API 密钥


# 读取 CSV
from openai import OpenAI

df = pd.read_csv("../original/all_zhidai.tsv", sep='\t')
df_label=df.label.tolist()
# 创建一个空列表来存储改写后的文本
rewritten_texts = []

# 遍历 text 列
for i, text in enumerate(df['text']):
    for i in range(1, 3):
        try:
            client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                base_url="https://api.xty.app/v1",
                api_key="sk-xxx",
            )
            completion = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': '请你帮我改写以下文本，进行指代替换,，将代词替换成全称，如公司替换成公司名称，客户替换为客户公司名称，只返回修改后的文本，不返回其他内容'},
                    {'role': 'user', 'content': text}
                ],
                temperature=0.5,
            )
            print(completion.choices[0].message.content)
            r = completion.choices[0].message.content
            rewritten_texts.append(r)
        except Exception as e:
            print(f"错误信息：{e}")
            time.sleep(15)
        else:
            break

# 替换原 text 列
df['text'] = rewritten_texts
df['label']=df_label
# 保存回原文件
df.to_csv('all_zhidai.tsv', sep='\t', index=False, encoding='utf-8')

print("✅ 改写完成，文件已保存！")
