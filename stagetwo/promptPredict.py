import re
import time
from typing import Optional, List

import requests
from openai import OpenAI


class PromptPredict:

    def __init__(
        self
    ):
        # self.tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)
        #                                                #pad_token=pad_token, trust_remote_code=True)
        # self.generator = pipeline("text-generation",
        #                           model=model,
        #                           tokenizer=self.tokenizer,
        #                           device=device_id,
        #                           trust_remote_code=True)
        #
        # self.end_punct = end_punct
        # self.control_output_length = control_output_length
        self.template="{prompt}输入：{sentence_1}"

    def yuce_generate(
            self,
            prompt: str,
            source_text: str,
            num_samples: int,
            ** kwargs
            ) -> List[str]:
            formatted_template = self.template.format(prompt=prompt,
                                                      sentence_1=source_text)
            # src_len = len(self.tokenizer(source_text)['input_ids'])
            #
            # pad_token_id = self.tokenizer.pad_token_id
            # vocab_size = self.tokenizer.vocab_size

            # 使用 assert 语句检查 pad_token_id 是否在有效范围内
            # assert 0 <= pad_token_id < vocab_size, f"pad_token_id ({pad_token_id}) is out of range. It should be between 0 and {vocab_size - 1}."
            #deepseek生成结果
            #generated_texts = deepseek_extract(formatted_template)
            generated_texts=chatGPT_extract(formatted_template)
            return generated_texts

    def val_generate(
            self,
            prompt: str,
            source_text: str,
            ** kwargs
            ) -> List[str]:
            formatted_template = self.template.format(prompt=prompt,
                                                      sentence_1=source_text)
            # src_len = len(self.tokenizer(source_text)['input_ids'])
            #
            # pad_token_id = self.tokenizer.pad_token_id
            # vocab_size = self.tokenizer.vocab_size

            # 使用 assert 语句检查 pad_token_id 是否在有效范围内
            # assert 0 <= pad_token_id < vocab_size, f"pad_token_id ({pad_token_id}) is out of range. It should be between 0 and {vocab_size - 1}."
            #deepseek生成结果
            # generated_texts = deepseek_extract(formatted_template)
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




def deepseek_extract(formatted_template):
    result = []

    for i in range(1, 3):
        try:
            res = requests.post(
                url="http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:1.5b",
                    "prompt": formatted_template,
                    "stream": False
                }
            )
            r = res.json()["response"]
            cleaned_text = re.sub(r"<think>.*?</think>", "", r, flags=re.DOTALL)
            cleaned_text = cleaned_text.replace(" ", "")
            result.append(cleaned_text)
        except Exception as e:
            print(f"错误信息：{e}")
            time.sleep(5)
        else:
            break
    return result
