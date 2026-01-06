# import  pandas as pd
# from sklearn.model_selection import train_test_split
# df =pd.read_csv("all.tsv", sep='\t')
#
# train_df,temp_df=train_test_split(df,test_size=0.9)
# val_df,test_df=train_test_split(temp_df,test_size=0.9)
#
# train_df.to_csv("train.tsv",sep='\t',index=False)
# test_df.to_csv("test.tsv",sep='\t',index=False)
# val_df.to_csv("dev.tsv",sep='\t',index=False)


import pandas as pd

def clean_text(text):
    """清理单个字段：去空格、换行符、制表符"""
    if isinstance(text, str):
        return ' '.join(text.strip().replace('\n', ' ').replace('\r', '').split())
    return text

def clean_tsv(input_path, output_path):
    # 读取TSV文件
    df = pd.read_csv(input_path, sep='\t', on_bad_lines='skip', low_memory=False)

    # 对每一列进行清洗
    df = df.applymap(clean_text)

    # 保存为新TSV文件
    df.to_csv(output_path, sep='\t', index=False)
    print(f"已清洗并保存至 {output_path}")

# 示例调用
clean_tsv("dev.tsv", "dev.tsv")