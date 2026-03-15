import pandas as pd


def count_label_occurrences(file_path):
    # 读取TSV文件
    df = pd.read_csv(file_path, sep='\t')

    # 统计label列中包含"生产"的行数
    count = df['label'].str.contains('供应', na=False).sum()

    return count



# 示例使用
file_path = 'test.tsv'
count = count_label_occurrences(file_path)
print(f'"生产"在label列中出现的次数: {count}')
