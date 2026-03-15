# import  pandas as pd
# from sklearn.model_selection import train_test_split
# df =pd.read_csv("train.tsv", sep='\t')
#
# train_df,temp_df=train_test_split(df,test_size=0.9)
# val_df,test_df=train_test_split(temp_df,test_size=0.9)
#
# train_df.to_csv("train.tsv",sep='\t',index=False)
# test_df.to_csv("test.tsv",sep='\t',index=False)
# val_df.to_csv("dev.tsv",sep='\t',index=False)

with open('train.tsv', 'r', encoding='utf-8') as f:
    next(f)  # 跳过第一行（表头）
    line_count = sum(1 for line in f)

print(f"总共有 {line_count} 条数据")