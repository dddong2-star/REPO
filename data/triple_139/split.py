import  pandas as pd
from sklearn.model_selection import train_test_split
df =pd.read_csv("../triple_139/all.tsv", sep='\t')

train_df,temp_df=train_test_split(df,test_size=0.9)
val_df,test_df=train_test_split(temp_df,test_size=0.9)

train_df.to_csv("train.tsv",sep='\t',index=False)
test_df.to_csv("test.tsv",sep='\t',index=False)
val_df.to_csv("dev.tsv",sep='\t',index=False)