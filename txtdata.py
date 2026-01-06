# def extract_id_title_abstract(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     with open(output_file, 'w', encoding='utf-8') as out_f:
#         for line in lines:
#             if '|t|' in line or '|a|' in line:
#                 out_f.write(line)
#
#     print(f"已提取标题和摘要到 {output_file}")
#
#
# if __name__ == '__main__':
#     extract_id_title_abstract(
#         r"F:\Graduate\al\Opprompt\Opprompt(1)\CDR_TrainingSet.PubTator.txt",
#         r"F:\Graduate\al\Opprompt\Opprompt(1)\CDR_TrainingSet.txt"
#     )

def extract_abstracts_to_tsv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 写入表头
        out_f.write("text\tlabel\n")
        for line in lines:
            if '|a|' in line:
                # 提取 |a| 后的内容
                abstract = line.split('|a|', 1)[-1].strip()
                out_f.write(f"{abstract}\t\n")  # label 列留空

    print(f"已提取摘要并保存到 {output_file}")


if __name__ == '__main__':
    #1
    # input_path = r"F:\Graduate\al\Opprompt\Opprompt(1)\CDR_TrainingSet.PubTator.txt"
    # output_path = r"F:\Graduate\al\Opprompt\Opprompt(1)\CDR_abstracts.tsv"
    # extract_abstracts_to_tsv(input_path, output_path)
    #2
    # import pandas as pd
    #
    # # 读取 TSV 文件
    # tsv_file = "CDR_abstracts.tsv"
    # df = pd.read_csv(tsv_file, sep='\t')
    #
    # # 保存为 Excel 文件
    # excel_file = "CDR_abstracts.xlsx"
    # df.to_excel(excel_file, index=False)
    #
    # print(f"已成功将 {tsv_file} 转换为 {excel_file}")

    #3
    import pandas as pd

    # 1. 读取Excel文件（默认第一个sheet）
    df = pd.read_excel("CDR_abstracts.xlsx")

    # 2. 只保留需要的两列：text 和 label
    df = df[['text', 'label']]

    # 3. 取前120条数据
    df_subset = df.head(120)

    # 4. 平均分为三份（各40条）
    train_df = df_subset.iloc[:40]
    test_df = df_subset.iloc[40:80]
    dev_df = df_subset.iloc[80:120]

    # 5. 保存为tsv文件（tab-separated values）
    train_df.to_csv("train.tsv", sep='\t', index=False)
    test_df.to_csv("test.tsv", sep='\t', index=False)
    dev_df.to_csv("dev.tsv", sep='\t', index=False)

    print("分割完成，已保存为 train.tsv、test.tsv 和 dev.tsv")