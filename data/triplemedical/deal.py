import re
import os

def process_line(line):
    """
    清洗一行数据：
    - 分割 text 和 label
    - 清洗 text 字段中的特殊符号
    - 统一 label 的三元组格式
    """
    if not line.strip():
        return None

    try:
        # 假设是以 \t 分隔的 TSV 格式
        text_part, label_part = line.strip().split('\t', 1)
    except ValueError:
        print(f"❌ 分割失败（可能不是标准 TSV 格式）: {line[:50]}...")
        return None

    # 清洗 text 部分
    cleaned_text = re.sub(r'@.*?(?=。|\n|$)', '', text_part)         # 删除 @ 后的内容直到句号
    cleaned_text = re.sub(r'\[.*?\]\(.*?\)', '', cleaned_text)       # 删除 Markdown 链接
    cleaned_text = re.sub(r'###.*?$', '', cleaned_text)              # 删除 ### 开头的行
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()         # 统一空格

    # 清洗 label 部分
    cleaned_label = re.sub(r'$([^$]+)$', r'（\1）', label_part)     # 英文括号 → 中文括号
    cleaned_label = re.sub(r'"([^"]+)"', r'“\1”', cleaned_label)     # 英文引号 → 中文引号
    cleaned_label = re.sub(r"'([^']+)'", r'“\1”', cleaned_label)
    cleaned_label = re.sub(r',\s*', '，“', cleaned_label)            # 替换逗号
    cleaned_label = re.sub(r'\s*$', '”', cleaned_label)

    return f"{cleaned_text}\t{cleaned_label}"


def clean_dataset(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8-sig') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        line_count = 0
        skipped_lines = 0

        for line in fin:
            line_count += 1
            cleaned_line = process_line(line)
            if cleaned_line is None:
                skipped_lines += 1
                continue
            fout.write(cleaned_line + '\n')

    print(f"✅ 数据清洗完成：共处理 {line_count} 行，跳过 {skipped_lines} 行异常数据")


if __name__ == '__main__':
    input_file = "F:/Graduate/al/Opprompt/Opprompt(1)/data/triple/train.tsv"
    output_file = "F:/Graduate/al/Opprompt/Opprompt(1)/data/triple/cleaned_train.tsv"

    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在，请检查路径：{input_file}")
    else:
        clean_dataset(input_file, output_file)