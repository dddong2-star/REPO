with open('all_zhidai.tsv', 'r', encoding='utf-8') as f:
    next(f)  # 跳过第一行（表头）
    line_count = sum(1 for line in f)

print(f"总共有 {line_count} 条数据")