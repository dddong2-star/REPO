from datetime import datetime

import pymysql
import json
import pymysql
IP = ""
MYSQLPWD = ''
DB = ''

connection = pymysql.connect(host=IP, user='root', passwd=MYSQLPWD, db=DB)

# 连接数据库
cursor = connection.cursor(pymysql.cursors.DictCursor)

# 执行查询
cursor.execute("SELECT * FROM hypos_logs_chatgpt limit 200")
data = cursor.fetchall()

# 关闭连接
cursor.close()
connection.close()
# 将 datetime 对象转换为字符串
for row in data:
    if 'timestamp' in row and isinstance(row['timestamp'], datetime):
        row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

# 将数据写入 JSON 文件，保持缩进
with open("hypos_logs_chatgpt_200.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("数据已成功导出到 hypos_logs_chatgpt_200.json")