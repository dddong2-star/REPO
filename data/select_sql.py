import  json
import re
import time
import pymysql
import csv

IP = ""
MYSQLPWD = ''
DB = ''
data=[['text','label']]
def query_SQL():
    conn = pymysql.connect(host=IP, user='root', passwd=MYSQLPWD, db=DB)
    cursor = conn.cursor()
    sql = "SELECT news_id,content,tuple  FROM  training_set  where tuple!='无'"
    cursor.execute(sql)
    rows = cursor.fetchall()
    print(len(rows))
    for row in rows:
        s = "SELECT title   FROM  key_news   where id=%s"
        param=(row[0])
        cursor.execute(s,param)
        r=cursor.fetchall()
        if len(r)==0:
            s = "SELECT title   FROM  concept_news   where id=%s"
            param = (row[0])
            cursor.execute(s, param)
            r = cursor.fetchall()
        title=r[0][0]
        content=row[1]
        text=title+content
        label=row[2]
        label = re.sub(r'^\[|\]$', '', label)
        item=[text,label]
        data.append(item)
    with open('triple/train.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerows(data)

if __name__ == '__main__':
    query_SQL()
