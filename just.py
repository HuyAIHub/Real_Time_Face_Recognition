

# d = dict.fromkeys([],1)
# vcc = []
# check = ['a','b','c','a','b','c','m','a','b','c','a','b','c','m','a','b','c','a','b','c','m','a','a','a','a','a','a','a','a']
# for i in check:
#     d[i] = d[i] + 1 if i in d else 1
#     if d[i] >= 5 and i not in vcc:
#         vcc.append(i)
#     elif d[i] >= 5:
#         break
# print(d)
# print(vcc)


from glob_var import GlobVar,db_connect, minio_connect

labels = 'huydq46'
conn, cur = db_connect()
get_data = 'SELECT employee_id FROM face_recognition.employee WHERE name = \'' +  labels + '\';'
cur.execute(get_data)
employee_id = cur.fetchone()[0]
print(employee_id)