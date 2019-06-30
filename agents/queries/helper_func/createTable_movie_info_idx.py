import psycopg2
try:
    #conn = psycopg2.connect(host="localhost", database="imdbload", user="postgres", password="admin")
    #conn = psycopg2.connect(host="localhost", database="imdbload", user="docker", password="docker")
    conn = psycopg2.connect(host="localhost", database="imdbload", user="postgres", password="docker")
except:
    print("I am unable to connect to the database")
# print(query)
cursor = conn.cursor()
#cursor.execute("""CREATE TABLE movie_info_idx AS SELECT * FROM movie_info;""")
cursor.execute("""SELECT * FROM movie_info LIMIT 1;""")
rows = cursor.fetchall()
print(rows)
