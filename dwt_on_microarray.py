import numpy
import os, glob, pandas
import matplotlib.pyplot as plt
import seaborn
import site
site.addsitedir('..')
from dtw import DTW
import itertools
import sqlite3
import json
import pickle


dire = r'C:\Users\Ciaran\Documents\DTW\Microarray'

db_file = os.path.join(dire, 'microarray_dwt.db')

if os.path.isfile(db_file):
    os.remove(db_file)

def connect(db_file):
    try:
        conn = sqlite3.Connection(db_file)

    except Exception as e:
        print e
    return conn

db = connect(db_file)

cur = db.cursor()

## create table
cur.execute("""CREATE TABLE IF NOT EXISTS dwt_data (
id integer PRIMARY KEY  AUTOINCREMENT,
x_gene TEXT, 
y_gene TEXT,
x_index TEXT,
y_index TEXT,
x TEXT,
y TEXT,
cost DOUBLE PRECISION,
path TEXT
);""")

db.commit()

# query = db.execute("SELECT * FROM sqlite_master WHERE type='table';")
# print query.fetchall()
#
#
# ## read data
df = pandas.read_excel(os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx'), index_col=[0, 1])

df = df.transpose()

df = df['TGFb'] / df['Control']

# put all other information in db as well so that plots can be produced from reading it


perm = [i for i in itertools.combinations(list(df.index), 2)]

DOALL = True
if DOALL:

    for x_gene, y_gene in perm:
        dwt = DTW(df.loc[x_gene], df.loc[y_gene])
        path = pickle.dumps(dwt.path)


        db.execute("""INSERT INTO dwt_data ( x, y, cost, path)
        VALUES ('{}', '{}', {}, '{}')""".format(
            x_gene, y_gene, dwt.cost, path
        )
        )
    db.commit()



# dtw1pair()































