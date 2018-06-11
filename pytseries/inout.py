import os, glob, pandas, numpy
import matplotlib.pyplot as plt
import seaborn
import sqlite3
import logging


logging.basicConfig()

LOG = logging.getLogger(__name__)




class DB(object):
    def __init__(self, db_file):
        self.db_file = db_file

    def __enter__(self):
        self.conn = self.connect()
        self.cur = self.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.conn.close()

    def connect(self):
        return sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)

    def cursor(self):
        return self.conn.cursor()

    def execute(self, sql):
        try:
            self.cur.execute(sql)
            self.conn.commit()
            return self.cur.fetchall()
        except AttributeError as e:
            if str(e) == "'DB' object has no attribute 'cur'":
                self.conn = self.connect()
                self.cur = self.cursor()
                self.cur.execute(sql)
                self.conn.commit()
                return self.cur.fetchall()

            else:
                import sys
                exc_class, exc, tb = sys.exc_info()
                raise AttributeError(exc_class, exc, tb)

    def query(self, sql):
        return self.execute(sql)

    def executemany(self, sql, seq_of_parameters):
        try:
            self.cur.executemany(sql, seq_of_parameters)
            self.conn.commit()
            return self.cur.fetchall()

        except AttributeError as e:
            if str(e) == "'DB' object has no attribute 'cur'":
                self.conn = self.connect()
                self.cur = self.cursor()
                self.cur.executemany(sql, seq_of_parameters)
                self.conn.commit()
                return self.cur.fetchall()
            else:
                import sys
                exc_class, exc, tb = sys.exc_info()
                raise AttributeError(exc_class, exc, tb)

    def tables(self):
        sql = """SELECT name FROM sqlite_master WHERE type='table';"""
        query = self.execute(sql)
        try:
            return [i for i in zip(*query)][0]
        except Exception as e:
            if str(e) == 'list index out of range':
                return [i for i in zip(*query)]
            else:
                raise e

    def read_table(self, table):
        """

        :param table:
        :return:
        """
        if isinstance(table, int):
            table = '"{}"'.format(table)
        sql = """SELECT * FROM {};""".format(table)
        self.cur.execute(sql)
        names = [i[0] for i in self.cur.description]
        data = self.cur.fetchall()
        df = pandas.DataFrame(data, columns=names)
        df = df.set_index('feature')
        df = df.drop('ID', axis=1)
        return df













