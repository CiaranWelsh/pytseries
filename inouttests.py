import os, glob, pandas, numpy
import matplotlib.pyplot as plt
import seaborn
import site
site.addsitedir('..')
from inout import *
import unittest


## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray'



class TestDB(unittest.TestCase):
    def setUp(self):
        self.db_file = os.path.join(dire, 'dbtest.db')

    def test_execute_sql(self):
        sql = """CREATE TABLE IF NOT EXISTS new_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                "15" DOUBLE PRECISION,
                "30" DOUBLE PRECISION,
                "60" DOUBLE PRECISION,
                "90" DOUBLE PRECISION,
                "120" DOUBLE PRECISION,
                "150" DOUBLE PRECISION,
                "180" DOUBLE PRECISION);"""

        with DB(self.db_file) as db:
            db.execute(sql)

        with DB(self.db_file) as db:
            tables = db.tables()

        self.assertTrue('new_table' == tables[0][0])




if __name__ == '__main__':
    unittest.main()
