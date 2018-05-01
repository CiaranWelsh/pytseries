import unittest
import os, glob, pandas, numpy
import sqlite3
import site
site.addsitedir(r'..')
from clust import *
from scipy.stats import ttest_ind



## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray'

class TestElement(unittest.TestCase):
    def setUp(self):
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        assert os.path.isfile(self.db_file)

        self.conn = sqlite3.connect(self.db_file)


        sql = """SELECT * FROM dwt_data WHERE x = 'CTGF' ORDER BY cost"""
        self.data = pandas.read_sql(sql, self.conn)
        self.data = self.data[['x', 'y', 'cost']]

    def test_element1(self):
        pass

if __name__ == '__main__':
    unittest.main()



































