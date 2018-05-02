import unittest
import os, glob, pandas, numpy
import sqlite3
import site
site.addsitedir(r'..')
from clust import *
from scipy.stats import ttest_ind



## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray'

class TestMonitor(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()
        self.data = self.data['TGFb'] / self.data['Control']

    def test_monitor_dict(self):
        m = Monitor()
        m[12] = 'foot'
        d = {12: 'foot'}
        self.assertEqual(str(d), str(m))







if __name__ == '__main__':
    unittest.main()


































