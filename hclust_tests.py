import unittest
import os, glob, pandas, numpy
import sqlite3
from matplotlib.figure import Figure
import site
site.addsitedir(r'..')
from clust import *
from scipy.stats import ttest_ind
from sklearn.cluster import AgglomerativeClustering


## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray'

class TestHClust(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()
        self.data = self.data['TGFb'] / self.data['Control']

    def test_cluster(self):
        tsg = TimeSeriesGroup(self.data.iloc[:20])
        c = HClust(tsg, n_clusters=5)
        clusts = c.clusters
        figs = []
        for i in clusts:
            ci = clusts[i]
            figs.append(ci.plot_centroid(label=i))

        [self.assertTrue(isinstance(i, Figure)) for i in figs]


    # def test_initialize(self):
    #     tsg = TimeSeriesGroup(self.data.iloc[:20])
    #     c = HClust(tsg)
    #     self.assertEqual(len(c.monitor), 20)
    #






if __name__ == '__main__':
    unittest.main()


































