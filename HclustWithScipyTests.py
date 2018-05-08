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
        tsg = TimeSeriesGroup(self.data.iloc[:100])
        c = HClustWithScipy(tsg, 'ward')
        fig = c.dendrogram(leaf_font_size=12,
                           distance_sort='ascending',
                           show_leaf_counts=True)
        plt.show()
        self.assertTrue(isinstance(fig, Figure))

    # def test2(self):
    #     tsg = TimeSeriesGroup(self.data.iloc[:50])
    #     c = HClustWithScipy(tsg, 'ward')
    #     print (c.get_clusters(get_leaves=True,
    #                           truncate_mode='lastp',
    #                           p=2))

if __name__ == '__main__':
    unittest.main()


































