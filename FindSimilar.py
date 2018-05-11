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
        self.tsg = TimeSeriesGroup(self.data)

    def test_x(self):
        f = FindSimilar(self.tsg, 'CTGF')
        self.assertTrue(isinstance(f.x, TimeSeries))

    def test_result(self):
        f = FindSimilar(self.tsg, 'CTGF')
        self.assertTrue(isinstance(f.result, TimeSeriesGroup))

    def test_dtw(self):
        fname = os.path.join(dire, 'FN1_find_similar.png')
        tsg = TimeSeriesGroup(self.data)
        tsg.interpolate(inplace=True, num=30)
        # print(tsg.features)
        f = FindSimilar(tsg, 'FN1', thresh=0.2)
        fig = f.tsg.plot(f.result.features, legend=True)
        fig.savefig(fname, dpi=300, bbox_inches='tight')

        # [i.plot() for i in f.dtw]
        plt.show()


        # fig = tsg.plot(tsg.features, legend=True)
        # fname = os.path.join(dire, 'plot.png')
        # fig.savefig(fname, bbox_inches='tight', dpi=300)


if '__main__' == __name__:
    unittest.main()

