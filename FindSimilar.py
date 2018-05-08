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
        tsg = TimeSeriesGroup(self.data.iloc[:50])
        tsg = tsg.norm()
        tsg.interpolate(inplace=True, num=15)
        f = FindSimilar(tsg, 'CTGF', thresh=0.01)
        f.tsg.plot(f.result.features, legend=True)
        plt.show()

        # [i.plot() for i in f.dtw]
        # plt.show()


        # fig = tsg.plot(tsg.features, legend=True)
        # fname = os.path.join(dire, 'plot.png')
        # fig.savefig(fname, bbox_inches='tight', dpi=300)


if '__main__' == __name__:
    unittest.main()

