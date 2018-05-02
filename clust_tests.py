import unittest
import os, glob, pandas, numpy
import sqlite3
import site
site.addsitedir(r'..')
from clust import *
from scipy.stats import ttest_ind



## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray'

class TestCTWClust(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()
        self.data = self.data['TGFb'] / self.data['Control']

    def test_choose_k(self):
        tsg = TimeSeriesGroup(self.data.iloc[:20])
        c = DTWClust(tsg)
        self.assertTrue(isinstance(c.choose_k(), numpy.int64))

    def test_make_random_population(self):
        tsg = TimeSeriesGroup(self.data.iloc[:20])
        c = DTWClust(tsg)
        pop_k3 = c.make_random_population(3)
        [self.assertTrue(isinstance(i, pandas.DataFrame) for i in pop_k3)]

    def test_make_initial_population(self):
        tsg = TimeSeriesGroup(self.data.iloc[:20])
        c = DTWClust(tsg)
        self.assertEqual(len(c.make_initial_population()), 20)

    # def test_eval_fitness(self):
    #     tsg = TimeSeriesGroup(self.data.iloc[:20])
    #     c = DTWClust(tsg)
    #     pop = c.make_initial_population()
    #     print(c.eval_fitness())

    def test_eval_fitness(self):
        tsg = TimeSeriesGroup(self.data.iloc[:20])
        c = DTWClust(tsg)
        pop = c.make_initial_population()
        fitness = c.eval_fitness()
        print (c.selection(fitness))



if __name__ == '__main__':
    unittest.main()


































