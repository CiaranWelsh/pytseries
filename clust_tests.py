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
        entry = self.data.iloc[0]
        e = Element(entry.x, entry.y, entry.cost)
        self.assertEqual(e.__str__(), 'Element(x=CTGF, y=TGFBI, cost=0.000578187355748)')


    def test_element4(self):
        entry1 = self.data.iloc[0]
        entry2 = self.data.iloc[1]
        e1 = Element(entry1.x, entry1.y, entry1.cost)
        e2 = Element(entry2.x, entry2.y, entry2.cost)
        self.assertNotEqual(e1, e2)

    def test_element2(self):
        entry1 = self.data.iloc[0]
        entry2 = self.data.iloc[1]
        e1 = Element(entry1.x, entry1.y, entry1.cost)
        e2 = Element(entry2.x, entry2.y, entry2.cost)
        self.assertTrue(e1 < e2)

    def test_element3(self):
        entry1 = self.data.iloc[0]
        entry2 = self.data.iloc[1]
        e1 = Element(entry1.x, entry1.y, entry1.cost)
        e2 = Element(entry2.x, entry2.y, entry2.cost)
        self.assertTrue(e2 > e1)



class TestCluster(unittest.TestCase):
    def setUp(self):
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        assert os.path.isfile(self.db_file)

        self.conn = sqlite3.connect(self.db_file)


        sql = """SELECT * FROM dwt_data WHERE x = 'CTGF' ORDER BY cost"""
        self.data = pandas.read_sql(sql, self.conn)
        self.data = self.data[['x', 'y', 'cost']]

        self.elements = []
        for i in range(self.data.shape[0]):
            entry = self.data.iloc[i]
            self.elements.append(Element(x=entry.x, y=entry.y, cost=entry.cost))

    def test_min_works(self):
        self.assertTrue(min(self.elements).__str__(), 'Element(x=CTGF, y=TGFBI, cost=0.000578187355748)')

    # def test_getitem(self):
    #     c = Cluster(0, self.elements)
    #     ele = c[(None, 'CTPS1')][0]
    #     self.assertEqual(ele.__str__(), 'Element(x=CTGF, y=CTPS1, cost=0.000666666136016)')
    #
    def test_mean_intrasquare_distance(self):
        c = Cluster(0, self.elements)
        intra_mean_dist = c.intra_dist
        self.assertAlmostEqual(intra_mean_dist, 214.15262137522112)

    def test_mean_intercluster_square_difference(self):
        c1 = Cluster(1, self.elements[:10])
        c2 = Cluster(0, self.elements[10:20])
        c1_m = c1.inter_dist(c2)
        self.assertAlmostEqual(c1_m, 2.5621238850316306e-07 )

    def test_mean(self):
        c1 = Cluster(1, self.elements[:10])
        self.assertAlmostEqual(0.000948791612261, c1.mean())

    def test_gl(self):
        c1 = Cluster(1, self.elements[:10])
        c2 = Cluster(0, self.elements[10:20])
        self.assertTrue(c1 < c2)

class TestHClust(object):
    def setUp(self):
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        assert os.path.isfile(self.db_file)

        self.conn = sqlite3.connect(self.db_file)


        sql = """SELECT * FROM dwt_data WHERE x = 'CTGF' ORDER BY cost"""
        self.data = pandas.read_sql(sql, self.conn)
        self.data = self.data[['x', 'y', 'cost']]

        self.elements = []
        for i in range(self.data.shape[0]):
            entry = self.data.iloc[i]
            self.elements.append(Element(x=entry.x, y=entry.y, cost=entry.cost))

    def test(self):
        HClust(self.elements)

if __name__ == '__main__':
    unittest.main()



































