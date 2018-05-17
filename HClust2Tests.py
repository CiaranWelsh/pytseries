import unittest
import os, glob, pandas, numpy
import sqlite3
from matplotlib.figure import Figure
import site
site.addsitedir(r'..')
from clust_old import *
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

    def tearDown(self):
        if os.path.isfile(self.db_file):
            os.remove(self.db_file)

    def test_distance_matrix(self):
        tsg = TimeSeriesGroup(self.data.iloc[:5])
        # print(tsg)

        c = HClustDTW2(tsg)
        self.assertTrue(isinstance(c.dist_matrix, pandas.DataFrame))

    def test_pick_prototype(self):
        tsg = TimeSeriesGroup(self.data.iloc[:5])
        c = HClustDTW2(tsg)
        prototype = c.pick_prototype(tsg[['CTGF', 'CTGF.1', 'SERPINE1']])
        self.assertTrue(prototype == 'CTGF')

    def test_select_pair_to_merge(self):
        tsg = TimeSeriesGroup(self.data.iloc[10:15])
        c = HClustDTW2(tsg)
        to_merge = c.select_pair_to_merge(c.clusters)
        ans = "('CITED2', 'HES1', 1.1458823307183974)"
        self.assertEqual(str(to_merge), ans)

    def test_fit(self):
        tsg = TimeSeriesGroup(self.data.iloc[10:15])
        c = HClustDTW2(tsg)
        clu_his, df = c.fit()
        self.assertEqual(df.iloc[2].iloc[3], 4)

    def test_fit2(self):
        tsg = TimeSeriesGroup(self.data.iloc[10:15])
        c = HClustDTW2(tsg)
        clu_his, df = c.fit()
        print(clu_his[3]['CITED2'])
        self.assertAlmostEqual(clu_his[3]['CITED2'].loc['ID1.1', 15],
                               0.98961837164605271)

    def test_dendrogram(self):
        tsg = TimeSeriesGroup(self.data.iloc[:10])
        c = HClustDTW2(tsg)
        cluster_history, iterations = c.fit()
        for i in cluster_history[2]:
            # for cluster in cluster_history[i]:
            cij = cluster_history[2][i]#['clusters']
            cij.plot(cij.features)
        plt.show()

                # print (i, iterations[i][cluster])

        # c.dendrogram(Z)

        # self.assertEqual(Z.iloc[2].iloc[3], 4)

        #
        # def test(self):
        #     tsg = TimeSeriesGroup(self.data.iloc[:5])
        #     tsg.interpolate(num=15)
        #     fast = HClustDTW(tsg, fast=False)
        #     fast.fit()

        # fast.to_db()
        #     import time
        #     now_fast = time.time()
        #     fast_clusters, fast_merge_pairs = fast.fit()
        #     fast_end_time = time.time() - now_fast
        #
        #     now_slow = time.time()
        #     slow = HClustDTW(tsg, fast=False, radius=5)
        #     slow_clusters, slow_merge_pairs = slow.fit()
        #     slow_end_time = time.time() - now_slow
        #     print('slow:', slow_end_time, 'fast:', fast_end_time)

        # self.assertTrue(len(clusters) == 5)

        # def test_full_run(self):
        #     fname = os.path.join(dire, 'full_dataset.db')
        #     fname = os.path.join(dire, 'full_dataset_by_median.db')
        #     if os.path.isfile(fname):
        #         os.remove(fname)
        #
        #     tsg = TimeSeriesGroup(self.data)
        #     tsg.norm(inplace=True)
        #     # c = HClustDTW(tsg, db_file=fname)
        #     # c.fit()
        #
        #     table_id = 215
        #     with DB(fname) as db:
        #         print(db.tables())
        #         table = db.read_table(table_id)
        #
        #     for label, df in table.groupby('cluster'):
        #         df = df[df['cluster'] == label]
        #         df = df.drop('cluster', axis=1)
        #         tsg = TimeSeriesGroup(df)
        #         if len(tsg) < 20:
        #             fig = tsg.plot(tsg.features, legend=True)
        #         else:
        #             fig = tsg.plot(tsg.features, legend=False)
        #
        #         d = os.path.join(dire, str(table_id))
        #         if not os.path.isdir(d):
        #             os.makedirs(d)
        #         fname = os.path.join(d, str(label) + '.png')
        #         print (fname )
        #         fig.savefig(fname, dpi=300, bbox_inches='tight')

        # plt.show()

        # self.assertEqual(table3['cluster'].unique()[0], 0)



if '__main__' == __name__:
    unittest.main()

