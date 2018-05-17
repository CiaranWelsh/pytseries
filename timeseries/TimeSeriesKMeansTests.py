import unittest
import os
import site
site.addsitedir(r'..')
from timeseries.clust import *

## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray2'

class TimeSeriesKMeansTests(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MeanMicroarrayDEGS.csv')
        # self.db_file = os.path.join(dire, 'microarray_dwt.db')

        # self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()
        # self.data = self.data['TGFb'] / self.data['Control']

        self.data = pandas.read_csv(self.data_file, index_col=[0])
        self.data.columns = [int(i) for i in self.data.columns]

    def tearDown(self):
        pass

    def test_number_of_clusters(self):

        tsg = TimeSeriesGroup(self.data.iloc[:30])
        K = TimeSeriesKMeans(tsg, n_clusters=5, metric='dtw')
        self.assertEqual(K.n_clusters, 5)

    def test_convert_feature_labels(self):
        tsg = TimeSeriesGroup(self.data.iloc[:30])
        k = TimeSeriesKMeans(tsg, n_clusters=5, metric='dtw')
        labels = k.convert_labels_to_features()
        self.assertEqual(labels.loc['ADAMTS1'], 1)

    def test_plot(self):
        tsg = TimeSeriesGroup(self.data[:50])
        k = TimeSeriesKMeans(tsg, n_clusters=5, metric='dtw')
        figs = k.plot_clusters()
        plt.show()

    # def test_convert_format_centers(self):
    #     tsg = TimeSeriesGroup(self.data.iloc[:30])
    #     k = TimeSeriesKMeans(tsg, n_clusters=10, metric='dtw',
    #                          max_iter=200, n_init=20, max_iter_barycenter=300)
    #     labels_pickle = os.path.join(dire, 'labels.pickle')
    #     centers_pickle = os.path.join(dire, 'centers.pickle ')
    #
    #     figs_dir = os.path.join(dire, 'figs')
    #     if not os.path.isdir(figs_dir):
    #         os.makedirs(figs_dir)
    #
    #     figs = k.plot_clusters()
    #     for i in figs:
    #         fname = os.path.join(figs_dir, '{}.png'.format(i))
    #         figs[i].savefig(fname, dpi=300, bbox_inches='tight')
    #
    #     with open(labels_pickle, 'wb') as f:
    #         pickle.dump(k.labels, f)
    #
    #     with open(centers_pickle, 'wb') as f:
    #         pickle.dump(k.centers, f)
        # plt.show()
































if __name__ == '__main__':
    unittest.main()











