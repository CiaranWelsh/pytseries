import unittest
from dtw import DTW
import numpy, pandas
from core import *



class TestDTW(unittest.TestCase):
    def setUp(self):
        self.x = numpy.array([1, 1, 2, 3, 2, 0])
        self.y = numpy.array([0, 1, 1, 2, 3, 2, 1])

        self.xts = TimeSeries(self.x, feature='x')
        self.yts = TimeSeries(self.y, feature='y')


    def test_calculate(self):
        matrix = DTW(self.x, self.y).calculate_cost()[0]
        acc_cost = [[1.,   1.,   1.,   2.,   6.,   7.,   7.],
                    [2.,   1.,   1.,   2.,   6.,   7.,   7.],
                    [6.,   2.,   2.,   1.,   2.,   2.,   3.],
                    [15.,   6.,   6.,   2.,   1.,   2.,   6.],
                    [19.,   7.,   7.,   2.,   2.,   1.,   2.],
                    [19.,   8.,   8.,   6.,  11.,   5.,   2.]]
        acc_cost = numpy.matrix(acc_cost)
        self.assertEqual(matrix.all(), acc_cost.all())

    def test_find_best_path(self):
        path, cost = DTW(self.x, self.y).find_best_path()
        path_answer = [[5, 6], [4, 5], [3, 4], [2, 3], [1, 2], [1, 1], [0, 1], [0, 0]]
        cost_ans = 2.0
        self.assertListEqual(path_answer, path)
        self.assertEqual(cost_ans, cost)

    def test_calculate_fromts(self):
        matrix = DTW(self.xts, self.yts).calculate_cost()[0]
        acc_cost = [[1.,   1.,   1.,   2.,   6.,   7.,   7.],
                    [2.,   1.,   1.,   2.,   6.,   7.,   7.],
                    [6.,   2.,   2.,   1.,   2.,   2.,   3.],
                    [15.,   6.,   6.,   2.,   1.,   2.,   6.],
                    [19.,   7.,   7.,   2.,   2.,   1.,   2.],
                    [19.,   8.,   8.,   6.,  11.,   5.,   2.]]
        acc_cost = numpy.matrix(acc_cost)
        self.assertEqual(matrix.all(), acc_cost.all())

    def test_find_best_path_fromts(self):
        path, cost = DTW(self.xts, self.yts).find_best_path()
        path_answer = [[5, 6], [4, 5], [3, 4], [2, 3], [1, 2], [1, 1], [0, 1], [0, 0]]
        cost_ans = 2.0
        self.assertListEqual(path_answer, path)
        self.assertEqual(cost_ans, cost)

    def test_plot(self):
        dtw = DTW(self.xts, self.yts)
        fig = dtw.plot()
        from matplotlib.figure import Figure
        self.assertTrue(isinstance(fig, Figure))

    def test_distance_cost_plot(self):
        dtw = DTW(self.xts, self.yts)
        fig = dtw.cost_plot()
        from matplotlib.figure import Figure
        self.assertTrue(isinstance(fig, Figure))

class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        dire = r'/home/b3053674/Documents/timeseries/Microarray'
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()

        self.data = self.data['TGFb'] / self.data['Control']

        self.CTGF = self.data.loc['CTGF']
        self.smad7 = self.data.loc['SMAD7']

    def test_real_data(self):
        dtw = DTW(self.CTGF, self.smad7)
        self.assertTrue(isinstance(dtw, DTW))


if __name__ == '__main__':
    unittest.main()


















