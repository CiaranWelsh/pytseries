import unittest
from pytseries.dtw import DTW, FastDTW
from pytseries.core import *
from matplotlib.figure import Figure


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
        path_answer = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
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
        path_answer = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
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
        dire = r'/home/b3053674/Documents/pytseries/Microarray'
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()

        self.data = self.data['TGFb'] / self.data['Control']

        self.CTGF = self.data.loc['CTGF']
        self.smad7 = self.data.loc['SMAD7']

    def test_real_data(self):
        dtw = DTW(self.CTGF, self.smad7)
        self.assertTrue(isinstance(dtw, DTW))

    def test_from_df_list(self):
        l = [DTW(self.CTGF, self.smad7),
             DTW(self.smad7, self.CTGF)]
        df = pandas.DataFrame(l)
        self.assertTrue(isinstance(df.iloc[0, 0].cost, float))

    def test(self):
        CTGF = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        CTGF.interpolate(inplace=True)
        smad7.interpolate(inplace=True)
        CTGF.norm(inplace=True)
        smad7.norm(inplace=True)
        dtw = DTW(CTGF, smad7)

        fig = dtw.cost_plot()
        fig = dtw.plot()
        plt.show()
        self.assertTrue(isinstance(fig, Figure))

    def test2(self):
        dir = r'/home/b3053674/Documents/Microarray/GSS2265/python'
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        tsg1 = tsg1.norm()
        tsg1.interpolate('linear', 30)
        ts_l = tsg1.to_ts()

        ts = ts_l[0]
        ts2 = ts_l[2]
        fname_plot = os.path.join(dir, 'CTGFVsCTGF.png')
        fname_map = os.path.join(dir, 'CTGFVsCTGFMap.png')
        fname_normed_plot = os.path.join(dir, 'CTGFVsCTGFnormed.png')
        fname_normed_map = os.path.join(dir, 'CTGFVsCTGFnormedMap.png')
        fname_normed_interp_plot = os.path.join(dir, 'CTGFVsCTGFnormedInterp.png')
        fname_normed_interp_map = os.path.join(dir, 'CTGFVsCTGFnormedInterpMap.png')
        d = DTW(ts, ts2)
        fig = d.plot()
        fig.savefig(fname_normed_interp_plot, dpi=300, bbox_inches='tight')
        fig2 = d.cost_plot()
        fig2.savefig(fname_normed_interp_map, dpi=300, bbox_inches='tight')


class TestFastDTW(unittest.TestCase):
    def setUp(self):
        dire = r'/home/b3053674/Documents/pytseries/Microarray'
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()

        self.data = self.data['TGFb'] / self.data['Control']

        self.CTGF = self.data.loc['CTGF']
        self.smad7 = self.data.loc['SMAD7']

    def test_parse_fom_ts(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        fdtw = FastDTW(ctgf, smad7)
        self.assertTrue(isinstance(fdtw.x, TimeSeries))

    def test_parse_fom_np(self):
        ctgf = TimeSeries(self.CTGF).to_array()
        smad7 = TimeSeries(self.smad7).to_array()
        fdtw = FastDTW(ctgf, smad7)
        self.assertTrue(isinstance(fdtw.x, TimeSeries))

    def test_parse_fom_series(self):
        fdtw = FastDTW(self.CTGF, self.smad7)
        self.assertTrue(isinstance(fdtw.x, TimeSeries))

    def test_parse_fom_list(self):
        timex = list(self.CTGF.index)
        timey = list(self.smad7.index)
        x = list(self.CTGF.values)
        y = list(self.smad7.values)
        x = [i for i in zip(timex, x)]
        y = [i for i in zip(timey, y)]
        fdtw = FastDTW(x, y)
        self.assertTrue(isinstance(fdtw.x, TimeSeries))

    def test_inheritance(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        dtw = FastDTW(ctgf, smad7)
        self.assertTrue(isinstance(dtw.x, TimeSeries))

    def test2(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        dtw = FastDTW(ctgf, smad7)
        self.assertAlmostEqual(dtw.cost, 0.3958995851507465)
        self.assertListEqual(dtw.path,
                             [(0, 0), (1, 1), (2, 1), (3, 1),
                              (4, 2),
                              (5, 3), (6, 4), (6, 5), (6, 6)]
                             )

    def test_same_example_from_DTW(self):
        x = numpy.array([1, 1, 2, 3, 2, 0])
        y = numpy.array([0, 1, 1, 2, 3, 2, 1])
        dtw1 = DTW(x, y)
        dtw2 = FastDTW(x, y, radius=1)
        self.assertAlmostEqual(dtw1.cost, dtw2.cost)




if __name__ == '__main__':
    unittest.main()





















