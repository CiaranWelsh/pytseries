import os, glob, pandas, numpy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn
# import site
# site.addsitedir('..')
from core import *
import unittest


## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray'



class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()

        self.data = self.data['TGFb'] / self.data['Control']

        self.CTGF = self.data.loc['CTGF']
        self.smad7 = self.data.loc['SMAD7']
        # self.conn = sqlite3.connect(self.db_file)

    def tearDown(self):
        if os.path.isfile(self.db_file):
            os.remove(self.db_file)


    def test_str(self):
        ts = """TimeSeries(data=[1.0053807204917966, 1.0181263186164504, 1.0448580330470829, 1.0682927301790315, 1.0960976392097332, 1.128088333880394, 1.1459306603252915], time=[15, 30, 60, 90, 120, 150, 180], feature="CTGF")"""
        self.assertEqual(TimeSeries(self.CTGF).__str__(), ts)

    def test_name(self):
        self.assertEqual(TimeSeries(self.CTGF)._feature, 'CTGF')

    def test_Time(self):
        self.assertListEqual(list(TimeSeries(self.CTGF).time), [15, 30, 60, 90, 120, 150, 180])

    def test_to_array(self):
        ts = TimeSeries(self.CTGF)
        ts_array = ts.to_array()
        self.assertTrue(ts_array.shape == (7, 2))

    def test_eucl_dist(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        ans = 0.0645769969607
        self.assertAlmostEqual(ctgf.eucl_dist(smad7), ans)


    def test_to_table(self):
        ts = TimeSeries(self.CTGF)
        ts.to_db(self.db_file, 'microarray')

        with DB(self.db_file) as db:
            tables = db.tables()

        self.assertIn('microarray', tables)

    # def test_as_dict(self):
    #     ts = TimeSeries(self.CTGF)
    #     self.assertTrue(isinstance(dict, ts.as_dict()))

    def test_to_series(self):
        ts = TimeSeries(self.CTGF)
        self.assertTrue(isinstance(ts.as_series(), pandas.Series))

    def test_feature_name(self):
        ts = TimeSeries(self.CTGF)
        ts.feature = 'new_ctgf'
        self.assertTrue(ts.feature == 'new_ctgf')

    def test_from_dct(self):
        ts = TimeSeries(self.CTGF)
        d = ts.as_dict()
        ts2 = TimeSeries(d)
        self.assertTrue(isinstance(ts2, TimeSeries))

    def test_indexing(self):
        """
        support dict type indexing. Select time point
        get value back
        :return:
        """
        ts = TimeSeries(self.CTGF)
        num = ts[15]
        self.assertAlmostEqual(num, 1.00538072049)

    def test_add_timeseries(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        ctgf1 = ctgf[15]
        smad71 = smad7[15]
        new = ctgf + smad7
        self.assertEqual(ctgf1 + smad71, new[15])

    def test_sub_timeseries(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        ctgf1 = ctgf[15]
        smad71 = smad7[15]
        new = ctgf - smad7
        self.assertEqual(ctgf1 - smad71, new[15])

    def test_mul_timeseries(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        ctgf1 = ctgf[15]
        smad71 = smad7[15]
        new = ctgf * smad7
        self.assertEqual(ctgf1 * smad71, new[15])

    def test_div_timeseries(self):
        ctgf = TimeSeries(self.CTGF)
        smad7 = TimeSeries(self.smad7)
        ctgf1 = ctgf[15]
        smad71 = smad7[15]
        new = ctgf / smad7
        self.assertEqual(ctgf1 / smad71, new[15])

    def test_mul_by_scalar(self):
        ctgf = TimeSeries(self.CTGF)
        scalar = 1.5
        val = ctgf[30] * scalar
        new_ts = ctgf*scalar
        self.assertEqual(val, new_ts[30])

    def test_plot(self):
        ctgf = TimeSeries(self.CTGF)
        fig = ctgf.plot(marker='o')
        self.assertEqual(type(fig), Figure)

    def test_ts_sum(self):
        ctgf = TimeSeries(self.CTGF)
        self.assertAlmostEqual(ctgf.sum(), 7.5067744357497794)

    def test_interpolate(self):
        ctgf = TimeSeries(self.CTGF)
        ctgf.interpolate(kind='linear', num=35, inplace=True)
        self.assertEqual(len(ctgf), 35)

    def test_interpolate2(self):
        ctgf = TimeSeries(self.CTGF)
        ctgf = ctgf.interpolate(kind='cubic', num=63)
        self.assertEqual(len(ctgf), 63)

    def test_summary_mean(self):
        ctgf = TimeSeries(self.CTGF)
        mean = ctgf.summary(numpy.mean)
        self.assertAlmostEqual(mean, 1.07239634796)

    def test_summary_min(self):
        ctgf = TimeSeries(self.CTGF)
        mean = ctgf.summary(numpy.min)
        self.assertAlmostEqual(mean, 1.0053807204917966)

    def test_norm_range(self):
        ctgf = TimeSeries(self.CTGF)
        ctgf.norm(inplace=True)
        ans = [0.0, 0.090683767917319122, 0.28087747744363217,
               0.44761320966600754, 0.64544260086774963,
               0.87305347504214714, 1.0]
        [self.assertAlmostEqual(ans[i], ctgf.values[i]) for i in range(len(ans))]


    # def test(self):
    #     plot_f = os.path.join(dire, 'CTGF_profile.png')
    #     interp_f = os.path.join(dire, 'CTGF_interped.png')
    #     normed_f = os.path.join(dire, 'CTGF_normed.png')
    #     interped_normed_f = os.path.join(dire, 'CTGF_interp_normed.png')
    #
    #     fnames = [plot_f, interp_f, normed_f, interped_normed_f]
    #     figs = []
    #
    #     print(self.CTGF)

        # ctgf = TimeSeries(self.CTGF)
        # ctgf.plot()
        #
        # ctgf = TimeSeries(self.CTGF)
        # ctgf = ctgf.interpolate('linear', num=30)
        # ctgf.plot()
        #
        # ctgf = TimeSeries(self.CTGF)
        # ctgf = ctgf.norm(method='minmax')
        # ctgf.plot()
        #
        # ctgf = TimeSeries(self.CTGF)
        # ctgf = ctgf.interpolate('linear', num=30)
        # ctgf = ctgf.norm(method='minmax')
        # ctgf.plot()
        #
        # for i in range(len(figs)):
        #     figs[i].savefig(fnames[i], dpi=300, bbox_inches='tight')
        #






        # smad7 = TimeSeries(self.smad7)


class TestTimeSeriesGroup(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()

        self.data = self.data['TGFb'] / self.data['Control']

    def tearDown(self):
        if os.path.isfile(self.db_file):
            os.remove(self.db_file)


    def test_from_df_and_time(self):
        tsg = TimeSeriesGroup(self.data)
        self.assertListEqual(list(tsg.time), [15, 30, 60, 90, 120, 150, 180])

    def test_append(self):
        data = self.data.iloc[:1]
        new = TimeSeries(self.data.iloc[10])
        tsg_old = TimeSeriesGroup(data)
        tsg_new = tsg_old.append(new)
        self.assertNotEqual(tsg_old.as_df().shape, tsg_new.as_df().shape)

    def test_getitem(self):
        tsg = TimeSeriesGroup(self.data)
        self.assertEqual(tsg['CTGF'].feature, 'CTGF')

    def test_loc(self):
        tsg = TimeSeriesGroup(self.data)
        print(tsg.loc['CTGF'])

    def test_deliterm(self):
        tsg = TimeSeriesGroup(self.data)
        del tsg['CTGF']
        self.assertNotIn('CTGF', tsg.features)

    def test_to_db_new_table(self):
        tgf = TimeSeriesGroup(self.data)
        tgf.to_db(self.db_file, 'TestTable')
        with DB(self.db_file) as db:
            data = db.read_table('TestTable')

        self.assertEqual(data.shape, (221, 8))

    def test_to_db_add_to_existing(self):
        first_half = self.data.iloc[:20]
        second_half = self.data.iloc[20:40]
        tsg1 = TimeSeriesGroup(first_half)
        tsg2 = TimeSeriesGroup(second_half)
        tsg1.to_db(self.db_file, 'TestTable')
        tsg2.to_db(self.db_file, 'TestTable')

        with DB(self.db_file) as db:
            data = db.read_table('TestTable')

        self.assertEqual(data.shape, (40, 8))

    def test_to_db_add_to_existing_with_cluster(self):
        first_half = self.data.iloc[:20]
        second_half = self.data.iloc[20:40]
        tsg1 = TimeSeriesGroup(first_half)
        tsg1.cluster = 1
        tsg2 = TimeSeriesGroup(second_half)
        tsg2.cluster = 2
        tsg1.to_db(self.db_file, 'TestTable')
        tsg2.to_db(self.db_file, 'TestTable')

        with DB(self.db_file) as db:
            data = db.read_table('TestTable')
        self.assertEqual([1, 2], list(set(data['cluster'])))


    def test_plot(self):
        tsg = TimeSeriesGroup(self.data)
        fig = tsg.plot(['CTGF', 'NET1'])
        self.assertTrue(isinstance(fig, Figure))

    def test_centroid(self):
        tsg = TimeSeriesGroup(self.data.iloc[:50])
        ans = [1.0039390119999707, 1.0359438639719978, 1.1199616125414489,
               1.1707734852034375, 1.1748721414325214,
               1.1828039333515956, 1.1887535775185674]
        self.assertListEqual(list(tsg.mean.values), ans)

    def test_intra_eucl_dist(self):
        tsg = TimeSeriesGroup(self.data.iloc[:10])
        self.assertAlmostEqual(tsg.intra_eucl_dist(), 3.4140343268673723)

    def test_inter_eucl_dist(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        tsg2 = TimeSeriesGroup(self.data.iloc[10:20])
        ans = 0.1816704557
        self.assertAlmostEqual(tsg1.inter_eucl_dict(tsg2), ans)

    # def test_intra_dwt_dist(self):
    #     tsg = TimeSeriesGroup(self.data.iloc[:10])
    #     self.assertAlmostEqual(tsg.intra_dtw_dist(), 1.5523596749962132)

    def test_inter_dwt_dist(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        tsg2 = TimeSeriesGroup(self.data.iloc[10:20])
        ans = 0.033031735452101314
        self.assertAlmostEqual(tsg1.inter_dtw_dist(tsg2), ans)

    # def test_gl(self):
    #     tsg1 = TimeSeriesGroup(self.data.iloc[:10])
    #     tsg2 = TimeSeriesGroup(self.data.iloc[10:20])
    #     self.assertTrue(tsg2 > tsg1)
    #
    # def test_lt(self):
    #     tsg1 = TimeSeriesGroup(self.data.iloc[:10])
    #     tsg2 = TimeSeriesGroup(self.data.iloc[10:20])
    #     self.assertTrue(tsg1 < tsg2)

    def calculate_dtw_matrix(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        self.assertTrue(isinstance(tsg1.dtw_matrix, pandas.DataFrame))

    def test_calculate_cost_matrix(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        self.assertTrue(isinstance(tsg1.dtw_cost_matrix, pandas.DataFrame))

    def test_center_profile(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        self.assertTrue(tsg1.center_profile == 'TRIB1')

    # def test(self):
    #     tsg1 = TimeSeriesGroup(self.data.iloc[:10])
    #     tsg1.warp_to_center_profile()

    def test_plot_centroid(self):
        tgs1 = TimeSeriesGroup(self.data.iloc[45:63])
        fig = tgs1.plot_centroid()
        self.assertTrue(isinstance(fig, Figure))

    def test_from_timecourse(self):
        ts = TimeSeries(self.data.loc['SMAD7'])
        tgs = TimeSeriesGroup(ts)
        self.assertTrue(isinstance(tgs, TimeSeriesGroup))

    def test_from_list_of_timecourses(self):
        ts1 = TimeSeries(self.data.iloc[4])
        ts2 = TimeSeries(self.data.iloc[6])
        tsg = TimeSeriesGroup([ts1, ts2])
        self.assertEqual(2, len(tsg))

    def test_to_ts(self):
        ts1 = TimeSeriesGroup(self.data.iloc[:10])
        self.assertEqual(len(ts1.to_ts()), 10)

    def test_norm(self):
        ts1 = TimeSeriesGroup(self.data.iloc[:10])
        ts1.norm()
        ## make list of ts
        l = ts1.to_ts()
        for i in l:
            self.assertTrue(0 in i)
            self.assertTrue(1 in i)

    def test_contains(self):
        ts1 = TimeSeriesGroup(self.data.iloc[:10])
        self.assertTrue('CTGF' in ts1)

    def test_delitem(self):
        tsg = TimeSeriesGroup(self.data)
        del tsg['CTGF']
        self.assertFalse('CTGF' in tsg)

    def test_get_list(self):
        tsg = TimeSeriesGroup(self.data)
        self.assertTrue (isinstance(tsg[['CTGF', 'SMAD7']], TimeSeriesGroup))

    def test_interpolate(self):
        tsg = TimeSeriesGroup(self.data)
        tsg = tsg.interpolate(num=86)
        self.assertEqual(86, tsg.shape[1])

    def test_to_singleton(self):
        tsg = TimeSeriesGroup(self.data)
        s = tsg.to_singleton()
        self.assertTrue(isinstance(s[0], TimeSeriesGroup))

    def test_merge(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        tsg2 = TimeSeriesGroup(self.data.iloc[10:20])
        tsg =tsg1.concat(tsg2)
        self.assertEqual(tsg.shape[0], 20)

    def test_sort(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        tsg1.sort(by='max')

    def test_max(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        ts_l = tsg1.to_ts()
        ts = ts_l[0]
        self.assertTrue(str(ts.max()) == '(180, 1.1459306603252915)')

    def test_eucl_dist_matrix(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        ans = 0.000651326459412
        m = tsg1.eucl_dist_matrix()
        self.assertAlmostEqual(m.loc['CTGF', 'CTGF.1'], 0.000651326459412)

    def test_centroid_by_eucl(self):
        tsg1 = TimeSeriesGroup(self.data.iloc[:10])
        self.assertTrue(tsg1.centroid_by_eucl.feature == 'TRIB1')

    def test(self):
        time_warp_interp = os.path.join(dire, 'time_warp_vdr_vegfa_plot.png')
        time_warp_cost_interp = os.path.join(dire, 'time_warp_vdr_vegfa_cost_plot.png')
        normed_f = os.path.join(dire, 'tsg_normed_profiles.png')
        interped_normed_f = os.path.join(dire, 'tsg.normed_interped_profiels.png')

        # fnames = [ctgf_repeat_plots)#, interp_f, normed_f, interped_normed_f]
        # figs = []

        tsg = TimeSeriesGroup(self.data)
        # print(tsg.features)
        tsg = tsg[['DUSP6', 'VEGFA']]
        tsg = tsg.interpolate(num=300)
        x = tsg.to_ts()[0]
        y = tsg.to_ts()[1]

        from dtw import DTW
        dtw = DTW(x, y)
        fig1 = dtw.plot()
        # fig1.savefig(time_warp_interp, dpi=300, bbox_inches='tight')


        fig2 = dtw.cost_plot()
        # fig2.savefig(time_warp_cost_interp, dpi=300, bbox_inches='tight')

        plt.show()
        print(dtw.path)
        print(dtw.cost)
        # # plt.show()


        # fig3.savefig(time_warp_cost, dpi=300, bbox_inches='tight')


        # smad7_junb = tsg[['SMAD7', 'JUNB']]
        # # smad7_junb.interpolate(num=)
        # smad7_junb.norm(inplace=True)
        # # smad7_junb.plot(smad7_junb.features)
        #
        # from dtw import DTW
        # dtw = DTW(smad7_junb['SMAD7'], smad7_junb['JUNB'])
        # # dtw.plot()
        # dtw.cost_plot()
        # plt.show()














