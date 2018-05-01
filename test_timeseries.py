import os, glob, pandas, numpy
import matplotlib.pyplot as plt
import seaborn
import site
site.addsitedir('..')
from timeseries import *
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
        # self.conn = sqlite3.connect(self.db_file)

    def tearDown(self):
        if os.path.isfile(self.db_file):
            os.remove(self.db_file)


    def test_str(self):
        ts = """TimeSeries(data=[ 1.00538072  1.01812632  1.04485803  1.06829273  1.09609764  1.12808833
  1.14593066], time=[ 15  30  60  90 120 150 180], feature="CTGF")"""
        self.assertEqual(TimeSeries(self.CTGF).__str__(), ts)

    def test_name(self):
        self.assertEqual(TimeSeries(self.CTGF).feature, 'CTGF')

    def test_Time(self):
        self.assertListEqual(list(TimeSeries(self.CTGF).time), [15, 30, 60, 90, 120, 150, 180])

    def test_to_table(self):
        ts = TimeSeries(self.CTGF)
        ts.to_db(self.db_file, 'microarray')

        with DB(self.db_file) as db:
            tables = db.tables()

        self.assertIn('microarray', tables)

    # def test(self):
    #     sql = """SELECT * FROM microarray"""
    #     with DB(self.db_file) as db:
    #         data = db.execute(sql)




class TestTimeSeriesGroup(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()

        self.data = self.data['TGFb'] / self.data['Control']

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

    def test_deliterm(self):
        tsg = TimeSeriesGroup(self.data)
        del tsg['CTGF']
        self.assertNotIn('CTGF', tsg.features)

    def test_to_db_new_table(self):
        tgf = TimeSeriesGroup(self.data)
        tgf.to_db(self.db_file, 'TestTable')
        with DB(self.db_file) as db:
            data = db.read_table('TestTable')

        self.assertEqual(data.shape, (227, 7))

    def test_to_db_add_to_existing(self):
        first_half = self.data.iloc[:20]
        second_half = self.data.iloc[20:40]
        tsg1 = TimeSeriesGroup(first_half)
        tsg2 = TimeSeriesGroup(second_half)
        tsg1.to_db(self.db_file, 'TestTable')
        tsg2.to_db(self.db_file, 'TestTable')

        with DB(self.db_file) as db:
            data = db.read_table('TestTable')

        self.assertEqual(data.shape, (40, 7))

    def tearDown(self):
        if os.path.isfile(self.db_file):
            os.remove(self.db_file)























