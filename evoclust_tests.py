import unittest
import site
site.addsitedir(r'..')
from timeseries.core import *

## folder to the microarray clustering
dire = r'/home/b3053674/Documents/timeseries/Microarray'



class TestElement(unittest.TestCase):
    def setUp(self):
        self.data_file = os.path.join(dire, 'MicroarrayDEGAgeravedData.xlsx')
        self.db_file = os.path.join(dire, 'microarray_dwt.db')

        self.data = pandas.read_excel(self.data_file, index_col=[0, 1]).transpose()

        self.data = self.data['TGFb'] / self.data['Control']

        self.CTGF = self.data.loc['CTGF']
        self.smad7 = self.data.loc['SMAD7']


    def test(self):
        tgs = TimeSeriesGroup(self.data.iloc[:45])
        print(tgs)


if __name__ == '__main__':
    unittest.main()








































