import os, glob, pandas, numpy
import matplotlib.pyplot as plt
import seaborn
from inout import DB
import sqlite3
import logging
from functools import reduce
logging.basicConfig()

LOG = logging.getLogger(__name__)






class TimeSeries(object):
    def __init__(self, values, time=None, feature=None):
        self.values = values
        self.time = time
        self.feature = feature

        if not isinstance(self.values, (pandas.DataFrame, pandas.Series, numpy.array, list)):
            raise TypeError('data shold be 1d list numpy.array, pandas series or 1d dataframe')

        if self.time is not None:
            if not isinstance(self.time, (pandas.Index, list, numpy.array)):
                raise TypeError('index arg shold be pandas.Index, list or numpy.array')

            if len(self.time) != len(self.values):
                raise ValueError('number of index shold match the dimensionality of the data')
            
        if self.time is None and isinstance(self.values, (pandas.DataFrame, pandas.Series)):
            self.time = numpy.array(self.values.index)

        if self.feature is None and isinstance(self.values, (pandas.DataFrame, pandas.Series)):
            self.feature = self.values.name

        if not isinstance(self.values, type(numpy.array)):
            self.values = numpy.array(self.values)

        if self.time is None:
            LOG.warning('time argument is None. Default time being used')
            self.time = range(len(self.values))

    def __str__(self):
        return """TimeSeries(data={}, time={}, feature="{}")""".format(self.values, self.time, self.feature)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))

    def to_db(self, dbfile, table):
        sql = """CREATE TABLE IF NOT EXISTS {} (
                id INTEGER PRIMARY KEY,
                feature TEXT,\n""".format(table)
        columns = reduce(lambda x, y: x + y, ['"{}" DOUBLE PRECISION,\n'.format(i) for i in self.time])
        columns = columns[:-2] + ');'
        sql = sql + columns

        with DB(dbfile) as db:
            db.execute(sql)

        sql = """INSERT OR IGNORE INTO {}(feature,""".format(table)
        sql = sql + reduce(lambda x, y: x + y, ['"{}",'.format(i) for i in self.time])[:-1] + ')'
        sql = sql + " values('{}', ".format(self.feature)
        sql += reduce(lambda x, y: x + y, ['{},'.format(i) for i in self.values])[:-1] + ");"

        with DB(dbfile) as db:
            db.execute(sql)

    # @staticmethod
    # def from_db(db_file, table):
    #     sql = """"""


class TimeSeriesGroup(object):
    def __init__(self, values, features=None, time=None):
        self.values = values
        self.features = features
        self.time = time

        if not isinstance(self.values, (numpy.ndarray, pandas.core.frame.DataFrame)):
            raise TypeError('values argument should be numpy.array 2d or pandas df')

        if not isinstance(self.values, numpy.ndarray):
            if type(self.values) == pandas.DataFrame:
                self.features = numpy.array(self.values.index)
                self.time = numpy.array(self.values.columns)
                self.values = self.values.as_matrix()

        else:
            if self.features is None:
                LOG.warning('No features specified. Features will be labelled with numbers')
                self.features = range(self.values.shape[0])

            if self.time is None:
                LOG.warning('No time has been specified. Time will increment linearly from 0.')
                self.time = range(self.values.shape[1])

        assert self.nfeat * self.ntime == self.values.shape[0] * self.values.shape[1]

    def __str__(self):
        return self.as_df().__str__()

    def __getitem__(self, feature):
        data = self.as_df()
        return TimeSeries(data.loc[feature])

    def __delitem__(self, feature):
        data = self.as_df()
        data = data.drop(feature, 0)
        new = TimeSeriesGroup(data)
        self.features = new.features
        self.values = new.values

    def as_df(self):
        return pandas.DataFrame(self.values, columns=self.time, index=self.features)

    @property
    def nfeat(self):
        return len(self.features)

    @property
    def ntime(self):
        return len(self.time)

    def append(self, ts):
        if not isinstance(ts, TimeSeries):
            raise TypeError('ts argument should be a TimeSeries')

        if self.time.all() != ts.time.all():
            raise ValueError('Time argument for "{}" must be same as the TimeSeriesGroup '
                             'you are entering it into.')
        features = numpy.append(self.features, ts.feature)
        values = numpy.vstack([self.values, ts.values])
        return TimeSeriesGroup(values=values, features=features, time=self.time)

    def plot(self, feature):
        pass

    def to_db(self, dbfile, table):
        """

        :param dbfile:
        :param table:
        :return:
        """
        if table not in DB(dbfile).tables():
            data = self.as_df()
            with DB(dbfile) as db:
                data.to_sql(name=table, con=db.conn, if_exists='fail', index_label='feature')
        else:

            s = ''
            vals = ''
            for i in self.time:
                s += '"{}",'.format(i)
                vals += '?, '
            sql = "INSERT OR REPLACE INTO " + table + "(feature," + s[:-1] + ")"
            sql += ' values(?, '
            sql += vals[:-2]
            sql += ');'

            data = self.as_df().reset_index()
            tup = [tuple(x) for x in data.values]

            with DB(dbfile) as db:
                try:
                    db.executemany(sql, tup)
                except sqlite3.Error as e:
                    print (e)































