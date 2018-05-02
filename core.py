import os, glob, pandas, numpy
import matplotlib.pyplot as plt
import seaborn
from inout import DB
import sqlite3
import logging
from functools import reduce
from collections import OrderedDict
# from dtw import DTW
logging.basicConfig()

LOG = logging.getLogger(__name__)


class TimeSeries(object):
    def __init__(self, values, time=None, feature=None):
        self.values = values
        self.time = time
        self._feature = feature

        if not isinstance(self.values, (pandas.DataFrame, pandas.Series, numpy.ndarray, list, dict)):
            raise TypeError('data should be 1d list, numpy.array, pandas series, or 1d dataframe, or dict[time:values]')

        if isinstance(self.values, dict):
            self.time = numpy.array(list(self.values.keys()))
            self.values = numpy.array(list(self.values.values()))

        if self.time is not None:
            if not isinstance(self.time, (pandas.Index, list, numpy.ndarray)):
                raise TypeError('index arg should be pandas.Index, list or numpy.array. Got "{}" instead'.format(type(self.time)))

            if len(self.time) != len(self.values):
                raise ValueError('number of index should match the dimensionality of the data')
            
        if self.time is None and isinstance(self.values, (pandas.DataFrame, pandas.Series)):
            self.time = numpy.array(self.values.index)

        if self._feature is None and isinstance(self.values, (pandas.DataFrame, pandas.Series)):
            self._feature = self.values.name

        if not isinstance(self.values, type(numpy.array)):
            self.values = numpy.array(self.values)

        if self.time is None:
            LOG.warning('time argument is None. Default time being used')
            self.time = range(len(self.values))

        # if self._feature is None:
        #     LOG.warning('"feature" argument is the name of your timeseries. While not '
        #                 'essential, you are reccommended to give your timeseries a feature name.')

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, name):
        self._feature = name

    def __str__(self):
        return """TimeSeries(data={}, time={}, feature="{}")""".format(list(self.values), list(self.time), self._feature)

    def __repr__(self):
        return self.__str__()

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

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        d = self.as_dict()
        if item not in d.keys():
            raise KeyError('key "{}" not in TimeSeries. Available keys are "{}"'.format(item, d.keys()))
        return d[item]

    def __setitem__(self, key, value):
        d = self.as_dict()
        d[key] = value
        self = self.to_dict()

    def __add__(self, other):
        if not isinstance(other, TimeSeries):
            raise TypeError('Cannot add TimeSeries with type "{}"'.format(type(other)))

        if other.time.all() != self.time.all():
            raise ValueError('Time vectors must be equal in order to perform numerical operations '
                             'on TimeSeries objects')

        new_vals = self.values + other.values
        return TimeSeries(values=new_vals, time=self.time)

    def __sub__(self, other):
        if not isinstance(other, TimeSeries):
            raise TypeError('Cannot add TimeSeries with type "{}"'.format(type(other)))

        if other.time.all() != self.time.all():
            raise ValueError('Time vectors must be equal in order to perform numerical operations '
                             'on TimeSeries objects')

        new_vals = self.values - other.values
        return TimeSeries(values=new_vals, time=self.time)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_vals = self.values * other

        elif isinstance(other, TimeSeries):
            if len(self) != len(other):
                raise ValueError('TimeSeries objects must be of '
                                 'the same length in order to apply mathematical '
                                 'operations.')
            new_vals = self.values * other.values

        else:
            raise TypeError('TimeSeries objects can be multiplied by other TimeSeries'
                            ' objects provided they are the same shape or by a scalar (int or float). '
                            'Got "{}" instead'.format(type(other)))

        return TimeSeries(values=new_vals, time=self.time)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            new_vals = self.values / other

        elif isinstance(other, TimeSeries):
            if len(self) != len(other):
                raise ValueError('TimeSeries objects must be of '
                                 'the same length in order to apply mathematical '
                                 'operations.')
            new_vals = self.values / other.values

        else:
            raise TypeError('TimeSeries objects can be multiplied by other TimeSeries'
                            ' objects provided they are the same shape or by a scalar (int or float). '
                            'Got "{}" instead'.format(type(other)))

        return TimeSeries(values=new_vals, time=self.time)

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            new_vals = self.values ** power

        else:
            raise TypeError('TimeSeries objects compute powers for int and floats but not other TimeSeries'
                            ' objects. Got "{}" instead'.format(type(power)))

        return TimeSeries(values=new_vals, time=self.time)

    def sum(self):
        return sum(self.values)


    def as_dict(self):
        return OrderedDict({self.time[i]: self.values[i] for i in range(len(self.values))})

    def from_dct(self, dct):
        time = dct.keys()
        vals = dct.values()
        return TimeSeries(time=time, values=vals, feature=dct.__name__)

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
        sql = sql + " values('{}', ".format(self._feature)
        sql += reduce(lambda x, y: x + y, ['{},'.format(i) for i in self.values])[:-1] + ");"

        with DB(dbfile) as db:
            db.execute(sql)

    def plot(self, **kwargs):
        seaborn.set_style('white')
        seaborn.set_context(context='talk', font_scale=2)
        fig = plt.figure()
        plt.plot(self.time, self.values, **kwargs)
        plt.xlabel('Time')
        plt.ylabel(self.feature)
        seaborn.despine(fig, top=True, right=True)
        return fig


class TimeSeriesGroup(object):
    def __init__(self, values, features=None, time=None):
        self.values = values
        self.features = features
        self.time = time

        if isinstance(self.values, pandas.Series):
            self.values = pandas.DataFrame(self.values).transpose()

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

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, feature):
        data = self.as_df()
        return TimeSeries(data.loc[feature])

    def __delitem__(self, feature):
        data = self.as_df()
        data = data.drop(feature, 0)
        new = TimeSeriesGroup(data)
        self.features = new.features
        self.values = new.values

    def __gt__(self, other):
        if not isinstance(other, TimeSeriesGroup):
            raise TypeError('cannot compare TimeSeriesGroup with "{}"'.format(type(other)))

        return self.intra_dtw_dist() > other.intra_dtw_dist()

    def __lt__(self, other):
        if not isinstance(other, TimeSeriesGroup):
            raise TypeError('cannot compare TimeSeriesGroup with "{}"'.format(type(other)))

        return self.intra_dtw_dist() < other.intra_dtw_dist()

    def __ge__(self, other):
        if not isinstance(other, TimeSeriesGroup):
            raise TypeError('cannot compare TimeSeriesGroup with "{}"'.format(type(other)))

        return self.intra_dtw_dist() >= other.intra_dtw_dist()

    def __le__(self, other):
        if not isinstance(other, TimeSeriesGroup):
            raise TypeError('cannot compare TimeSeriesGroup with "{}"'.format(type(other)))

        return self.intra_dtw_dist() <= other.intra_dtw_dist()

    def __len__(self):
        return self.values.shape[0]

    def as_df(self):
        return pandas.DataFrame(self.values, columns=self.time, index=self.features)

    @property
    def shape(self):
        return self.as_df().shape

    @property
    def iloc(self):
        return self.as_df().iloc

    @property
    def loc(self):
        return self.as_df().loc

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

    def plot(self, feature, legend=True, **kwargs):
        seaborn.set_context(context='talk', font_scale=2)
        seaborn.set_style('white')
        fig = plt.figure()

        if isinstance(feature, str):
            feature = [feature]

        for f in feature:
            if f not in self.features:
                raise ValueError('TimeSeriesGroup does not contain feature "{}". '
                                 'These features are available: "{}"'.format(f, self.features))
            plt.plot(self.time, self.as_df().loc[f], label=f,
                     marker='o', **kwargs)
        if legend:
            plt.legend(loc=(1, 0.5))
        plt.ylabel('AU (n={})'.format(self.nfeat))
        plt.xlabel('Time')

        seaborn.despine(fig, top=True, right=True)
        return fig

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
                    print(e)

    def do_statistic(self, stat):
        """

        :param stat: callable
        :return:
        """
        return TimeSeries(stat(self.values, 0), time=self.time, feature=stat.__class__.__name__)

    @property
    def mean(self):
        return TimeSeries(numpy.mean(self.values, 0), time=self.time, feature='mean')

    @property
    def sd(self):
        return TimeSeries(numpy.std(self.values, 0), time=self.time, feature='std')

    @property
    def var(self):
        return TimeSeries(numpy.var(self.values, 0), time=self.time, feature='var')

    @property
    def coeff_var(self):
        return TimeSeries(self.sd.values / self.mean.values, time=self.time, feature='std')

    @property
    def dtw_matrix(self):
        """
        get the profile which has the lowest DTW distance to all other
        profiles

        square matrix where forwards is same as backwards.
        Therefore if needed can reduce computation
        :return:
        """
        from dtw import DTW
        # matrix = numpy.ndarray((self.shape[0], self.shape[0]))
        matrix = pandas.DataFrame(numpy.zeros((self.shape[0], self.shape[0])))
        for i in range(self.shape[0]):
            for j in range(self.shape[0]):
                if i == j:
                    matrix.iloc[i, j] = numpy.nan
                else:
                    xfeat = self.features[i]
                    yfeat = self.features[j]
                    x = TimeSeries(self.loc[xfeat], time=self.time, feature=xfeat)
                    y = TimeSeries(self.loc[yfeat], time=self.time, feature=yfeat)
                    matrix.iloc[i, j] = DTW(x, y)
        matrix.index = self.features
        matrix.columns = self.features
        return matrix


    @property
    def dtw_cost_matrix(self):
        matrix = self.dtw_matrix
        for row in self.features:
            for col in self.features:
                try:
                    matrix.loc[row, col] = matrix.loc[row, col].cost
                except AttributeError as e:
                    if str(e) == "'float' object has no attribute 'cost'":
                        continue
                    else:
                        import sys
                        exc_class, exc, tb = sys.exc_info()
                        raise(exc_class, exc, tb)
        return matrix

    @property
    def center_profile(self):
        return self.dtw_cost_matrix.sum().idxmin()

    def warp_to_center_profile(self):
        """
        warp all other profiles to the center profile
        :return:
        """
        df_list = []
        for j in self.dtw_matrix:
            try:
                dtw = self.dtw_matrix.loc[self.center_profile, j]
                align = dtw.get_alignment()
                df = pandas.DataFrame(align)
                df_list.append(df)
        #         df['timex'] = align['timex']
        #         df['timey'] = align['timey']
        #         df['x'] = align['y']
        #         df['y'] = align['x']
        #         dct[j] = df
        #
            except AttributeError as e:
                if "'float' object has no attribute" in str(e):
                    continue
                else:
                    raise e
        return df_list
        # df = pandas.concat(dct, axis=1)
        # print (df)
        # for i in df.columns.get_level_values(0):
        #     print (i)
        #     plt.figure()
        #     d = df[i]
        #     plt.plot(d['timex'], d['x'], marker='o')
        #     plt.plot(d['timey'], d['y'], marker='o')
        #     plt.legend(loc='best')
        #
        # plt.show()


                    # print(df)
                # if j == numpy.nan:
                #     dct[i][j] = j

        # for i in range(matrix.shape[0]):
        #     for j in range(matrix.shape[1]):
        #         # if not numpy.isnan(self.dtw_matrix.iloc[i, j]):
        #         if i != j:
        #             dtw = self.dtw_matrix.iloc[i, j]
        #             matrix[i, j] = dtw.cost
        # return matrix

    def intra_eucl_dist(self):
        """
        objective function 1. Squared sum of all DTW distances
        in the cluster
        :return:
        """
        dct = OrderedDict()
        for i in range(self.values.shape[0]):
            profile_i = TimeSeries(self.values[i], time=self.time, feature=self.features[i])
            dct[i] = (self.mean - profile_i) ** 2
            dct[i] = dct[i].sum()

        df = pandas.DataFrame(dct, index=[0])
        return float(df.sum(axis=1))

    def inter_eucl_dict(self, other):
        if not isinstance(other, TimeSeriesGroup):
            raise TypeError('Argument "other" should be of type TimeSeriesGroup. '
                            'got "{}" instead'.format(type(other)))

        return ((self.mean - other.mean) ** 2).sum()

    def intra_dtw_dist(self):
        """
        sum of DTW(ci, cj) squared for all i and j in the set of profiles and i != j
        :return:
        """
        ##import into local space because of a conflict
        from dtw import DTW
        dct = OrderedDict()
        for i in range(self.values.shape[0]):
            profile_i = TimeSeries(self.values[i], time=self.time, feature=self.features[i])
            dct[i] = DTW(self.mean, profile_i).cost ** 2
            dct[i] = dct[i].sum()

        df = pandas.DataFrame(dct, index=[0])
        return float(df.sum(axis=1))

    def intra_dtw_dist_normalized_by_clustsize(self):
        """
        sum of DTW(ci, cj) squared for all i and j in the set of profiles and i != j
        :return:
        """
        ##import into local space because of a conflict
        from dtw import DTW
        dct = OrderedDict()
        for i in range(self.values.shape[0]):
            profile_i = TimeSeries(self.values[i], time=self.time, feature=self.features[i])
            dct[i] = (DTW(self.mean, profile_i).cost ** 2) / self.nfeat
            dct[i] = dct[i].sum()

        df = pandas.DataFrame(dct, index=[0])
        return float(df.sum(axis=1))

    def inter_dtw_dist(self, other):
        if not isinstance(other, TimeSeriesGroup):
            raise TypeError('Argument "other" should be of type TimeSeriesGroup. '
                            'got "{}" instead'.format(type(other)))

        ##import into local space because of a conflict
        from dtw import DTW

        return (DTW(self.mean, other.mean).cost ** 2).sum()

    def plot_centroid(self, **kwargs):
        seaborn.set_context('talk', font_scale=2)
        seaborn.set_style('white')
        center_data = self.loc[self.center_profile]
        fig = plt.figure()
        plt.errorbar(x=self.time, y=center_data.values,
                     yerr=self.sd.values, marker='o',
                     **kwargs)
        plt.ylabel('Centroid Profile')
        plt.xlabel('Time')
        seaborn.despine(fig, top=True, right=True)
        return fig




























