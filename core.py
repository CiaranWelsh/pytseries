import os, glob, pandas, numpy
import matplotlib.pyplot as plt
import seaborn
from inout import DB
import sqlite3
import logging
from functools import reduce
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from copy import deepcopy
from multiprocessing import Pool, cpu_count

from itertools import combinations
# from dtw import DTW
logging.basicConfig()

LOG = logging.getLogger(__name__)


class TimeSeries(object):
    def __init__(self, values, time=None, feature=None,
                 time_unit='min', feature_unit='AU',
                 ):
        self.values = values
        self.time = time
        self._feature = feature
        self.time_unit = time_unit
        self.feature_unit = feature_unit

        if not isinstance(self.values, (pandas.DataFrame, pandas.Series, numpy.ndarray, list, dict)):
            raise TypeError('data should be 1d list, numpy.array, pandas series, or 1d dataframe, or dict[time:values]. Got "{}"'.format(type(self.values)))

        if isinstance(self.values, dict):
            self.time = numpy.array(list(self.values.keys()))
            self.values = numpy.array(list(self.values.values()))

        if self.time is not None:
            if not isinstance(self.time, (pandas.Index, list, numpy.ndarray)):
                raise TypeError('index arg should be pandas.Index, list or numpy.array. Got "{}" instead'.format(type(self.time)))

            if len(self.time) != len(self.values):
                raise ValueError('number of index should match the dimensionality of the data')

        if self.time is None and isinstance(self.values, (pandas.Series)):
            self.time = numpy.array(self.values.index)

        if self.time is None and isinstance(self.values, (pandas.DataFrame)):
            self.time = numpy.array(self.values.columns)

        if self._feature is None and isinstance(self.values, pandas.Series):
            self._feature = self.values.name

        if self._feature is None and isinstance(self.values, pandas.DataFrame):
            self._feature = self.values.index

        if not isinstance(self.values, type(numpy.array)):
            self.values = numpy.array(self.values)

        if self.time is None:
            LOG.warning('time argument is None. Default time being used')
            self.time = range(len(self.values))

        # ## if interpolation
        # if self.interp_kind is not None:
        #     self.time, self.values = self.interpolate()

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, name):
        self._feature = name

    def interpolate(self, kind='linear', num=100, inplace=False):

        f = interp1d(self.time, self.values, kind=kind)
        x = numpy.linspace(self.time[0], self.time[-1], num=num)
        args = deepcopy(self.__dict__)
        del args['_feature']
        args['values'] = f(x)
        args['time'] = x
        ts = TimeSeries(**args, feature=self.feature)
        if inplace:
            self.__dict__ = ts.__dict__

        return ts

    def dydt(self):
        pass

    def max(self):
        max = numpy.max(self.values)
        max_idx = numpy.argmax(self.values)
        return self.time[max_idx], max

    def min(self):
        min = numpy.min(self.values)
        min_idx = numpy.argmin(self.values)
        return self.time[min_idx], min

    def eucl_dist(self, other):
        if not isinstance(other, TimeSeries):
            raise TypeError('other should be of type TimeSeries. Got "{}"'.format(type(TimeSeries)))
        l = []
        for i, j in zip(self.values, other.values):
            l.append((i - j) ** 2)
        return sum(l)

    def norm(self, method='minmax', inplace=False):
        result = {}
        if method == 'minmax':
            for t, v in zip(self.time, self.values):
                ts_max = self.summary(numpy.max)
                ts_min = self.summary(numpy.min)
                result[t] = (v - ts_min) / (ts_max - ts_min)

        args = deepcopy(self.__dict__)
        del args['_feature']
        time, values = zip(*result.items())
        args['time'] = list(time)
        args['values'] = list(values)
        ts = TimeSeries(**args, feature=self.feature)

        if inplace:
            self.__dict__ = ts.__dict__

        return ts

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

    def __delitem__(self, key, value):
        d = self.as_dict()
        print(d)
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

    def __contains__(self, item):
        return self.values.__contains__(item)

    def sum(self):
        return sum(self.values)

    def as_dict(self):
        return OrderedDict({self.time[i]: self.values[i] for i in range(len(self.values))})

    def summary(self, stat=numpy.mean):
        """

        :param stat: callable. default = numpy.mean
        :return:
        """
        return stat(self.as_series())

    def time_summary(self, stat=None):
        """

        :param stat: callable. default=None
        :return:
        """
        if stat is None:
            return self
        else:
            return stat(list(self.as_series().index))

    def as_series(self):
        return pandas.Series(self.values, index=self.time, name=self.feature)

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
        if kwargs.get('marker') is None:
            marker = 'o'
        else:
            marker = kwargs.pop('marker')

        seaborn.set_style('white')
        seaborn.set_context(context='talk', font_scale=2)
        fig = plt.figure()
        plt.plot(self.time, self.values, marker=marker, **kwargs)
        plt.xlabel('Time ({})'.format(self.time_unit))
        plt.ylabel(self.feature + ' ({})'.format(self.feature_unit))
        seaborn.despine(fig, top=True, right=True)
        return fig

    def to_array(self):
        """
        output as 2d numpy array
        :return:
        """
        return numpy.array([i for i in zip(self.time, self.values)])


class TimeSeriesGroup(object):
    def __init__(self, values, features=None, time=None,
                 cluster=numpy.nan, meta=None, err=None):
        self.values = values
        self.features = features
        self.time = time
        self.meta = meta
        self.err = err
        self._cluster = cluster

        if isinstance(self.values, pandas.Series):
            self.values = pandas.DataFrame(self.values).transpose()

        if not isinstance(self.values, (numpy.ndarray, pandas.DataFrame, list,
                                        TimeSeries, pandas.Series)):
            raise TypeError('Invalid type. Got "{}"'.format(type(self.values)))

        if isinstance(self.values, TimeSeries):
            time_series = self.values
            self.values = numpy.matrix(time_series.values)
            self.features = [time_series.feature]
            assert isinstance(self.features, list)
            self.time = time_series.time

        ## unpack a list of TimeSeries
        if isinstance(self.values, list):
            all_time_series = True
            for i in self.values:
                if not isinstance(i, TimeSeries):
                    all_time_series = False

            if all_time_series:
                ## package into list of series and recall the innit
                df = pandas.concat([i.as_series() for i in self.values], axis=1).transpose()
                tgs = TimeSeriesGroup(df)
                ## unpack variables into rightful place
                self.values = tgs.values
                self.features = tgs.features
                self.time = tgs.time


        if not isinstance(self.values, numpy.ndarray):
            if type(self.values) == pandas.DataFrame or type(self.values) == pandas.Series:
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

        ## make sure features are unique
        if len(self.features) != len(set(self.features)):
            raise ValueError('There are duplicated features in '
                             'your data. Please make feature IDs '
                             'unique.')
        ## for meta df and err df, make sure the
        ## index are the same as features

        if self.meta is not None:
            for i in list(self.meta.index):
                if i not in self.features:
                    raise ValueError('Feature "{}" from meta data frame'
                                     ' is not an existing feature. There are your '
                                     'existing features "{}"'.format(i, self.features))

        if self.err is not None:
            for i in list(self.err.index):
                if i not in self.features:
                    raise ValueError('Feature "{}" from err data frame'
                                     ' is not an existing feature. There are your '
                                     'existing features "{}"'.format(i, self.features))

    def __str__(self):
        return self.as_df().__str__()

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, feature):
        data = self.as_df()
        if isinstance(feature, str):
            return TimeSeries(data.loc[feature])
        elif isinstance(feature, (list, numpy.ndarray)):
            return TimeSeriesGroup(data.loc[feature])
        else:
            raise TypeError('Cannot get item of type "{}" from TimeSeriesGroup'.format(
                type(feature)
            ))

    def __delitem__(self, ts):
        if isinstance(ts, TimeSeries):
            item = ts.feature
            if ts.feature not in self:
                raise ValueError("{} not in '{}'".format(self.ts.feature,
                                                         self.features))
        elif isinstance(ts, str):
            item = ts

        else:
            raise TypeError('Cannot delete "{}" from "{}"'.format(
                type(ts), type(self)
            ))
        data = self.as_df()

        data = data.drop(item, axis=0)
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

    def __iter__(self):
        return self.as_df().__iter__()

    def __next__(self):
        return self.as_df().__next__()

    def __contains__(self, item):
        return item in self.features

    @property
    def cluster(self):
        return self._cluster

    @cluster.setter
    def cluster(self, name):
        self._cluster = name

    @cluster.deleter
    def cluster(self):
        self._cluster = numpy.nan

    def concat(self, other, **kwargs):
        if not isinstance(other, TimeSeriesGroup):
            raise ValueError('Must merge TimeSeriesGroup objects. Got "{}"'.format(type(other)))

        other = other.as_df()
        df = pandas.concat([self.as_df(), other], **kwargs)
        return TimeSeriesGroup(df)

    def as_df(self):

        return pandas.DataFrame(self.values, columns=self.time, index=self.features)

    def norm(self, method='minmax', inplace=True):
        if method == 'minmax':
            ts_list = self.to_ts()
            normed = [i.norm(method=method, inplace=True) for i in ts_list]
            tsg = TimeSeriesGroup(normed)
            if inplace:
                self.__dict__ = tsg.__dict__
            return tsg

    def to_ts(self):
        """
        convert tgs into a list of ts
        objects
        :return:
        """
        ts = []
        for i in range(self.as_df().shape[0]):
            ts.append(TimeSeries(self.as_df().iloc[i]))
        return ts

    def annotate(self, id):
        """
        For use when feature names are
        non-readable IDs. Must have meta defined
        which is a dataframe containing mappings.
        index must be feature IDs and other columns
        can be other data pertaining to that id.
        :param id:
        :return:
        """
        if self.meta is None:
            raise ValueError('No meta dataframe has been '
                             'given. Please speficy a meta df '
                             'to use the annotate method')
        if id not in self.features:
            raise ValueError('id "{}" is not in '
                             'features'.format(id))

        return self.meta.loc[id]

    def getIDFromMeta(self, field, value):
        """
        For use when feature names are not human readable
        id's. Return field (column name) from meta who's
        value is value
        :return:
        """
        if self.meta is None:
            raise ValueError('No meta dataframe has been '
                             'given. Please speficy a meta df '
                             'to use the annotate method')
        if field not in self.meta.columns:
            raise ValueError('field "{}" is not in '
                             'meta df'.format(field))

        if value not in self.meta[field].values:
            raise ValueError('Value "{}" not in field "{}"'.format(value, field))

        return self.meta[self.meta[field] == value]


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
        plt.ylabel('AU (n={})'.format(len(feature)))
        plt.xlabel('Time')

        seaborn.despine(fig, top=True, right=True)
        return fig

    def heatmap(self, cmap='jet', yticklabels=False, **kwargs):
        """

        :return:
        """
        seaborn.set_context('talk', font_scale=2)
        seaborn.set_style('white')
        fig = plt.figure()

        seaborn.heatmap(self.as_df(), cmap=cmap,
                        yticklabels=yticklabels, **kwargs)
        seaborn.despine(fig, top=True, right=True)
        return fig


    def to_db(self, dbfile, table):
        """
        Write to db_file in a table called table.

        :param dbfile:
            Name of the database to write to
        :param table:
            Name of the table to write to
        :return:
        """
        if isinstance(table, int):
            table = '"{}"'.format(table)
        data = self.as_df()
        data.columns = [round(i, 8) for i in data.columns]
        columns = reduce(lambda x, y: x + y, ['"{}" DOUBLE PRECISION, '.format(i) for i in list(data.columns)])[:-2]
        create_table = "CREATE TABLE IF NOT EXISTS {} (".format(table) +\
                       "ID INTEGER PRIMARY KEY, " \
                       "cluster INTEGER," \
                       "feature TEXT NOT NULL," + columns + ');'
        if table not in DB(dbfile).tables():
            with DB(dbfile) as db:
                db.execute(create_table)
                # data.to_sql(name=table, con=db.conn, if_exists='append', index_label='feature')
        s = ''
        vals = ''
        for i in self.time:
            s += '"{}",'.format(i)
            vals += '?, '
        if not numpy.isnan(self.cluster):
            sql = "INSERT INTO " + table + "(cluster, feature, " + s[:-1] + ")"
            sql += ' values(?, ?, '
            # sql += vals[:-2]

        else:
            sql = "INSERT INTO " + table + "(feature," + s[:-1] + ")"
            sql += ' values(?, '

        sql += vals[:-2]
        sql += ');'

        data = self.as_df().reset_index()
        if not numpy.isnan(self.cluster):
            ## convert to typle
            tup = [tuple(x) for x in data.values]

            ## add cluster number to data
            tup = [[self.cluster] + list(tuple(i)) for i in tup]

            ##reconvert to tuples
            tup = [tuple(x) for x in tup]
        else:
            tup = [tuple(x) for x in data.values]

        with DB(dbfile) as db:
            db.executemany(sql, tup)

        return True

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
    def median(self):
        return TimeSeries(numpy.median(self.values, 0), time=self.time, feature='median')


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
        from dtw import DTW, FastDTW
        from threading import Thread
        from multiprocessing.pool import ThreadPool
        TP = ThreadPool(cpu_count() - 1)
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
                    thread = TP.apply_async(FastDTW, args=(x, y))
                    matrix.iloc[i, j] = thread.get()

        matrix.index = self.features
        print(matrix)
        matrix.columns = self.features
        return matrix


    @property
    def dtw_cost_matrix(self):
        matrix = self.dtw_matrix
        print(matrix, len(matrix))
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

    #
    # def _compute_dtw(self, vec):
    #     from dtw import DTW
    #     x = self.tsg.loc[vec[0]]
    #     y = self.tsg.loc[vec[1]]
    #     return DTW(x, y)
    #
    # @property
    # def dtw_matrix(self):
    #     comb = combinations(self.features, 2)
    #     P = Pool(cpu_count() - 1)
    #     return P.map(self._compute_dtw, comb)

    # @property
    # def cost_matrix(self):

    def sum_of_squared_dist(self, x, y):
        pass

    def eucl_dist_matrix(self):
        """
        calculate the distance matrix using func.
        :param func:
            Callable. Function to calculate distance matrix. Default=numpy.mean
        :return:
        """
        p = Pool(cpu_count() - 1)
        # comb = combinations(self.features, 2)
        dct = {}
        for i in self.features:
            x = self[i]
            dct[i] = {}
            for j in self.features:
                y = self[j]
                if i != j:
                    dct[i][j] = x.eucl_dist(y)
                else:
                    dct[i][j] = numpy.nan
        return pandas.DataFrame(dct)

    @property
    def centroid_by_dtw(self):
        id = self.dtw_cost_matrix().sum().idxmin()
        return self[id]

    @property
    def centroid_by_eucl(self):
        id = self.eucl_dist_matrix().sum().idxmin()
        return self[id]

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

    def intra_dtw_dist(self, stat=numpy.mean):
        """
        sum of DTW(ci, cj) squared for all i and j in the set of profiles and i != j
        :return:
        """
        ##import into local space because of a conflict
        from dtw import DTW
        dct = OrderedDict()
        for i in range(self.values.shape[0]):
            profile_i = TimeSeries(self.values[i], time=self.time, feature=self.features[i])
            dct[i] = DTW(stat, profile_i).cost ** 2
            dct[i] = dct[i].sum()

        df = pandas.DataFrame(dct, index=[0])
        return float(df.sum(axis=1))


    def intra_dtw_dist_normalized_by_clustsize(self, stat=numpy.mean):
        """
        sum of DTW(ci, cj) squared for all i and j in the set of profiles and i != j
        :return:
        """
        ##import into local space because of a conflict
        from dtw import DTW
        dct = OrderedDict()
        for i in range(self.values.shape[0]):
            profile_i = TimeSeries(self.values[i], time=self.time, feature=self.features[i])
            dct[i] = (DTW(self.do_statistic(stat), profile_i).cost ** 2) / self.nfeat
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

    def interpolate(self, kind='linear', num=20, inplace=False):
        ts_list = self.to_ts()
        interp_ts_list = [i.interpolate(kind, num, inplace) for i in ts_list]
        tsg = TimeSeriesGroup(interp_ts_list)
        if inplace:
            self.__dict__ = tsg.__dict__

        return tsg

    def to_singleton(self):

        ts = []
        for i in range(self.as_df().shape[0]):
            ts.append(TimeSeriesGroup(self.as_df().iloc[i]))
        return ts

    def sort(self, by=None):
        if by is None:
            return self

        # if by == 'max':
        #     print ([i.max() for i in self.to_ts())
        #
        # else:
        #     raise TypeError('cannot sort by "{}"'.format(by))


























