import pandas, numpy
import matplotlib.pyplot as plt
import seaborn
from .inout import DB
import logging
from functools import reduce
from collections import OrderedDict
from scipy.interpolate import interp1d
from copy import deepcopy

from multiprocessing import Pool, cpu_count

# from dtw import DTW
logging.basicConfig()

LOG = logging.getLogger(__name__)


##todo gaussian resampling based on values as mean and an error matrix

class TimeSeries(object):
    """
    An object for storing and manipulating timeseries data.

    arguments
    ---------
    values: list-like. Stores time series data

    keyword arguments
    -----------------
    time: list-like. Time index for values.
    feature: scalar variable. Name of time series.
    time_unit: `str` default='min'
    feature_unit: `str` default='AU


    Create a TimeSeries object
    --------------------------
    >>> time = [15, 30, 60, 90, 120, 150, 180]
    >>> x_values = [1, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459]
    >>> ts = TimeSeries(time=time, values=x_values, feature='x')
    >>> print(ts)
    TimeSeries(data=[1.0, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459], \
    time=[15, 30, 60, 90, 120, 150, 180], feature="x")


    Perform numerical operations on TimeSeries
    ------------------------------------------
    Operator overloading is used so that numerical operators work as expected.
    >>> time = [15, 30, 60, 90, 120, 150, 180]
    >>> x_values = [1, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459]
    >>> y_values = [0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204]
    >>> tsx = TimeSeries(time=time, values=x_values, feature='x')
    >>> tsy = TimeSeries(time=time, values=y_values, feature='y')
    >>> tsx + tsy
    TimeSeries(data=[1.9889999999999999, 2.041, 2.277, 2.273, 2.254, 2.304, 2.3499], time=[15, 30, 60, 90, 120, 150, 180], feature=None)
    >>> tsx * tsy
    TimeSeries(data=[0.989, 1.04131, 1.287252, 1.2869400000000002, 1.269168, 1.3265279999999997, 1.3796635999999998], time=[15, 30, 60, 90, 120, 150, 180], feature=None)
    >>> tsx / tsy
    TimeSeries(data=[1.0111223458038423, 0.979631425800194, 0.8467153284671532, 0.8863070539419087, 0.9464594127806565, 0.9591836734693877, 0.9517441860465116], time=[15, 30, 60, 90, 120, 150, 180], feature=None)
    >>> tsx - tsy
    TimeSeries(data=[0.01100000000000001, -0.020999999999999908, -0.18900000000000006, -0.137, -0.06199999999999983, -0.04800000000000004, -0.05810000000000004], time=[15, 30, 60, 90, 120, 150, 180], feature=None)
    >>> tsx ** 2
    TimeSeries(data=[1.0, 1.0201, 1.089936, 1.140624, 1.2012160000000003, 1.2723839999999997, 1.3130868099999997], time=[15, 30, 60, 90, 120, 150, 180], feature=None)


    Indexing operations
    -------------------
    TimeSeries objects can be indexed to retrieve individual values by time point.
    >>> tsx = TimeSeries(time=time, values=x_values, feature='x')
    >>> tsx[15]
    0.989


    Distance Operations
    --------------------
    Compute sum of element wise euclidean distance between two time series

    >>> tsx.eucl_dist(tsy)
    0.06457561


    min/max
    -------
    >>> tsx.max()
    (180, 1.1459)
    >>> tsx.min()
    (15, 1.0)


    Normalization
    -------------
    The minmax method is used which scales time series so that minimum
    value is 0 and maximum value is 1
    >>> tsx.norm(inplace=True)
    TimeSeries(data=[0.0, 0.0685, 0.3015, 0.466, 0.657, 0.877, 1.0], time=[15, 30, 60, 90, 120, 150, 180], feature="x")


    Coersion
    --------
    To numpy.array
    >>> tsx.to_array()
    [[ 15.       1.    ]
     [ 30.       1.01  ]
     [ 60.       1.044 ]
     [ 90.       1.068 ]
     [120.       1.096 ]
     [150.       1.128 ]
     [180.       1.1459]]


    visualization
    -------------
    >>> import matplotlib.pyplot as plt
    >>> tsx.plot()
    >>> plt.show()
    """
    def __init__(self, values, time=None, feature=None,
                 time_unit='min', feature_unit='AU'):
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

        for i in self.time:
            if isinstance(i, str):
                raise ValueError('Time must be int or float. Got "{}"'.format(type(i)))

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, name):
        self._feature = name

    def interpolate(self, kind='linear', num=20, inplace=False):
        """
        Use scipy interp1d for interpolation
        :param kind: str. argument to `scipy.interpolate.interp1d`
        :param num: int. argument to `scipy.interpolate.interp1d`. Number of required data points
        :param inplace: Defualt=False. Perform operation on current object.
        :return: `TimeSeries`

        Example
        -------
        >>> time = [15, 30, 60, 90, 120, 150, 180]
        >>> y_values = [0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204]
        >>> tsx = TimeSeries(time=time, values=x_values, feature='x')
        >>> tsx.interpolate(kind='linear', num=10)
        TimeSeries(data=[1.0, 1.0137, 1.0345, 1.052, 1.066, 1.08355, 1.1013, 1.12088, 1.1349, 1.1459], time=[15.0, 33.33, 51.66, 70.0, 88.33, 106.66, 125.0, 143.33, 161.66, 180.0], feature="x")
        """
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

    def sample(self, size=1, err=0.1):
        """
        Sample from time series with the means = values
        and errors from err
        :return:
        """
        if err is None:
            raise ValueError('An argument must be specified to the err '
                             'attribute in order to sample from the pytseries')
        if isinstance(err, (float, int)):
            err = [err] * len(self)

        res = {}
        for i in range(len(self)):
            res[i] = self.values[i] + numpy.random.normal(self.values[i], self.values[i]*err[i], size=size)
            for i in res[i]:
                if i < 0:
                    raise ValueError('Sampled value less than 0')
        df = pandas.DataFrame(res)
        df.columns = self.time
        if size is 1:
            return TimeSeries(df, feature=self.feature)
        elif size > 1:
            return TimeSeriesGroup(df)
        else:
            raise ValueError('size must be a positive integer')


    def dydt(self, n=1, inplace=False):
        """
        Calculate the n-th discrete difference along values and time.
        The first difference is given by out[n] = a[n+1] - a[n],
        higher differences are calculated by using diff recursively.

        This is a wrapper around `numpy.diff'. The time vector is accumulated
        differentiation while the values is just differentiated.

        :param n: int (optional) default=1
            The number of times values are differenced.
            If zero, the input is returned as-is.
        :return: TimeSeries
            TimeSeries with its values differentiated


        Example
        -------
        >>> TimeSeries(data=[2, 5, 9, 14, 19], time=[1, 2, 4, 8, 12], feature="doubles")

        Will dydx to:
        >>> TimeSeries(data=[3, 4, 5, 5], time=[1, 3, 7, 11], feature="doubles")

        Note
        ----
        Any normalization or interpolation required should be applied before using dydx
        """
        ts = deepcopy(self)
        ts.__dict__ = deepcopy(self.__dict__)
        ts.values = numpy.diff(self.values, n=n)
        ts.time = numpy.cumsum(numpy.diff(self.time, n=n))
        if inplace:
            self.__dict__ = deepcopy(ts.__dict__)
        else:
            return ts

    # def dydt0(self, n=1, tol=1e-6):
    #     """
    #     get where dydt equals 0 within some degree
    #     of tolerance tol
    #     :return:
    #     """
    #     dydt = self.dydt(n, inplace=False)
    #     l = []
    #     for i in dydt.as_dict():
    #         if numpy.abs(dydt.as_dict()[i]) < tol:
    #             l.append((i, dydt.as_dict()[i]))
    #     if not l:
    #         print('iempty')
    #         return pandas.DataFrame()
    #
    #     print('not empty')
    #     df = pandas.DataFrame(l)
    #     df.columns = ['time', 'value']
    #     # LOG.info('df of values with derivaative < {}'.format(tol))
    #     df = df.set_index('time')
    #     return df

    def dydt0(self, n=1, tol=1e-6):
        """
        get where dydt equals 0 within some degree
        of tolerance tol
        :return:
        """
        dydt = self.dydt(n, inplace=False)
        l = []
        for i in dydt.as_dict():
            if numpy.abs(dydt.as_dict()[i]) < tol:
                l.append((i, dydt.as_dict()[i]))
        if not l:
            return pandas.DataFrame()

        df = pandas.DataFrame(l)
        df.columns = ['time', 'value']
        LOG.info('df of values with derivaative < {}'.format(tol))
        df = df.set_index('time')
        return df

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

        # if ts.err is not None:
        #     LOG.warning('The errors associated with the TimeSeries have '
        #                 'not been normalised with the time series data because'
        #                 ' it does not make sense to do so. Please reassign the errors'
        #                 'using the err property. ')

        return ts

    def __str__(self):
        if isinstance(self.feature, str):
            feature = '"{}"'.format(self.feature)

        else:
            feature = self.feature

        return """TimeSeries(data={}, time={}, feature={})""".format(
            list(self.values), list(self.time), feature)

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
        d[key] = value
        self = self.to_dict()

    def __add__(self, other):
        if not isinstance(other, TimeSeries):
            raise TypeError('Cannot add TimeSeries with type "{}"'.format(type(other)))

        if all(other.time) != all(self.time):
            raise ValueError('Time vectors must be equal in order to perform numerical operations '
                             'on TimeSeries objects')

        new_vals = self.values + other.values
        return TimeSeries(values=new_vals, time=self.time)

    def __sub__(self, other):
        if not isinstance(other, TimeSeries):
            raise TypeError('Cannot add TimeSeries with type "{}"'.format(type(other)))

        if all(other.time) != all(self.time):
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

    def to_copasi_format(self, fname=None, indep_vars=None):
        """
        Format data for copasi parameter estimation
        :param fname: str
            Name of file to output data to
        :return:
        """
        if indep_vars is not None:
            if not isinstance(indep_vars, dict):
                raise TypeError('indep_vars should be of type dict. Got "{}"'.format(type(indep_vars)))

            for i in indep_vars:
                if not isinstance(i, str):
                    raise TypeError('indep_vars should be dict[str] = (int, float)')

                if not isinstance(indep_vars[i], (int, float)):
                    raise TypeError('indep_vars[i] should be dict[str] = (int, float)')

                if i[-6:] != '_indep':
                    new_label = i+'_indep'
                    indep_vars[new_label] = indep_vars[i]
                    del indep_vars[i]

        if fname is not None:
            if not isinstance(fname, str):
                raise TypeError('fname arg should be of type str. Got "{}"'.format(type(fname)))

        df = pandas.DataFrame(self.as_series())
        df.index.name = 'Time'
        if indep_vars is not None:
            for i in indep_vars:
                df[i] = indep_vars[i]

        if fname is not None:
            df.to_csv(fname, index=True, sep='\t')
        return df


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

    def plot(self, fig=None, **kwargs):
        if kwargs.get('marker') is None:
            marker = 'o'
        else:
            marker = kwargs.pop('marker')

        seaborn.set_style('white')
        seaborn.set_context(context='talk', font_scale=2)
        if fig is None:
            fig = plt.figure()
        plt.plot(self.time, self.values, marker=marker, **kwargs)
        plt.xlabel('Time ({})'.format(str(self.time_unit)))
        plt.ylabel('{} ({})'.format(self.feature, self.feature_unit))
        seaborn.despine(fig, top=True, right=True)
        return fig

    def to_array(self):
        """
        output as 2d numpy array
        :return:
        """
        return numpy.array([numpy.array(i) for i in zip(self.time, self.values)])


class TimeSeriesGroup(object):
    """
    Object for collecting a group of time series

    Arguments
    =========
    values: List of TimeSeries or Pandas.DataFrame. Rows are features. Columns are time points

    Keyword Arguments
    =================
    features: list of features. Extracted from  values if pandas.DataFrame provided
    time: list of time points for columns in values. Extracted from values if pandas.DataFrame provided

    Create TimeSeries Object
    ========================

    From list of TimeSeries
    -----------------------
    >>> time = [15, 30, 60, 90, 120, 150, 180]
    >>> x_values = [1, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459]
    >>> y_values = [0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204]
    >>> tsx = TimeSeries(time=time, values=x_values, feature='x')
    >>> tsy = TimeSeries(time=time, values=y_values, feature='y')
    >>> tsg = TimeSeriesGroup([tsx, tsy])
         15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040

    From pandas.DataFrame
    ---------------------
    >>> df = pandas.DataFrame([x_values, y_values], columns=time, index=['x', 'y'])
    >>> tsg = TimeSeriesGroup(df)
    >>> tsg
        15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040

    Append new time series
    ----------------------
    >>> z_values = [i*2 for i in y_values]
    >>> tsz = TimeSeries(time=time, values=z_values, feature='z')
    >>> tsg.append(tsz, inplace=True)
        15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040
    z  1.978  2.062  2.466  2.410  2.316  2.352  2.4080

    Concatonate two TimeSeriesGroups
    --------------------------------
    >>> a_values = [i*3 for i in y_values]
    >>> tsa = TimeSeries(time=time, values=a_values, feature='a')
    >>> tsg1 = TimeSeriesGroup([tsx, tsy])
    >>> tsg2 = TimeSeriesGroup([tsa, tsz])
    >>> tsg = tsg1.concat(tsg2)
        15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040
    a  2.967  3.093  3.699  3.615  3.474  3.528  3.6120
    z  1.978  2.062  2.466  2.410  2.316  2.352  2.4080


    pandas.DataFrame indexing operations
    ====================================
    Many features that we know from the pandas.DataFrame also
    apply here.

    ## retrieve the 'x' time series.
    ## Note: currently return a pandas.Series. In the future this will
    ## return a TimeSeries
    >>> tsg.loc['x']
    15     1.0000
    30     1.0100
    60     1.0440
    90     1.0680
    120    1.0960
    150    1.1280
    180    1.1459
    Name: x, dtype: float64

    >>> time = [15, 30, 60, 90, 120, 150, 180]
    >>> x_values = [1, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459]
    >>> y_values = [0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204]
    >>> z_values = [i*2 for i in y_values]
    >>> df = pandas.DataFrame([x_values, y_values, z_values], columns=time, index=['x', 'y', 'z'])
    >>> tsg = TimeSeriesGroup(df)
    >>> tsg
        15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040
    z  1.978  2.062  2.466  2.410  2.316  2.352  2.4080
    >>> tsg.loc[['x', 'y']])
        15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040

    >>> tsg.iloc[[0, 1]]
        15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040
    >>> tsg.shape
    (3, 7)

    Note: currently the indexer of a TimeSeriesGroup
    returns a TimeSeriesGroup feature. This will be changed
    so that it will return a column of time points.
    >>> tsg[0]
    Currently fails because there is no feature called 0. Whereas
    in a future release will return the column of features at time 0.


    Statistics on TimeSeriesGroups
    ==============================
    >>> time = [15, 30, 60, 90, 120, 150, 180]
    >>> x_values = [1, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459]
    >>> y_values = [0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204]
    >>> z_values = [i*2 for i in y_values]
    >>> df = pandas.DataFrame([x_values, y_values, z_values], columns=time, index=['x', 'y', 'z'])
    >>> tsg = TimeSeriesGroup(df)
    >>> tsg.mean
    TimeSeries(data=[1.32, 1.36, 1.58, 1.561, 1.52, 1.55, 1.58], time=[15, 30, 60, 90, 120, 150, 180], feature="mean")
    >>> tsg.median
    TimeSeries(data=[1.48, 1.5465, 1.84, 1.8075, 1.736, 1.763, 1.806], time=[15, 30, 60, 90, 120, 150, 180], feature="median")
    >>> tsg.sd
    TimeSeries(data=[0.8175, 0.859, 1.067, 1.031, 0.974, 0.986, 1.011], time=[15, 30, 60, 90, 120, 150, 180], feature="std")
    >>> tsg.var
    TimeSeries(data=[0.6684, 0.738, 1.139, 1.063, 0.949, 0.972, 1.023], time=[15, 30, 60, 90, 120, 150, 180], feature="var")
    >>> tsg.coeff_var
    TimeSeries(data=[0.471, 0.47, 0.505, 0.497, 0.484, 0.481, 0.483], time=[15, 30, 60, 90, 120, 150, 180], feature="std")

    ## More generally, any numpy function that makes sense can be passed as argument to do_statistic
    >>> import numpy
    >>> tsg.do_statistic(numpy.median)
    TimeSeries(data=[1.48, 1.5465, 1.84, 1.8075, 1.73, 1.76, 1.806], time=[15, 30, 60, 90, 120, 150, 180], feature="function")


    Distance Matrices
    =================
    >>> tsg.dtw_cost_matrix
            x       y        a       z
    x      NaN  0.3963  16.4961  8.5001
    y   0.3963     NaN   15.992   7.996
    a  16.4961  15.992      NaN   7.996
    z   8.5001   7.996    7.996     NaN

    Warning: Takes a long time for large TimeSeriesGroups

    >>> tsg.eucl_dist_matrix()
            a          x          y          z
    a        NaN  39.240745  36.747808   9.186952
    x  39.240745        NaN   0.064576  10.465708
    y  36.747808   0.064576        NaN   9.186952
    z   9.186952  10.465708   9.186952        NaN


    Centroids
    =========
    Get the timeseries which has the minimum distance to all other timeseries
    - using the dtw distance
    >>> tsg.centroid_by_dtw()
    TimeSeries(data=[0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204], time=[15, 30, 60, 90, 120, 150, 180], feature="y")

    >>> tsg.centroid_by_eucl()
    TimeSeries(data=[1.978, 2.062, 2.466, 2.41, 2.316, 2.352, 2.408], time=[15, 30, 60, 90, 120, 150, 180], feature="z")


    Coersion
    ========
    ## convert into numpy.array
    >>> tsg.to_array()
    [[[ 15.       1.    ]
      [ 30.       1.01  ]
      [ 60.       1.044 ]
      [ 90.       1.068 ]
      [120.       1.096 ]
      [150.       1.128 ]
      [180.       1.1459]]

     [[ 15.       0.989 ]
      [ 30.       1.031 ]
      [ 60.       1.233 ]
      [ 90.       1.205 ]
      [120.       1.158 ]
      [150.       1.176 ]
      [180.       1.204 ]]

     [[ 15.       2.967 ]
      [ 30.       3.093 ]
      [ 60.       3.699 ]
      [ 90.       3.615 ]
      [120.       3.474 ]
      [150.       3.528 ]
      [180.       3.612 ]]

     [[ 15.       1.978 ]
      [ 30.       2.062 ]
      [ 60.       2.466 ]
      [ 90.       2.41  ]
      [120.       2.316 ]
      [150.       2.352 ]
      [180.       2.408 ]]]


    ##convert to list of TimeSeries objects
    >>> tsg.to_ts()
    [TimeSeries(data=[1.0, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459], time=[15, 30, 60, 90, 120, 150, 180], feature="x"),
     TimeSeries(data=[0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204], time=[15, 30, 60, 90, 120, 150, 180], feature="y"),
     TimeSeries(data=[2.967, 3.093, 3.6990000000000003, 3.615, 3.4739999999999998, 3.5279999999999996, 3.612], time=[15, 30, 60, 90, 120, 150, 180], feature="a"),
     TimeSeries(data=[1.978, 2.062, 2.466, 2.41, 2.316, 2.352, 2.408], time=[15, 30, 60, 90, 120, 150, 180], feature="z")]
    >>> tsg.as_df()
        15     30     60     90     120    150     180
    x  1.000  1.010  1.044  1.068  1.096  1.128  1.1459
    y  0.989  1.031  1.233  1.205  1.158  1.176  1.2040
    a  2.967  3.093  3.699  3.615  3.474  3.528  3.6120
    z  1.978  2.062  2.466  2.410  2.316  2.352  2.4080


    transforming operations
    =======================
    Transformation methods are the same as those in TimeSeries and are applied to each
    TimeSeries individually

    Normalisation
    -------------
    Using the minmax method so that minimum values in the time series are 0 and maximum values are 1
    >>> tsg.norm()
        15        30        60        90        120       150       180
    x  0.0  0.068540  0.301576  0.466073  0.657985  0.877313  1.000000
    y  0.0  0.172131  1.000000  0.885246  0.692623  0.766393  0.881148
    a  0.0  0.172131  1.000000  0.885246  0.692623  0.766393  0.881148
    z  0.0  0.172131  1.000000  0.885246  0.692623  0.766393  0.881148

    interpolation
    --------------
    >>> tsg.interpolate('linear', num=10)
        15.000000   33.333333   51.666667   70.000000   88.333333   106.666667  \
    x       1.000    1.013778    1.034556    1.052000    1.066667    1.083556
    y       0.989    1.053444    1.176889    1.223667    1.206556    1.178889
    a       2.967    3.160333    3.530667    3.671000    3.619667    3.536667
    z       1.978    2.106889    2.353778    2.447333    2.413111    2.357778

       125.000000  143.333333  161.666667  180.000000
    x    1.101333    1.120889    1.134961      1.1459
    y    1.161000    1.172000    1.186889      1.2040
    a    3.483000    3.516000    3.560667      3.6120
    z    2.322000    2.344000    2.373778      2.4080

    differentiation
    ---------------
    >>> tsg.dydt()
        15     45     75     105    135     165
    x  0.010  0.034  0.024  0.028  0.032  0.0179
    y  0.042  0.202 -0.028 -0.047  0.018  0.0280
    a  0.126  0.606 -0.084 -0.141  0.054  0.0840
    z  0.084  0.404 -0.056 -0.094  0.036  0.0560

    dydt0
    -----
    Find where dydt is equal to 0. This function is experimental.
    dydt is considered 0 if the dydt < tolerance

    ## first interpolate so that we have greater chance of finding a dydt
    ## smaller than tolerance argument
    >>> tsg.interpolate(num=25, inplace=True)
    >>> tsg.dydt0(tol=1e-4)
               value
      time
    y 110.0  0.000063

    """
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

            else:
                raise ValueError('If list, must be a list of TimeSeries.')

        if not isinstance(self.values, numpy.ndarray):
            if type(self.values) == pandas.DataFrame or type(self.values) == pandas.Series:
                if self.features is None:
                    self.features = numpy.array(self.values.index)
                if self.time is None:
                    self.time = numpy.array(self.values.columns)
                if self.values is None:
                    self.values = self.values.as_matrix()


        else:
            if self.features is None:
                LOG.warning('No features specified. Features will be labelled with numbers')
                self.features = range(self.values.shape[0])

            if self.time is None:
                LOG.warning('No time has been specified. Time will increment linearly from 0.')
                self.time = range(self.values.shape[1])

        assert self.nfeat * self.ntime == self.values.shape[0] * self.values.shape[1]

        # make sure features are unique
        if len(self.features) != len(set(self.features)):
            raise ValueError('There are duplicated features in '
                             'your data. Please make feature IDs '
                             'unique.')
        # for meta df and err df, make sure the
        # index are the same as features

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

        # make features immutable
        # self.features = OrderedSet(self.features)

    def __str__(self):
        return self.as_df().__str__()

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, feature):
        if isinstance(feature, list):
            for i in feature:
                if i not in self.features:
                    raise ValueError("feature '{}' not in features: '{}'".format(i, self.features))
        else:
            if feature not in self.features:
                raise ValueError('feature "{}" not in features: {}'.format(feature, self.features))

        data = self.as_df()
        if isinstance(feature, (str, int, numpy.int64)):
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
        return self.as_df().index.__iter__()

    def __next__(self):
        return self.as_df().index.__next__()

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

    def dydt(self, n=1, inplace=False):
        l = []
        for i in self.to_ts():
            l.append(i.dydt(n, inplace=False))

        tsg = TimeSeriesGroup(l)
        if inplace:
            self.__dict__ = deepcopy(tsg.__dict__)
        else:
            return tsg

    def to_ts(self):
        """
        convert tgs into a list of ts
        objects
        :return:
        """
        ts = []
        for i in range(len(self.features)):
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

    def to_array(self):
        return numpy.array([i.to_array() for i in self.to_ts()])

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

    def append(self, ts, inplace=False):
        if not isinstance(ts, TimeSeries):
            raise TypeError('ts argument should be a TimeSeries')

        if all(self.time) != all(ts.time):
            raise ValueError('Time argument for "{}" must be same as the TimeSeriesGroup '
                             'you are entering it into.')
        features = numpy.append(list(self.features), ts.feature)
        values = numpy.vstack([self.values, ts.values])
        tsg = TimeSeriesGroup(values=values, features=features, time=self.time)
        if inplace == True:
            self.__dict__ = deepcopy(tsg.__dict__)

    def plot(self, feature, legend=True, legend_loc=(1, 0.1),
             ylabel='AU', **kwargs):
        seaborn.set_context(context='talk', font_scale=2)
        seaborn.set_style('white')
        fig = plt.figure()

        if isinstance(feature, str):
            feature = [feature]

        for f in feature:
            if f not in self.features:
                raise ValueError('TimeSeriesGroup does not contain feature "{}". '
                                 'These features are available: "{}"'.format(f, self.features))

            plt.plot(self.time, self.as_df().loc[f], label=f,**kwargs)
        if legend is True:
            plt.legend(loc=legend_loc)
        elif isinstance(legend, int):
            if len(self) < legend:
                plt.legend(loc=legend_loc)
        plt.ylabel('{} (n={})'.format(ylabel, len(feature)))
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

    def dydt0(self, n=1, tol=1e-6):
        """
        TSG version of ts.dydt0
        :return:
        """
        df_dct = {}
        for i in self.to_ts():
            df_dct[i.feature] = i.dydt0(n, tol)

        df = pandas.concat(df_dct)
        return df


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
        from pytseries.dtw import FastDTW
        from multiprocessing.pool import ThreadPool
        TP = ThreadPool(cpu_count() - 1)
        # matrix = numpy.ndarray((self.shape[0], self.shape[0]))
        matrix = pandas.DataFrame(numpy.zeros((self.shape[0], self.shape[0])))
        for i in range(self.shape[0]):
            for j in range(self.shape[0]):
                if i == j:
                    matrix.iloc[i, j] = numpy.nan
                else:
                    xfeat = list(self.features)[i]
                    yfeat = list(self.features)[j]
                    x = TimeSeries(self.loc[xfeat], time=self.time, feature=xfeat)
                    y = TimeSeries(self.loc[yfeat], time=self.time, feature=yfeat)
                    thread = TP.apply_async(FastDTW, args=(x, y))
                    matrix.iloc[i, j] = thread.get()

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
        id = self.dtw_cost_matrix.sum().idxmin()
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
    #
    # def intra_eucl_dist(self):
    #     """
    #     objective function 1. Squared sum of all DTW distances
    #     in the cluster
    #     :return:
    #     """
    #     dct = OrderedDict()
    #     for i in range(self.values.shape[0]):
    #         profile_i = TimeSeries(self.values[i], time=self.time, feature=self.features[i])
    #         dct[i] = (self.mean - profile_i) ** 2
    #         dct[i] = dct[i].sum()
    #
    #     df = pandas.DataFrame(dct, index=[0])
    #     return float(df.sum(axis=1))
    #
    # def inter_eucl_dict(self, other):
    #     if not isinstance(other, TimeSeriesGroup):
    #         raise TypeError('Argument "other" should be of type TimeSeriesGroup. '
    #                         'got "{}" instead'.format(type(other)))
    #
    #     return ((self.mean - other.mean) ** 2).sum()

    def intra_dtw_dist(self, stat=numpy.mean):
        """
        sum of DTW(ci, cj) squared for all i and j in the set of profiles and i != j
        :return:
        """
        ##import into local space because of a conflict
        from pytseries.dtw import DTW
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
        from pytseries.dtw import DTW
        dct = OrderedDict()
        for i in range(self.values.shape[0]):
            profile_i = TimeSeries(self.values[i], time=self.time, feature=self.features[i])
            dct[i] = (DTW(self.do_statistic(stat), profile_i).cost ** 2) / self.nfeat
            dct[i] = dct[i].sum()

        df = pandas.DataFrame(dct, index=[0])
        return float(df.sum(axis=1))

    # def inter_dtw_dist(self, other):
    #     if not isinstance(other, TimeSeriesGroup):
    #         raise TypeError('Argument "other" should be of type TimeSeriesGroup. '
    #                         'got "{}" instead'.format(type(other)))
    #
    #     ##import into local space because of a conflict
    #     from dtw import DTW
    #
    #     return (DTW(self.mean, other.mean).cost ** 2).sum()

    # def plot_centroid(self, **kwargs):
    #     seaborn.set_context('talk', font_scale=2)
    #     seaborn.set_style('white')
    #     center_data = self.loc[self.center_profile]
    #     fig = plt.figure()
    #     plt.errorbar(x=self.time, y=center_data.values,
    #                  yerr=self.sd.values, marker='o',
    #                  **kwargs)
    #     plt.ylabel('Centroid Profile')
    #     plt.xlabel('Time')
    #     seaborn.despine(fig, top=True, right=True)
    #     return fig

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


    def to_copasi_format(self, fname=None, indep_vars=None):
        df_lst = []
        for i in self.features:
            df_lst.append(self[i].to_copasi_format(indep_vars=None))

        df = pandas.concat(df_lst, axis=1)
        if indep_vars is not None:
            for i in indep_vars:
                df[i] = indep_vars[i]

        if fname is not None:
            df.to_csv(fname, sep='\t', index=True)

        return df























