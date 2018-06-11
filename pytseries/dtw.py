import pandas, numpy
import seaborn
import matplotlib.pyplot as plt
# import site
# site.addsitedir('..')
from pytseries.core import TimeSeries
from .fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class _DTWBase(object):
    """
    Base class used by DTW classes.

    :param x: :py:class:`core.TimeSeries`
    :param y: :py:class:`core.TimeSeries`
    :param labels: dict {'x': xlabel, 'y': ylabel}. defaults to x.feature and y.feature
    :param dist: callable. Function to use as distance metric. See `py:mod:scipy.spatial.distance`

    """
    def __init__(self, x, y, labels=None, dist=euclidean):
        self.x = x
        self.y = y
        self.dist = dist
        self.labels = labels

        if not callable(self.dist):
            raise TypeError("dist arg should be a callable function. Got '{}'".format(type(self.dist)))

        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError

        if not self.x.__class__.__name__ == 'TimeSeries':
            self.x = self.coerse_to_timeseries(self.x)

        if not self.y.__class__.__name__ == 'TimeSeries':
            self.y = self.coerse_to_timeseries(self.y)

        if not self.x.__class__.__name__ == 'TimeSeries':
            raise TypeError('x and y arguments should be of type "core.TimeSeries"')

        if not self.y.__class__.__name__ == 'TimeSeries':
            raise TypeError('x and y arguments should be of type "core.TimeSeries"')

        if labels is None:
            self.labels = {'x': self.x.feature,
                           'y': self.y.feature}

    @staticmethod
    def coerse_to_timeseries(var):
        """
        if x and y not TimeSeries, convert to TS
        :param var:
        :return:
        """
        if not var.__class__.__name__ == 'TimeSeries':
            var = TimeSeries(var)
        return var

    def cost_plot2(self, interpolation='nearest', cmap='GnBu',
                           xlabel=None, ylabel=None, title=None,
                           **kwargs):
        if xlabel is None:
            xlabel = self.x.feature

        if ylabel is None:
            ylabel = self.y.feature

        df = pandas.DataFrame(self.acc_cost, index=self.x.time,
                              columns=self.y.time)

        # print(df)
        fig = plt.figure()
        seaborn.heatmap(df)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        # plt.grid()
        # cb = plt.colorbar()
        # cb.ax.set_ylabel('Cost')
        path_x, path_y = [i for i in zip(*self.path)]

        plt.plot(path_x, path_y, color='red')
        return fig

    def cost_plot(self, interpolation='nearest', cmap='GnBu',
                           xlabel=None, ylabel=None, title=None,
                           **kwargs):
        """
        Plot matrix of DTW distance betwen x and y

        :param interpolation: str. nearest
        :param cmap: colour map supported by matplotlib
        :param xlabel: str. label on x axis
        :param ylabel: str. Label on y axis
        :param title: str. Title of plot
        :param kwargs: other kwargs passed onto `py:class:matplotlib.pyplot:imshow`
        :return: :py:class:`matplotlib.Figure`
        """
        if xlabel is None:
            xlabel = "{} ({})".format(self.x.feature, self.x.time_unit)

        if ylabel is None:
            ylabel = "{} ({})".format(self.y.feature, self.y.time_unit)

        df = pandas.DataFrame(self.acc_cost, index=self.x.time,
                              columns=self.y.time)

        # print(df)
        fig = plt.figure()
        plt.imshow(df, interpolation=interpolation,
                   cmap=cmap, origin='lower', extent=(min(df.columns), max(df.columns),
                                                      min(df.index), max(df.index)),
                   **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        plt.grid()
        cb = plt.colorbar()
        cb.ax.set_ylabel('Cost')
        path_x, path_y = [i for i in zip(*self.path)]
        # x = [self.x.time[i] for i in path_x]
        # y = [self.y.time[i] for i in path_y]
        plt.plot(path_x, path_y, color='red')
        return fig

    def plot(self, legend_loc=None):
        """
        Plot time series plot with red lines
        indicating the optimum DTW distance for every point
        :param legend_loc: 2-element tuple. Location of the legend.
        :return: :py:class:`matplotlib.Figure`
        """
        seaborn.set_style('white')
        seaborn.set_context('talk', font_scale=2)
        fig = plt.figure()
        plt.plot(self.x.time, self.x.values, 'bo-', label=self.x.feature)
        plt.plot(self.y.time, self.y.values, 'g^-', label=self.y.feature)

        if legend_loc is not None:
            plt.legend(loc=legend_loc);
        else:
            plt.legend()
        for [map_x, map_y] in self.path:
            # map_x = self.x.time[map_x]
            # map_y = self.y.time[map_y]
            plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')

        seaborn.despine(fig, top=True, right=True)
        plt.xlabel('Time')
        plt.ylabel('AU')
        return fig


class DTW(_DTWBase):
    """
    Compute DTW distance using dynamic programming algorithm. Arguments
    are as in :py:class:`_DTWBase`

    Compute the DTW distance between two :py:class:`core.TimeSeries` objects
        >>> time = [15, 30, 60, 90, 120, 150, 180]
        >>> x_values = [1, 1.01, 1.044, 1.068, 1.096, 1.128, 1.1459]
        >>> y_values = [0.989, 1.031, 1.233, 1.205, 1.158, 1.176, 1.204]
        tsx = TimeSeries(time=time, values=x_values, feature='x')
        tsy = TimeSeries(time=time, values=y_values, feature='y')
        >>> dtwxy = DTW(x=tsx, y=tsy)
        >>> dtwxy
        DTW(x=x, y=y, cost=0.4073)
        >>> dtwxy.path
        [(15, 15), (15, 15), (30, 30), (60, 30), (90, 30), (120, 60), (150, 90), (180, 120), (180, 150), (180, 180)]
        >>> dtwxy.cost
        0.40730000000000044

    Plot the DTW cost matrix
        >>> import matplotlib.pyplot as plt
        >>> dtwxy.cost_plot()
        >>> plt.show()

    Plot time series plot with red lines indicating the optimum path
        >>> import matplotlib.pyplot as plt
        >>> dtwxy.plot()
        >>> plt.show()

    For some applications interpolation (or normalization) is useful
        >>> tsx = tsx.interp('linear', num=30)
        >>> tsy = tsy.interp('linear', num=30)
        >>> dtwxy = DTW(x=tsx, y=tsy)
        >>> dtwxy.plot()
        >>> plt.show()

    """
    def __init__(self, x, y, labels=None, dist=euclidean):
        super().__init__(x, y, labels=labels, dist=dist)

        self.acc_cost, self.distances = self.calculate_cost()
        self.path, self.cost = self.find_best_path()

        self.path = self.get_time_indices_for_path()

    def __str__(self):
        return "DTW(x={}, y={}, cost={})".format(self.x.feature, self.y.feature, round(self.cost, 4))

    def __repr__(self):
        return self.__str__()

    def get_time_indices_for_path(self):
        path_x, path_y = [i for i in zip(*self.path)]
        x = [self.x.time[i] for i in path_x]
        y = [self.y.time[i] for i in path_y]
        return [i for i in zip(x, y)]

    def calculate_cost(self):
        """
        Compute cost plot
        :return:
        """
        distances = numpy.zeros((len(self.y), len(self.x)))
        for i in range(len(self.y.time)):
            for j in range(len(self.x.time)):
                xtime = self.x.time[j]
                ytime = self.y.time[i]
                distances[i, j] = self.dist(self.x[xtime], self.y[ytime])
        acc_cost = numpy.zeros((len(self.y), len(self.x)))

        ## set acc_cost 0 0 to distances 0 0
        acc_cost[0, 0] = distances[0, 0]

        ## do accum cost for (0, x)
        for i in range(1, len(self.x)):
            acc_cost[0, i] = distances[0, i] + acc_cost[0, i-1]

        ## now in the y direction (y, 0)
        for i in range(1, len(self.y)):
            acc_cost[i, 0] = distances[i, 0] + acc_cost[i-1, 0]

        ## now the other elements
        for i in range(1, len(self.y)):
            for j in range(1, len(self.x)):
                acc_cost[i, j] = min(
                    acc_cost[i-1, j-1],
                    acc_cost[i-1, j],
                    acc_cost[i, j-1]
                ) + distances[i, j]

        return acc_cost, distances

    def find_best_path(self):
        cost = 0
        path = [[len(self.x) - 1, len(self.y) - 1]]
        i = len(self.y) - 1
        j = len(self.x) - 1
        while i > 0 and j > 0:
            if i == 0:
                j = j - 1
            elif j == 0:
                i = i - 1
            else:
                if self.acc_cost[i - 1, j] == min(
                        self.acc_cost[i - 1, j - 1],
                        self.acc_cost[i - 1, j],
                        self.acc_cost[i, j - 1]
                ):
                    i = i - 1
                elif self.acc_cost[i, j - 1] == min(
                        self.acc_cost[i - 1, j - 1],
                        self.acc_cost[i - 1, j],
                        self.acc_cost[i, j - 1]
                ):
                    j = j - 1
                else:
                    i = i - 1
                    j = j - 1
            path.append([j, i])
        path.append([0, 0])
        for y, x in path:
            cost = cost + self.distances[x, y]
        path = [i for i in reversed(path)]
        return path, cost


class FastDTW(_DTWBase):
    """
    Wrapper around fastdtw from (this)[https://github.com/slaypni/fastdtw] repository.
    The same arguments and plotting functions described in :py:class:`DTW` apply here.
    """
    def __init__(self, x, y, radius=1, dist=euclidean, labels=None):
        super().__init__(x, y, dist=dist, labels=labels)
        self.radius = radius
        self.cost, self.path = self.dtw()

    def dtw(self):
        return fastdtw(self.x.values, self.y.values, radius=self.radius, dist=self.dist)



















