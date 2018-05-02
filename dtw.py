import pandas, numpy
import os, glob
import seaborn
import matplotlib.pyplot as plt
# import site
# site.addsitedir('..')
from core import TimeSeriesGroup, TimeSeries


class DTW(object):
    def __init__(self, x, y, labels=None):
        self.x = x
        self.y = y
        self.labels = labels

        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError

        if not isinstance(self.x, TimeSeries):
            self.x = self.coerse_to_timeseries(self.x)

        if not isinstance(self.y, TimeSeries):
            self.y = self.coerse_to_timeseries(self.y)

        if not isinstance(self.x, TimeSeries):
            raise TypeError('x and y arguments should be of type "core.TimeSeries"')

        if not isinstance(self.y, TimeSeries):
            raise TypeError('x and y arguments should be of type "core.TimeSeries"')

        if labels is None:
            self.labels = {'x': self.x.feature,
                           'y': self.y.feature}

        self.acc_cost, self.distances = self.calculate_cost()
        self.path, self.cost = self.find_best_path()

    def __str__(self):
        return "DTW(x={}, y={}, cost={})".format(self.x.feature, self.y.feature, round(self.cost, 4))

    def __repr__(self):
        return self.__str__()

    def coerse_to_timeseries(self, var):
        """
        if x and y not TimeSeries, convert to TS
        :param var:
        :return:
        """
        if not isinstance(var, TimeSeries):
            var = TimeSeries(var)
        return var

    def calculate_cost(self):
        distances = numpy.zeros((len(self.y), len(self.x)))
        for i in range(len(self.y.time)):
            for j in range(len(self.x.time)):
                xtime = self.x.time[j]
                ytime = self.y.time[i]
                distances[i, j] = (self.x[xtime] - self.y[ytime])**2
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

        return path, cost

    def cost_plot(self, interpolation='nearest', cmap='GnBu',
                           xlabel=None, ylabel=None, title=None,
                           **kwargs):
        if xlabel is None:
            xlabel = self.x.feature

        if ylabel is None:
            ylabel = self.y.feature

        fig = plt.figure()
        plt.imshow(self.acc_cost, interpolation=interpolation, cmap=cmap, **kwargs)
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        plt.grid()
        cb = plt.colorbar()
        cb.ax.set_ylabel('Cost')
        path_x = [point[0] for point in self.path]
        path_y = [point[1] for point in self.path]
        plt.plot(path_x, path_y, color='red')
        return fig

    def plot(self):
        seaborn.set_style('white')
        seaborn.set_context('talk', font_scale=2)
        fig = plt.figure()
        plt.plot(self.x.time, self.x.values, 'bo-', label=self.x.feature)
        plt.plot(self.y.time, self.y.values, 'g^-', label=self.y.feature)
        plt.legend();
        for [map_x, map_y] in self.path:
            map_x = self.x.time[map_x]
            map_y = self.y.time[map_y]
            plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')

        seaborn.despine(fig, top=True, right=True)
        plt.xlabel('Time')
        plt.ylabel('AU')
        return fig

    # def get_alignment(self):
    #     """
    #     use indices in path to
    #     provide the mapped timeseries and plot
    #     :return:
    #     """
    #     seaborn.set_style('white')
    #     seaborn.set_context('talk', font_scale=2)
    #     timex = []
    #     timey = []
    #     x = []
    #     y = []
    #     for [map_x, map_y] in self.path:
    #         timex.append(self.x.time[map_x])
    #         x.append(self.x[self.x.time[map_x]])
    #         timey.append(self.y.time[map_y])
    #         y.append(self.y[self.y.time[map_y]])
    #     return {
    #         self.x.feature: x,
    #         self.y.feature: y,
    #         'time_{}'.format(self.x.feature): timex,
    #         'time_{}'.format(self.y.feature): timey
    #     }
    #
    # def plot_alignment(self):
    #     """
    #
    #     :return:
    #     """
    #     align = self.get_alignment()
    #     print (align)
        # fig = plt.figure()
        # plt.plot(align['timex'], align['x'], label=self.x.feature, marker='o')
        # plt.plot(align['timey'], align['y'], label=self.y.feature, marker='o')
        # plt.xlabel('Time')
        # plt.ylabel('AU')
        # plt.legend(loc=(1, 0.5))
        # seaborn.despine(fig, top=True, right=True)
        # return fig




















