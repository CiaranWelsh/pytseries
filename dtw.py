import pandas, numpy
import os, glob
import seaborn
import matplotlib.pyplot as plt





class DTW(object):
    def __init__(self, x, y, labels=None):
        self.x = x
        self.y = y
        self.labels = labels

        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError

        if labels == None:
            labels = {'x': 'x',
                      'y': 'y'}

        self.x = numpy.array(self.x)
        self.y = numpy.array(self.y)

        self.acc_cost = self.calculate_cost()
        self.path, self.cost = self.find_best_path()

    def calculate_cost(self):
        self.distances = numpy.zeros((len(self.y), len(self.x)))
        for i in range(len(self.y)):
            for j in range(len(self.x)):
                self.distances[i, j] = (self.x[j] - self.y[i])**2
        acc_cost = numpy.zeros((len(self.y), len(self.x)))

        ## set acc_cost 0 0 to distances 0 0
        acc_cost[0, 0] = self.distances[0, 0]

        ## do accum cost for (0, x)
        for i in range(1, len(self.x)):
            acc_cost[0, i] = self.distances[0, i] + acc_cost[0, i-1]

        ## now in the y direction (y, 0)
        for i in range(1, len(self.y)):
            acc_cost[i, 0] = self.distances[i, 0] + acc_cost[i-1, 0]

        ## now the other elements
        for i in range(1, len(self.y)):
            for j in range(1, len(self.x)):
                acc_cost[i, j] = min(
                    acc_cost[i-1, j-1],
                    acc_cost[i-1, j],
                    acc_cost[i, j-1]
                ) + self.distances[i, j]

        return acc_cost

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

    def distance_cost_plot(self):
        plt.figure()
        im = plt.imshow(self.distances, interpolation='nearest', cmap='jet')
        plt.gca().invert_yaxis()
        plt.xlabel(self.labels['x'])
        plt.ylabel(self.labels['y'])
        plt.grid()
        plt.colorbar();
        path_x = [point[0] for point in self.path]
        path_y = [point[1] for point in self.path]
        plt.plot(path_x, path_y)

    def dwt_plot(self):
        plt.figure()
        plt.plot(self.x, 'bo-', label=self.labels['x'])
        plt.plot(self.y, 'g^-', label=self.labels['y'])
        plt.legend();
        for [map_x, map_y] in self.path:
            plt.plot([map_x, map_y], [self.x[map_x], self.y[map_y]], 'r')




x = 'CTGF'
y = 'TGFBI'

def dtw1pair(x, y):
    d = DTW(df.loc[x], df.loc[y], labels={'x': x, 'y': y})
    d.distance_cost_plot()
    d.dwt_plot()
    plt.show()


















