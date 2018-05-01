import pandas, numpy, os, glob
import matplotlib.pyplot as plt
import seaborn
from itertools import combinations
from core import *
from dtw import *
from DB import *
from collections import OrderedDict
from scipy.stats import ttest_ind


class DTWClust(object):
    def __init__(self, tsg):
        self.tsg = tsg

        if not isinstance(self.tsg, TimeSeriesGroup):
            raise TypeError('Input to tsg argument should be a TimeSeriesGroup '
                            'object. Got "{}" instead'.format(type(self.tsg)))

    def initialize(self):
        pass













'''
some old code: delete when ready

class Element(object):
    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.cost == other.cost:
            return True
        else:
            return False

    def __ne__(self, other):
        if self.x == other.x and self.y == other.y and self.cost == other.cost:
            return False
        else:
            return True

    def __gt__(self, other):
        if self.cost > other.cost:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.cost < other.cost:
            return True
        else:
            return False

    def __le__(self, other):
        if self.cost == other.cost or self.cost > other.cost:
            return True

        else:
            return False

    def __ge__(self, other):
        if self.cost == other.cost or self.cost < other.cost:
            return True

        else:
            return False

    def __str__(self):
        return "Element(x={}, y={}, cost={})".format(self.x, self.y, self.cost)

    def __repr__(self):
        return self.__str__()

    def __sub__(self, other):
        try:
            return self.cost - other
        except AttributeError:
            return self.cost - other.cost

    def __add__(self, other):
        try:
            return self.cost - other

        except AttributeError:
            return self.cost + other.cost

    def __mul__(self, other):
        try:
            return self.cost * other
        except AttributeError:
            return self.cost * other.cost

    def __pow__(self, power):
        try:
            return self.cost ** power

        except AttributeError:
            return self.cost ** power

    def __truediv__(self, other):
        try:
            return self.cost / other

        except AttributeError:
            return self.cost / other.cost

    def __hash__(self):
        return hash((self.x, self.y, self.cost))


class Cluster(object):
    def __init__(self, id, elements):
        ## list of elements
        self.id = id
        self._elements = elements

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, value):
        assert isinstance(value, list)
        self._elements = value

    @elements.deleter
    def elements(self, index):
        assert isinstance(index, int)
        del self.elements[index]

    def __len__(self):
        return len(self.elements)

    @property
    def size(self):
        return len(self)

    def __str__(self):
        return "Cluster(id={}, size={})".format(self.id, self.size)

    def __repr__(self):
        return self.__str__()

    def append(self, new_element):
        assert isinstance(new_element, Element)
        self.elements.append(new_element)

    # def __getitem__(self, tup):
    #     if tup[0] is None:
    #         return [i for i in self.elements if i.y == tup[1]]
    #
    #     elif tup[1] is None:
    #         return [i for i in self.elements if i.x == tup[0]]
    #
    #     else:
    #         return [i for i in self.elements if i.x == tup[0] and i.y == tup[1]]

    def __delitem__(self, tup):
        assert len(tup) == 2
        idx = [i for i in range(len(self.elements)) if self.elements[i].y == tup[1] and self.elements[i].x == tup[0]]
        del self.elements[idx]

    def mean(self):
        return sum([i.cost for i in self.elements]) / len(self.elements)

    @property
    def intra_dist(self):
        """
        distance between data point in cluster with all other data points in cluster
        :return:
        """
        if len(self.elements) == 1:
            return self.mean()

        dist = []
        for i in self.elements:
            dist = (i - self.mean())**2
        return numpy.mean(dist)

    def inter_dist(self, other):
        """
        distance between data point in cluster with all other data points in cluster
        :return:
        """
        ## get combinations
        assert isinstance(other, Cluster)
        return numpy.mean([self.intra_dist,
                           other.intra_dist])

    def __eq__(self, other):
        if self.intra_dist == other.intra_dist:
            return True
        else:
            return False

    def __ne__(self, other):
        if self.intra_dist == other.intra_dist:
            return False
        else:
            return True

    def __gt__(self, other):
        if self.intra_dist > other.intra_dist:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.intra_dist < other.intra_dist:
            return True
        else:
            return False

    def __le__(self, other):
        if self.intra_dist == other.intra_dist or self.intra_dist > other.intra_dist:
            return True

        else:
            return False

    def __ge__(self, other):
        if self.intra_dist == other.intra_dist or self.intra_dist < other.intra_dist:
            return True

        else:
            return False



'''











