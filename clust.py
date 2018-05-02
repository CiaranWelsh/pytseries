import pandas, numpy, os, glob
import matplotlib.pyplot as plt
import seaborn
from itertools import combinations
from core import TimeSeriesGroup, TimeSeries
from dtw import DTW
from inout import DB
from collections import OrderedDict
from scipy.stats import ttest_ind
from numpy.random import choice
from sklearn.cluster import AgglomerativeClustering

import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)

LOG.setLevel(logging.INFO)

'''
It would be more efficient to represent
data types with built-in python objects

def evolutionary_algorithm():
    'Pseudocode of an evolutionary algorithm'    
    populations = [] # a list with all the populations

    populations[0] =  initialize_population(pop_size)
    t = 0

    while not stop_criterion(populations[t]):
        fitnesses = evaluate(populations[t])
        offspring = matting_and_variation(populations[t],
                                          fitnesses)
        populations[t+1] = environmental_selection(           
                                          populations[t],
                                          offspring)
        t = t+1
'''

class Individual(object):
    """
    for holding data about an individual in
    a population of possible clusters
    """

class DTWClust(object):
    def __init__(self, tsg, pop_size=20):
        self.tsg = tsg
        self.pop_size = pop_size
        if not isinstance(self.tsg, TimeSeriesGroup):
            raise TypeError('Input to tsg argument should be a TimeSeriesGroup '
                            'object. Got "{}" instead'.format(type(self.tsg)))

        self.pop = self.make_initial_population()


    def choose_k(self):
        """
        randomly select k parameter

        :return:
        """
        return choice(numpy.arange(1, len(self.tsg)))

    def make_random_population(self, k=None):
        """
        Make k random sub populations from
        self.tsg
        :param k:
        :return:
        """
        if k is None:
            k = self.choose_k()

        ind = numpy.arange(len(self.tsg))
        numpy.random.shuffle(ind)
        clust_alloc = numpy.array_split(ind, k)
        tsgs = {}
        for i in range(len(clust_alloc)):
            tsgs[i] = TimeSeriesGroup(self.tsg.as_df().iloc[clust_alloc[i]])
        return tsgs

    def make_initial_population(self):
        return {i: self.make_random_population() for i in range(self.pop_size)}

    def eval_fitness(self):
        """

        :return:
        """
        scores = {}
        for i, indiv in self.pop.items():
            score = 0
            comb = combinations(indiv.values(), 2)
            for j, tsg in indiv.items():
                score = score + tsg.intra_dtw_dist()

                for x, y in comb:
                    score = score - x.inter_dtw_dist(y)
            scores[i] = score
        return scores

    def selection(self, scores):
        """
        pick individuals proportional to their fitness
        :return:
        """
        ## pick random number of individuals to mate
        # # k = choice(numpy.arange(scores.keys()))
        # rank_population = sorted(scores.values())
        # rank_population = {i: rank_population[int(i)] for i in rank_population}
        # rank_pop = OrderedDict()
        # for i in range(len(scores)):
        #     rank_pop[i] =


    def mutation(self):
        pass


class Monitor(object):
    def __init__(self, dire=None, ext='png', dpi=400):
        self.dire = dire
        self.ext = ext
        self.dpi = dpi

        if self.ext.startswith('.'):
            self.ext = self.ext[1:]


    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __next__(self):
        return self.__dict__.__next__()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def copy(self):
        return self.__dict__.copy()

    def get(self, item):
        return self.__dict__.get(item)

    def plot_centroids(self):
        if not os.path.isdir(self.dire):
            os.makedirs(self.dire)

        figs = []
        for i in self:
            if isinstance(self[i], TimeSeriesGroup):
                fname = os.path.join(self.dire, 'centroid{}.{}'.format(i, self.ext))
                fig = self[i].plot_centroid()
                figs.append(fig)
                fig.savefig(fname, dpi=self.dpi, bbox_inches='tight')
        LOG.info('plots saved to "{}"'.format(self.dire))
        return figs

    def plot_features(self):
        if not os.path.isdir(self.dire):
            os.makedirs(self.dire)

        figs = []
        for i in self:
            if isinstance(self[i], TimeSeriesGroup):
                fname = os.path.join(self.dire, 'feature_plot{}.{}'.format(i, self.ext))
                LOG.info('plotting "{}"'.format(fname))
                fig = self[i].plot(self[i].features, legend=False)
                figs.append(fig)
                fig.savefig(fname, dpi=self.dpi, bbox_inches='tight')
        LOG.info('plots saved to "{}"'.format(self.dire))
        return figs


class HClust(object):
    def __init__(self, tsg, dire=None, **kwargs):
        self.tsg = tsg
        self.dire = dire

        if self.dire is None:
            self.dire = os.path.join(os.getcwd(),
                                     os.path.split(__file__)[1][:-2])

        ## passed onto Agglomerative Clustering
        self.kwargs = kwargs

        if self.kwargs.get('pooling_func') is None:
            self.kwargs['pooling_func'] = self.dtw_wrapper

        self.agg_clust = self._instantiate_agg_clustering()
        self.clusters = self.cluster()

    def __str__(self):
        return self.agg_clust.__str__()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def dtw_wrapper(x, y):
        return DTW(x, y).cost

    ## private
    def _instantiate_agg_clustering(self):
        return AgglomerativeClustering(**self.kwargs)

    def cluster(self):
        self.agg_clust.fit(self.tsg.as_df())
        labels = self.agg_clust.labels_
        df = self.tsg.as_df()
        df['cluster'] = labels
        mon = Monitor(dire=self.dire)
        for label, df in df.groupby(by='cluster'):
            df = df.drop('cluster', axis=1)
            mon[label] = TimeSeriesGroup(df)
        return mon









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











