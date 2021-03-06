import pandas, numpy, os
import matplotlib.pyplot as plt
import seaborn
from pytseries.core import TimeSeriesGroup
from pytseries.dtw import DTW, FastDTW
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from multiprocessing import Pool, cpu_count
import inspect
from itertools import combinations
from copy import deepcopy
import pickle
from scipy.spatial.distance import euclidean

import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)

LOG.setLevel(logging.INFO)



class Monitor(object):
    def __init__(self, dire=None, ext='png', dpi=400):
        self.dire = dire
        self.ext = ext
        self.dpi = dpi
        self.clusters = {}

        if self.ext.startswith('.'):
            self.ext = self.ext[1:]


    def __getitem__(self, item):
        return self.clusters[item]
        # dct = {i: j for (i, j) in self.__dict__.items() if isinstance(j, TimeSeriesGroup)}
        # return dct[item]

    def __setitem__(self, key, value):
        self.clusters[key] = value

    def __delitem__(self, key):
        del self.clusters[key]

    def __str__(self):
        return self.clusters.__str__()

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        return self.clusters.__iter__()

    def __next__(self):
        return self.clusters.__next__()

    def keys(self):
        return self.clusters.keys()

    def values(self):
        return self.clusters.values()

    def items(self):
        return self.clusters.items()

    def copy(self):
        return self.clusters.copy()

    def get(self, item):
        return self.clusters.get(item)

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

    def total_dtw_dist_score(self):
        scores = {}
        for i in self:
            if isinstance(self[i], TimeSeriesGroup):
                scores[i] = self[i].intra_dtw_dist()

        print(scores)


class HClustWithSklearn(object):
    def __init__(self, tsg, dire=None, kscan=False, ks=None, **kwargs):
        self.tsg = tsg
        self.dire = dire
        self.kscan = kscan
        self.ks = ks

        if self.kscan:
            if self.ks is None:
                raise ValueError('Please provide a list of integers for '
                                 'the ks argument in order to do a kscan')
        if self.dire is None:
            self.dire = os.path.join(os.getcwd(),
                                     os.path.split(__file__)[1][:-2])

        ## passed onto Agglomerative Clustering
        self.kwargs = kwargs

        if self.kwargs.get('pooling_func') is None:
            self.kwargs['pooling_func'] = self.dtw_wrapper

        # if self.kwargs.get('affinity') is None:
        #     self.kwargs['affinity'] = self.dtw_wrapper

        # print (self.kwargs['affinity'])

        if not self.kscan:
            self.agg_clust = self._instantiate_agg_clustering()
            self.clusters = self.cluster()

        else:
            self.kscan_result = self.do_kscan()

    def __str__(self):
        return self.agg_clust.__str__()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def dtw_wrapper(x, y, axis=1):
        return DTW(x, y).cost

    # @staticmethod
    # def dtw_wrapper(l):
    #     import functools
    #     return functools.reduce(lambda x, y: DTW(x, y).cost, l)

    ## private
    def _instantiate_agg_clustering(self):
        return AgglomerativeClustering(**self.kwargs)

    def cluster(self):
        data = self.tsg.as_df()
        self.agg_clust.fit(data)
        labels = self.agg_clust.labels_
        df = self.tsg.as_df()
        df['cluster'] = labels
        mon = Monitor(dire=self.dire)
        for label, df in df.groupby(by='cluster'):
            df = df.drop('cluster', axis=1)
            mon[label] = TimeSeriesGroup(df)
        return mon

    def do_kscan(self):
        kscan = {}
        for i in self.ks:
            agg_clust_i = AgglomerativeClustering(n_clusters=i, **self.kwargs)
            agg_clust_i.fit(self.tsg.as_df())
            labels = agg_clust_i.labels_
            df = self.tsg.as_df()
            df['cluster'] = labels
            mon = Monitor(dire=self.dire)
            for label, df in df.groupby(by='cluster'):
                df = df.drop('cluster', axis=1)
                mon[label] = TimeSeriesGroup(df)
            kscan[i] = mon
        return kscan

    def get_obj1(self):
        kscan = self.kscan_result
        vals = {}
        for k in kscan:
            sum = 0
            for i in kscan[k]:
                sum += kscan[k][i].intra_dtw_dist(numpy.median)
            vals[k] = sum

        return pandas.DataFrame(vals,index=['k']).transpose()

    def get_obj2(self):
        vals = {}
        kscan = self.kscan_result
        for k in kscan:
            sum = 0
            for i in kscan[k]:
                sum += kscan[k][i].intra_dtw_dist_normalized_by_clustsize(numpy.median)
            vals[k] = sum
        return pandas.DataFrame(vals, index=['k']).transpose()

    def get_obj3(self):
        vals = {}
        kscan = self.kscan_result
        for k in kscan:
            sum = 0
            for i in kscan[k]:
                sum += kscan[k][i].intra_dtw_dist(numpy.median)
            vals[k] = sum + k
        return pandas.DataFrame(vals, index=['k']).transpose()

    def get_obj4(self):
        vals = {}
        kscan = self.kscan_result
        for k in kscan:
            sum = 0
            for i in kscan[k]:
                sum += kscan[k][i].intra_dtw_dist(numpy.median)
                if len(kscan[k][i]) == 1:
                    sum += 1
            vals[k] = sum
        return pandas.DataFrame(vals, index=['k']).transpose()

    def get_obj5(self):
        vals = {}
        kscan = self.kscan_result
        for k in kscan:
            sum = 0
            for i in kscan[k]:
                sum += kscan[k][i].intra_dtw_dist(numpy.median)
                if len(kscan[k][i]) == 1:
                    sum += 1
            vals[k] = sum + k
        return pandas.DataFrame(vals, index=['k']).transpose()

    def plot_kscan(self):
        seaborn.set_style('white')
        seaborn.set_context('talk', font_scale=2)

        obj1 = self.get_obj1()
        obj2 = self.get_obj2()
        obj3 = self.get_obj3()
        obj4 = self.get_obj4()
        obj5 = self.get_obj5()

        objs = [obj1, obj2, obj4]

        fig = plt.figure()
        for i in objs:
            plt.plot(i.index, i['k'], label=self.retrieve_name(i))
            plt.legend(loc='best')
            plt.xlabel('K')
            plt.ylabel('Obj function value')
            seaborn.despine(fig=fig, top=True, right=True)
        plt.show()


    def retrieve_name(self, var):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is var]


class HClustWithScipy(object):
    def __init__(self, tsg, linkage_method,
                 dire=None,
                 kscan=False, ks=None, **kwargs):
        self.tsg = tsg
        self.linkage_method = linkage_method
        self.dire = dire
        self.kscan = kscan
        self.ks = ks

        if self.kscan:
            if self.ks is None:
                raise ValueError('Please provide a list of integers for '
                                 'the ks argument in order to do a kscan')
        if self.dire is None:
            self.dire = os.path.join(os.getcwd(),
                                     os.path.split(__file__)[1][:-2])

        self.kwargs = kwargs

        self.z = self.cluster()



    def cluster(self):
        z = linkage(self.tsg.as_df(), method=self.linkage_method)
        return z

    def cophenet_score(self):
        return cophenet(self.z, pdist(self.tsg.as_df()))[0]

    def dendrogram(self, figsize=(24,24), **kwargs):
        if figsize is not None:
            assert isinstance(figsize, tuple)
        seaborn.set_context('talk', font_scale=2)
        seaborn.set_style('white')

        fig = plt.figure(figsize=figsize)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        d = dendrogram(
            self.z,
            leaf_rotation=90.,  # rotates the x axis labels
            labels=list(self.tsg.as_df().index),
            **kwargs,
        )
        for k, v in d.items():
            print (k, v)
        seaborn.despine(fig, top=True, right=True)
        return fig

    def get_clusters(self, level='all', **kwargs):
        d = dendrogram(
            self.z,
            leaf_rotation=90.,
            labels=list(self.tsg.as_df().index),
            no_plot=True,
            **kwargs,
        )
        for k in d.keys():
            print (k, d[k])


class HClustDTW(object):
    def __init__(self, tsg, nclust=1, db_file=None,
                 dtw_dist=euclidean, fast=False, radius=1):
        self.nclust = nclust
        self.dtw_dist = dtw_dist
        self.fast = fast
        self.radius = radius

        self.tsg = tsg
        self.clusters = {int(i): self.tsg.to_singleton()[i] for i in range(self.tsg.shape[0])}
        self.evolution_of_clusters = {}

        self.db_file = db_file
        if self.db_file is None:
            self.db_file = os.path.join(os.getcwd(), os.path.split(__file__)[1][:-3]+'.db')

        if not isinstance(self.nclust, int):
            raise ValueError("nclust should be of type int")

    def compute_dtw(self, vec):
        x = self.clusters[vec[0]].centroid_by_eucl
        y = self.clusters[vec[1]].centroid_by_eucl
        dtw = DTW(x, y, dist=self.dtw_dist)
        return vec[0], vec[1], dtw.cost

    def compute_fastdtw(self, vec):
        x = self.clusters[vec[0]].centroid_by_eucl
        y = self.clusters[vec[1]].centroid_by_eucl
        dtw = FastDTW(x, y, dist=self.dtw_dist, radius=self.radius)
        return vec[0], vec[1], dtw.cost

    def dist_matrix(self, clusters):
        comb = [i for i in combinations(clusters.keys(), 2)]
        if self.fast:
            result = map(self.compute_fastdtw, comb)
        else:
            result = map(self.compute_dtw, comb)
        x, y, cost = zip(*result)
        df = pandas.DataFrame([x, y, cost])#, index=['ci, cj', 'cost'])
        df = df.transpose()
        df.columns = ['ci', 'cj', 'cost']
        df.set_index(['ci', 'cj'], inplace=True)
        return df.unstack()['cost']

    @staticmethod
    def get_pair_to_merge(dist_matrix):
        min_value = dist_matrix.min().min()
        matrix = dist_matrix[dist_matrix == min_value]
        min_val = matrix.dropna(how='all').dropna(how='all', axis=1)
        return list(min_val.index)[0], list(min_val.columns)[0], min_val.values[0]

    def fit(self):
        merge_pairs = {}
        i = 0
        go = True
        pool = Pool(cpu_count() - 1)
        dct = {}

        while (len(self.clusters) != self.nclust) and go:
            try:
                LOG.info('iteration "{}"'.format(i))
                # print ('\n new iteration')
                # print('go is', go)
                # print('len clusters is', len(self.clusters))
                i += 1
                table = i
                # print(i)
                dist = self.dist_matrix(self.clusters)

                ci, cj, distance = self.get_pair_to_merge(dist)
                merge_pairs[i] = (ci, cj)

                cluster = self.clusters[ci].concat(self.clusters[cj])

                cluster_size = len(cluster)
                dct[table] = [ci, cj, distance, cluster_size]

                # print('merged cluster is: ', cluster)

                # print ('\n\n slf.clusters before update is :')

                if len(self.clusters) == 1:
                    go = False
                    continue
                else:

                    del self.clusters[ci]
                    del self.clusters[cj]

                    ## add merged cluster to max keys plus 1
                    new_key = 0
                    while new_key in self.clusters.keys():
                        new_key += 1

                    if new_key in self.clusters.keys():
                        raise ValueError('new key "{}" in keys'.format(new_key))

                    self.clusters[new_key] = cluster
                    # pool.map(self.map_to_db, (self.clusters, table))
                    # pool.start()
                    # pool.join()
                    # pool.close()
                    # for ci in self.clusters:
                    #     self.clusters[ci].cluster = ci
                        # self.clusters[ci].to_db(self.db_file, table)

                    # print('\n\n slf.clusters after  update is :')

                    self.evolution_of_clusters[i] = deepcopy(self.clusters)

            except ValueError as e:
                LOG.debug(e)
                LOG.warning(e)
                if 'not enough values to unpack' in str(e):
                    go = False

                else:
                    raise e
        return dct
        # print(dct)
        # LOG.info("Results stored in database at '{}'".format(self.db_file))
        # return self.evolution_of_clusters, merge_pairs

    def to_pickle(self, fle):
        with open(fle, 'wb') as f:
            pickle.dump(self.evolution_of_clusters, f)

    def to_db(self):
        """
        call a tsg to_db method
        :param tsg:
        :return:
        """




        sql_list = []
        for table in self.evolution_of_clusters:
            print(table, self.evolution_of_clusters[table])
        # for table in res:
        #     for cluster in res[table]:
        #         print (cluster)
                # cl = res[table][cluster]
                # print(cl)
                # sql_list.append(cl.to_db(self.db_file, table))



class HClustDTW2(object):
    def __init__(self, tsg, nclust=1, db_file=None,
                 dtw_dist=euclidean, fast=False, radius=1):
        self.nclust = nclust
        self.dtw_dist = dtw_dist
        self.fast = fast
        self.radius = radius

        self.tsg = tsg
        self.clusters = {i: TimeSeriesGroup(self.tsg[i]) for i in self.tsg.features}
        self.evolution_of_clusters = {}

        self.db_file = db_file
        if self.db_file is None:
            self.db_file = os.path.join(os.getcwd(), os.path.split(__file__)[1][:-3]+'.db')

        if not isinstance(self.nclust, int):
            raise ValueError("nclust should be of type int")

    @property
    def dist_matrix(self):
        dct = {}
        matrix = numpy.zeros((len(self.tsg), len(self.tsg)))
        matrix[:] = numpy.nan
        df = pandas.DataFrame(matrix, columns=self.tsg.features, index=self.tsg.features)
        comb = [i for i in combinations(self.tsg.features, 2)]
        for i, j in comb:
            cost = DTW(self.tsg[i], self.tsg[j]).cost
            df.loc[i, j] = cost
            df.loc[j, i] = cost
        return df

    def pick_prototype(self, cluster):
        """

        :param cluster:
        :return:
        """
        sub_dist_matrix = self.dist_matrix.loc[cluster.features, cluster.features]
        return sub_dist_matrix.sum(axis=0).idxmin()

    def select_pair_to_merge(self, clusters):
        """

        :return:
        """
        p = Pool(cpu_count() - 1)
        prototypes = []
        # print(clusters)
        for i in clusters:
            prototypes.append(self.select_pair_to_merge(i))
        # prototypes = p.map_async(self.pick_prototype, clusters.values())
        # prototypes = prototypes.get()
        comb = [i for i in combinations(prototypes, 2)]

        dct = {(i, j): self.dist_matrix.loc[i, j] for i, j in comb}
        x, y = min(dct)
        return x, y, dct[min(dct)]

    def select_pair_to_merge_map(self, clusters):
        """

        :return:
        """
        p = Pool(cpu_count() - 1)
        prototypes = []
        # print(clusters)
        prototypes = p.map_async(self.pick_prototype, clusters.values())
        prototypes = prototypes.get()
        comb = [i for i in combinations(prototypes, 2)]

        dct = {(i, j): self.dist_matrix.loc[i, j] for i, j in comb}
        x, y = min(dct)
        return x, y, dct[min(dct)]

    def fit(self):
        clusters = deepcopy(self.clusters)
        iterations = []
        cluster_history = {}
        for i in range(len(self.tsg) -1):
            cluster_history[i] = {}
            try:
                xclust, yclust, dist = self.select_pair_to_merge(clusters)
            except ValueError as e:
                if str(e) == 'min() arg is an empty sequence':
                    break
                else:
                    raise e
            new = clusters[xclust].concat(clusters[yclust])
            del clusters[xclust]
            del clusters[yclust]
            clusters[self.pick_prototype(new)] = new
            iteration_info = [xclust, yclust, dist, len(new)]
            iterations.append(iteration_info)
            cluster_history[i] = deepcopy(clusters)

        df = pandas.DataFrame(iterations)
        df.columns= ['xclust', 'yclust', 'dist', 'n']
        df.index.name = 'iteration'
        return cluster_history, df

    # @staticmethod
    def dendrogram(self, Z):
        print(self.clusters)
        left = []
        left_middle = []
        for i in Z['xclust']:
            for j in range(len(self.tsg.features)):
                if i == self.tsg.features[j]:
                    left.append(j)
        for i in Z['yclust']:
            for j in range(len(self.tsg.features)):
                if i == self.tsg.features[j]:
                    left_middle.append(j)

        Z['xclust'] = left
        Z['yclust'] = left_middle
        print(Z)
        matrix = Z.as_matrix()
        fig = dendrogram(matrix)
        plt.show()

        # matrix = Z.as_matrix()
        # print(matrix)
        # print(self.tsg.features)
        # for i, j, l, m in matrix:
        #     for k in self.tsg.features:
        #         if i == k:
        #             zero_list
        # return dendrogram(matrix)

    #
    # def compute_dtw(self, vec):
    #     x = self.clusters[vec[0]].centroid_by_eucl
    #     y = self.clusters[vec[1]].centroid_by_eucl
    #     dtw = DTW(x, y, dist=self.dtw_dist)
    #     return vec[0], vec[1], dtw.cost
    #
    # def compute_fastdtw(self, vec):
    #     x = self.clusters[vec[0]].centroid_by_eucl
    #     y = self.clusters[vec[1]].centroid_by_eucl
    #     dtw = FastDTW(x, y, dist=self.dtw_dist, radius=self.radius)
    #     return vec[0], vec[1], dtw.cost
    #
    # def dist_matrix(self):
    #     from multiprocessing.pool import ThreadPool
    #     comb = [i for i in combinations(self.clusters.keys(), 2)]
    #     P = ThreadPool(cpu_count() - 1)
    #     if self.fast:
    #         result = P.map_async(self.compute_fastdtw, comb)
    #         # result = map(self.compute_fastdtw, comb)
    #     else:
    #         result = P.map_async(self.compute_dtw, comb)
    #         # result = map(self.compute_dtw, comb)
    #     x, y, cost = zip(*result.get())
    #     # x, y, cost = zip(*result)
    #     df = pandas.DataFrame([x, y, cost])#, index=['ci, cj', 'cost'])
    #     df = df.transpose()
    #     df.columns = ['ci', 'cj', 'cost']
    #     df.set_index(['ci', 'cj'], inplace=True)
    #     return df.unstack()['cost']
    #
    # @staticmethod
    # def get_pair_to_merge(dist_matrix):
    #     min_value = dist_matrix.min().min()
    #     matrix = dist_matrix[dist_matrix == min_value]
    #     min_val = matrix.dropna(how='all').dropna(how='all', axis=1)
    #     return list(min_val.index)[0], list(min_val.columns)[0], min_val.values[0]
    #
    # def fit(self):
    #     merge_pairs = {}
    #     i = 0
    #     go = True
    #     pool = Pool(cpu_count() - 1)
    #     dct = {}
    #
    #     dist = self.dist_matrix()
    #     print(self.clusters)
    #     print(self.dist_matrix())
        # while (len(self.clusters) != self.nclust) and go:
        #     try:
        #         LOG.info('iteration "{}"'.format(i))
        # #         # print ('\n new iteration')
        # #         # print('go is', go)
        # #         # print('len clusters is', len(self.clusters))
        # #         i += 1
        # #         table = i
        # #         # print(i)
        # #
        # #         ci, cj, distance = self.get_pair_to_merge(dist)
        # #         merge_pairs[i] = (ci, cj)
        # #
        # #         cluster = self.clusters[ci].concat(self.clusters[cj])
        # #
        # #         cluster_size = len(cluster)
        # #         dct[table] = [ci, ci, distance, cluster_size]
        # #
        # #         # print('merged cluster is: ', cluster)
        # #
        # #         # print ('\n\n slf.clusters before update is :')
        # #
        # #         if len(self.clusters) == 1:
        # #             go = False
        # #             continue
        # #         else:
        # #
        # #             del self.clusters[ci]
        # #             del self.clusters[cj]
        # #
        # #             ## add merged cluster to max keys plus 1
        # #             new_key = 0
        # #             while new_key in self.clusters.keys():
        # #                 new_key += 1
        # #
        # #             if new_key in self.clusters.keys():
        # #                 raise ValueError('new key "{}" in keys'.format(new_key))
        # #
        # #             self.clusters[new_key] = cluster
        # #             # pool.map(self.map_to_db, (self.clusters, table))
        # #             # pool.start()
        # #             # pool.join()
        # #             # pool.close()
        # #             # for ci in self.clusters:
        # #             #     self.clusters[ci].cluster = ci
        # #             # self.clusters[ci].to_db(self.db_file, table)
        # #
        # #             # print('\n\n slf.clusters after  update is :')
        # #
        # #             self.evolution_of_clusters[i] = deepcopy(self.clusters)
        # #
        #     except ValueError as e:
        #         LOG.debug(e)
        #         LOG.warning(e)
        #         if 'not enough values to unpack' in str(e):
        #             go = False
        #
        #         else:
        #             raise e
        # print(dct)
        # LOG.info("Results stored in database at '{}'".format(self.db_file))
        # return self.evolution_of_clusters, merge_pairs

    def to_db(self):
        """
        call a tsg to_db method
        :param tsg:
        :return:
        """
        sql_list = []
        for table in self.evolution_of_clusters:
            for cluster in self.evolution_of_clusters[table]:
                cl = self.evolution_of_clusters[table][cluster]
                sql_list.append(cl.to_db(self.db_file, table))


class FindSimilar(object):
    def __init__(self, tsg, x, thresh=0.01):
        self.tsg = tsg
        self.x = x
        self.thresh = thresh

        if self.x not in tsg.features:
            raise ValueError('"{}" not in tsg. These are in tsg: "{}"'
                             ''.format(self.x), self.tsg.features)

        self.x = self.tsg[self.x]

    def __str__(self):
        return "FindSimilar(x='{}', thesh={})".format(
            self.x, self.thresh
        )

    def compute_dtw(self, y):
        return DTW(self.x, y)

    def compute_dtw_parallel(self):
        num_cpu = cpu_count() - 1
        with Pool(num_cpu) as p:
            res = p.map(self.compute_dtw, self.tsg.to_ts())
        return res

    @property
    def dtw(self):
        return [i for i in self.compute_dtw_parallel() if i.cost < self.thresh]

    @property
    def result(self):
        ys = [i.y.feature for i in self.dtw]
        return TimeSeriesGroup(self.tsg.loc[ys])





