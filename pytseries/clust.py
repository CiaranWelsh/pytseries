import pandas, numpy
import matplotlib.pyplot as plt
from pytseries.core import TimeSeriesGroup
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from pytseries.dtw import DTW
from tslearn import utils, clustering, neighbors

import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)

LOG.setLevel(logging.INFO)


class _BaseClust(object):
    """
    Clustering methods in `pytseries.clust` inherit from tslearn
    classes. This class provides additional methods for interacting with the
    pytseries classes and plotting
    """
    def __init__(self, tsg):
        self.tsg = tsg
        self.tslearn_formatted_data = self.to_tslearn_format()

    def to_tslearn_format(self):
        return utils.to_time_series_dataset(self.tsg.to_array())

    def plot_clusters(self, legend=True,
                      legend_loc=(1, 0.1), **kwargs):
        tsg_dct = {}
        for label, df in self.labels.groupby(by='cluster'):
            # print (df[df['cluster'] == label]
            clust_features = list(df.index)
            tsg_dct[label] = self.tsg[clust_features]

        figs = {}
        for i in tsg_dct:
            fig = tsg_dct[i].plot(tsg_dct[i].features, legend=legend,
                                  legend_loc=legend_loc, **kwargs)
            plt.plot(self.centers.loc[i].index,
                     self.centers.loc[i]['value'],
                     linestyle='--', color='black',
                     linewidth=4)
            plt.ylabel('C{} (n={})'.format(i, len(tsg_dct[i])))
            figs[i] = fig

        return figs


class TimeSeriesKMeans(_BaseClust, clustering.TimeSeriesKMeans):
    """
    K-means clustering for time-series data. This is a wrapper
    around the implementation in tslearn, the documentation
    for which can be found at http://tslearn.readthedocs.io/en/latest/.

    The majority of the below documentation is directly from the original source. I
    have added the tsg parameter which is a `pytseries.TimeSeriesGroup` object

    Parameters
    ----------
    tsg: A `pytseries.TimeSeriesGroup` object
        Data that will be fit.
    n_clusters: int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm stops.
    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.
    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter computation. If "dtw", DBA is used for barycenter
        computation.
    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process. Only used if `metric="dtw"` or `metric="softdtw"`.
    metric_params : dict or None
        Parameter values for the chosen metric. Value associated to the `"gamma_sdtw"` key corresponds to the gamma
        parameter in Soft-DTW.
    dtw_inertia: bool
        Whether to compute DTW inertia even if DTW is not the chosen metric.
    verbose : bool (default: True)
        Whether or not to print information about the inertia while learning the model.
    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global
        numpy random number generator.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.
    cluster_centers_ : numpy.ndarray
        Cluster centers.
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    labels : pandas.DataFrame
        the `labels_` attribute converted into a
        `pandas.DataFrame` with better readability
        than the original `labels_` attribute
    centers: pandas.DataFrame
        Similar to `cluster_centers` but parsed
        into a pandas.DataFrame for better readibility.

    Note
    ----
        If `metric` is set to `"euclidean"`, the algorithm expects a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, verbose=False, random_state=0).fit(X)
    >>> km.cluster_centers_.shape
    (3, 32, 1)
    >>> dists = cdist(X.reshape((50, 32)), km.cluster_centers_.reshape((3, 32)))
    >>> numpy.alltrue(km.labels_ == dists.argmin(axis=1))
    True
    >>> numpy.alltrue(km.labels_ == km.predict(X))
    True
    >>> numpy.alltrue(km.fit(X).predict(X) == km.fit_predict(X))
    True
    >>> km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, max_iter_barycenter=5, verbose=False, \
                                  random_state=0).fit(X)
    >>> km_dba.cluster_centers_.shape
    (3, 32, 1)
    >>> dists = cdist_dtw(X, km_dba.cluster_centers_)
    >>> numpy.alltrue(km_dba.labels_ == dists.argmin(axis=1))
    True
    >>> numpy.alltrue(km_dba.labels_ == km_dba.predict(X))
    True
    >>> numpy.alltrue(km_dba.fit(X).predict(X) == km_dba.fit_predict(X))
    True
    >>> km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5, max_iter_barycenter=5, \
                                   metric_params={"gamma_sdtw": .5}, verbose=False, random_state=0).fit(X)
    >>> km_sdtw.cluster_centers_.shape
    (3, 32, 1)
    >>> dists = cdist_soft_dtw(X, km_sdtw.cluster_centers_, gamma=.5)
    >>> numpy.alltrue(km_sdtw.labels_ == dists.argmin(axis=1))
    True
    >>> numpy.alltrue(km_sdtw.labels_ == km_sdtw.predict(X))
    True
    >>> numpy.alltrue(km_sdtw.fit(X).predict(X) == km_sdtw.fit_predict(X))
    True
    >>> TimeSeriesKMeans(n_clusters=101, verbose=False, random_state=0).fit(X).X_fit_ is None
    True
    """
    def __init__(self, tsg, n_clusters=3, **kwargs):
        _BaseClust.__init__(self, tsg)
        clustering.TimeSeriesKMeans.__init__(self, n_clusters=n_clusters, **kwargs)
        self.tsg = tsg
        self.n_cluster = n_clusters
        self.kwargs = kwargs

        self.fit(self.tslearn_formatted_data)

        self.labels = self.convert_labels_to_features()
        self.centers = self.format_centers()

    def convert_labels_to_features(self):
        """
        Converts list of numbers representing cluster assignmetn
        to dict[feature] = assignment
        :return:
        """
        return pandas.DataFrame(dict(zip(self.tsg.features, self.labels_)),
                                index=['cluster']).transpose()

    def format_centers(self):
        """

        :return:
        """
        df_dct = {}
        for i in range(len(self.cluster_centers_)):
            df = pandas.DataFrame(self.cluster_centers_[i])
            df.columns = ['time', 'value']
            df = df.set_index('time')
            df_dct[i] = df
        df = pandas.concat(df_dct)
        df.index.name = 'cluster'
        return df

    # def format_inertia(self):
    #     print(self.inertia_)




class KNearestNeighbours(_BaseClust, neighbors.KNeighborsTimeSeries):
    """Unsupervised learner for implementing neighbor searches for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    metric : {'dtw', 'euclidean', 'sqeuclidean', 'cityblock'} (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [3, 3, 2, 0], [1, 2, 2, 4]]
    >>> knn = KNeighborsTimeSeries(n_neighbors=1).fit(time_series)
    >>> dist, ind = knn.kneighbors([[1, 1, 2, 2, 2, 3, 4]], return_distance=True)
    >>> dist
    array([[ 0.]])
    >>> ind
    array([[0]])

    """

    def __init__(self, tsg, n_neighbors=5, metric="dtw", metric_params=None):
        neighbors.KNeighborsTimeSeries.__init__(self, n_neighbors=n_neighbors, metric=metric,
                                                metric_params=metric_params)
        _BaseClust.__init__(self, tsg)
        self.metric = metric
        self.metric_params = metric_params
        print(self.tslearn_formatted_data)

        self.do()

    def do(self):
        self.fit(self.tslearn_formatted_data)
        print(self.kneighbors(self.tslearn_formatted_data))



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



















