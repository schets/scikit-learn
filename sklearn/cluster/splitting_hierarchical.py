"Splitting Hierarchical Clustering"

# Author: Sam Schetterer <samschet@gmail.com>
# License: BSD 3 clause

# Taken from scikit-learn dev branch at github.com/schets/scikit-learn

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MiniBatchKMeans

import heapq as heap
import numpy as np

def _is_clusterer(inobj):
    "Returns true of the passed object is a scikit-learn clusterer"
    return isinstance(inobj, ClusterMixin)

def _make_clusterer_seqfn(incl):
    "Turns a passed clusterer into a sequence-based function"

    def groupfn(inarray, indicies):
        clusters = incl.fit_predict(inarray)
        group_dict = {}
        index = 0
        for (ind, clid) in enumerate(clusters):
            if clid in group_dict:
                group_dict[clid].append(ind)
            else:
                group_dict[clid] = [ind]

        #a way to make this a view...?
        return ((inarray[inds], indicies[inds])
                for inds in group_dict.itervalues())

    return groupfn

class _heap_fit_by(object):
    "class for allowing custom metrics in heap"
    def __init__(self, inpoints, score):
        self.data = inpoints
        self.score = score

    def __less__(self, other): #we want highest scores to be processed
        return self.score > other.score

def _len_predicate(x, clusterer):
    return x

class SplittingClustering(BaseEstimator, ClusterMixin):
    """
    Splitting Clustering

    Recursively splits clusters using a given clusterer

    Parameters
    ----------
    n_clusters : int, default = 2
        The minimum of clusters that will be found.
        The algorithm will split clusters until the numbers
        of clusters is >= n_clusters. If the split that passes
        this number generates many child clusters, they will be used.
        By default, this algorithm uses bisecting kmeans and will return
        exactly n_clusters clusters

    cluster_cutoff : int, default 1
        The size of a list of points for which
        the algorithm will cease clustering

    clusterer : scikit-learn clusterer or callable, default MiniBatchKmeans
        The clusterer used to cluster the data.
        A passed scikit-learn clusterer will only accept datatypes that
        can perform advanced indexing on rows (i.e. arr[[1, 2, 3]])
        If passed a callable, the callable should return a sequence of clusters.
        In addition, the callable should expect to be able to take
        the input passed to the fit function and the cluster type it returns
        along with the the current set of indicies (1d numpy array)


    keep_hierarchy : boolean, default False
        If True then the tree constructed by the algorithm will be kept,
        but only the parts required to create the clusters.
    """
    def __init__(self,
                 n_clusters=2,
                 split_predicate=_len_predicate,
                 clusterer=None,
                 cluster_cutoff=1,
                 keep_hierarchy=False):

        if clusterer is None:
            clusterer = MiniBatchKMeans(n_clusters=2)
        if _is_clusterer(clusterer):
            clusterer = _make_clusterer_seqfn(clusterer)

        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.cluster_cutoff = cluster_cutoff
        self.keep_hierarchy = keep_hierarchy

        self._tree = None
        self._finished = False
        self.labels_ = None

    def fit(self, X, y=None):
        """
        Computes the cluster labels and hierarchy

        Parameters
        ----------
        X : array-like or sparse matrix (or whatever the clusterer takes...)

        """
        self._finished = False
        self._labels = np.empty(len(X))
        self._labels[:] = -1
        self._strict_fit()
        return self

    def fit_predict(self, X, y=None):
        """
        Performs clustering on X and returns cluster labels.
        Equivalent to calling fit()

        Parameters
        ----------
        X : array-like or sparse matrix (or whatever the given clusterer takes)

        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X)
        return self.labels_


    def _set_indicies(self, clnum, inds):
        "initializes the labels"
        for i in inds:
            self._labels[i] = clnum

    def _strict_fit(self, X):
        "performs the actual clustering and analysis of the tree"
        nclusters = self.n_clusters

        ssort = lambda x, incl: _heap_fit_by(x, self.split_predicate(x, incl))
        clusters = [_heap_fit_by(X, 0)] #score doesn't matter here
        clnum = 0

        # Keep generating more splits until n_clusters has been satisfied
        # The tree is generated lazily, so n_clusters = 2 will only cluster once
        while clnum < nclusters:

            # ran out of clusters, shouldn't happen unless bad parameters
            # or clusterers are passed.

            to_split = heap.heappop(clusters).data

            to_split = 
            for x in to_split["children"]:
                if "children" not in x: # x is a final cluster
                    self._set_indicies(clnum, to_split["cluster"])
                    clnum += 1
                else: # x is still a candidate for splitting
                    heap.heappush(clusters, ssort(x))

        #update final indices in labels
        for x in clusters:
            dat = x.data
            self._set_indicies(clnum, dat["cluster"])
            if "children" in dat: # clear out extra references to clusters
                del dat["children"]

        self._finished = True
