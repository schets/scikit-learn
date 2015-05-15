"Splitting Hierarchical Clustering"

# Author: Sam Schetterer <samschet@gmail.com>
# License: BSD 3 clause

from ..base import BaseEstimator, ClusterMixin
from kmeans import MiniBatchKMeans

import heapq as heap
import numpy as np

def _lazy_clusters(incl, points):
    "a generator which only calls the clusterer when needed"
    #this should only call incl upon demand, should a timing unit test
    clusters = incl(points)
    del points #remove extra reference to points as soon as possible
    for cluster in clusters:
        yield cluster

def _gen_lazy_tree(clfn, ncutoff, points_inds):
    """
    This lazily generates the tree structure used by clusters
    This only computes clusters on demand, so the clustering algorithm can take
    this tree and progressively process it
    """
    (points, indicies) = points_inds
    if len(points) <= ncutoff:
        return {"cluster" : indicies}
    else:
        #since _gen_lazy_tree itself is a generator,
        #it won't call cluster until actually iterated
        #tested 2.7 and 3.2
        children = (_gen_lazy_tree(clfn, ncutoff, cluster)
                    for cluster in _lazy_clusters(clfn, points))
        return {"children" : children, "cluster" : indicies}

def _force_tree(intree, modfn):
    "forces evaluation of a tree"
    children = intree.get("children", None)
    if children:
        intree["children"] = [_force_tree(child, modfn) for child in children]
    return modfn(intree)

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
    def __init__(self, inpoints, scorefn):
        self.data = inpoints
        self.score = scorefn(inpoints)

    def __less__(self, other): #we want highest scores to be processed
        return self.score > other.score

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
        but only the parts required to create the clusters. See complete_tree

    complete_hierarchy : boolean, default False
        If True (and keep_hierarchy is True) then the entire tree
        will be calculated. If this is enables, you must ensure that
        each group of points will be clusterable - for example,
        cluster_cutoff must be greater then the number of clusters the
        passed clusterer will expect to recieve

    keep_clusters : boolean, default False
        If True, then the indicies of datapoints contained in the children
        of each branch will be stored with the branch. Otherwise, point data
        will only be stored in the leafs
    """
    def __init__(self,
                 n_clusters=2,
                 clusterer=None,
                 cluster_cutoff=1,
                 keep_hierarchy=False,
                 complete_hierarchy=False,
                 keep_clusters=False):

        if clusterer is None:
            clusterer = MiniBatchKMeans(n_clusters=2)
        if _is_clusterer(clusterer):
            clusterer = _make_clusterer_seqfn(clusterer)

        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.cluster_cutoff = cluster_cutoff
        self.keep_hierarchy = keep_hierarchy
        self.complete_hierarchy = complete_hierarchy
        self.keep_clusters = keep_clusters
        self.store_tree = keep_hierarchy and not complete_hierarchy

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
        self._make_tree(X)
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


    def _make_tree(self, points):
        "generates the lazy tree"
        initial_inds = np.linspace(0, len(points) - 1, len(points))
        self._tree = _gen_lazy_tree(self.clusterer,
                                    self.cluster_cutoff,
                                    (points, initial_inds))
        if self.keep_hierarchy and self.complete_hierarchy:
            if self.keep_clusters:
                self._tree = _force_tree(self._tree, lambda x: x)

    def _set_indicies(self, clnum, inds):
        "initializes the labels"
        for i in inds:
            self._labels[i] = clnum

    def _strict_fit(self):
        "performs the actual clustering and analysis of the tree"
        tree = self._tree
        assert tree is not None, "Tree has not been generated"

        nclusters = self.n_clusters

        # Ensure that an initial clustering was performed
        if "children" not in tree and nclusters > 1:
            errmess = "Unable to perform any clustering"
            raise ValueError(errmess)

        clsize = lambda x: len(x["cluster"])
        ssort = lambda x: _heap_fit_by(x, clsize)
        clusters = [ssort(tree)]
        clnum = 1

        # Keep generating more splits until n_clusters has been satisfied
        # The tree is generated lazily, so n_clusters = 2 will only cluster once
        while clnum < nclusters:

            # ran out of clusters, shouldn't happen unless bad parameters
            # or clusterers are passed.
            if len(clusters) == 0:
                errm = "Was unable to generate {0} clusters from input"
                raise ValueError(errm.format(nclusters))

            to_split = heap.heappop(clusters).data

            if self.store_tree: # make child storage for each branch permanent
                to_split["children"] = list(to_split["children"])

            for x in to_split["children"]:
                if "children" not in x: # x is a final cluster
                    self._set_indicies(clnum, to_split["cluster"])
                    clnum += 1
                else: # x is still a candidate for splitting
                    heap.heappush(clusters, ssort(x))

        #update final indices in labels
        if not self.keep_clusters:
            for x in clusters:
                dat = x.data
                if "children" in dat: # clear out extra references to clusters
                    del dat["clusters"]
                self._set_indicies(clnum, dat["cluster"])
        else:
            for x in clusters:
                self._set_indicies(clnum, dat["cluster"])
 
        self._finished = True
