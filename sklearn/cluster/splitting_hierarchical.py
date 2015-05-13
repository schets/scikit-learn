"This module contains a basic algorithm for splitting hierarchical clustering"
from itertools import imap, chain

def imapcat(fnc, ins):
    chain.from_iterable(imap(fnc, ins))

def _lazy_clusters(incl, points):
    "a generator which only calls the clusterer when needed"
    for cluster in incl(points): #this should only call incl upon demand
        yield cluster

def _gen_lazy_tree(clfn, ncutoff, points):
    """This lazily generates the tree structure used by clusters
    This only computes clusters on demand, so the clustering algorithm can take
    this tree and progressively process it"""
    if len(points) <= ncutoff:
        return {"cluster" : points}
    else:
        children = (_gen_lazy_tree(clfn, ncutoff, cluster)
                    for cluster in _lazy_clusters(clfn, points))
        return {"children" : children}

def _force_tree(intree):
    "forces evaluation of a tree"
    children = intree.get("children", None)
    if children:
        intree["children"] = [_force_tree(child) for child in children]
    return intree
