# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:27:33 2019
"""


import torch
import matplotlib.pyplot as plt
import numpy as np
from ..tools import progress_bar
from .SLIC import SLIC


class MSLIC_wrapper():
    """
    Wrapper around SLIC to run multiple instances at once, using a the
    combined of each instance to assign clusters in every iteration

    MSLIC_wrapper(imgs, bin_grid, combo_metric='max', combo_metric_args=[],
                  dist_metric=None, dist_metric_args=[1.,])

    Parameters
    ----------

    imgs : list of 3d numpy arrays
        The imgs on which the SLIC algorithm is to be run in parallel.
        Each must have the same shape and have a

    bin_grid : tuple length 2 (same as SLIC)
        The number of initial partitions in x and y respectivly,
        therefore the number of segments is the product of these.
        Both must be a factor of the x, y dimensions for the whole
        image. This intial segmentation restrains the kmeans centers
        in space, forcing locality of segments and speeding up the
        algorithm.

    combo_metric : string or callable, optional
        The metric by which the distances between vectors and centroids
        in each image are to be combined. Allowed string options are:
        - 'max' maximum of the distances in each image
        - 'min' minimum of the distances in each image
        - 'mean' average of the distances in each image
        - 'sum' total of the distances in each image

        Alternatively a callable function can be used, this function must
        have form f(t, *args) where t is a rank 3 tensor of distances with
        shape (img, vectors, clusters) and return a rank 2 tensor of distances
        with shape (vectors, clusters).

    combo_metric_args : tuple, optional
        arguments to passed into combo_metric function if given

    dist_metric : function, optional
            Passed into SLIC objects.
            The method of calculating distance between vectors and cluster
            centers. This must have form f(vecs, clusts, *args) where vecs and
            clusts are rank 2 tensors with samples on the first axis. This
            should return a rank 2 tensor of distances with shape (clusters, vectors)
            such that taking the argmin along dim=1 gives an array of closest
            cluster centroids for each vector
    
    dist_metric_args : tuple, optional
        Passed into SLIC objects.
        arguments to passed into dist_metric function if given

    """

    _default_combo_metrics = {'max': lambda t, *args: t.max(dim=0)[0],
                              'min': lambda t, *args: t.min(dim=0)[0],
                              'mean': lambda t, *args: t.mean(dim=0),
                              'sum': lambda t, *args: t.sum(dim=0)}

    def __init__(self, imgs, bin_grid, combo_metric='max',
                 combo_metric_args=None, dist_metric=None,
                 dist_metric_args=[1.,]):

        # validate all given images
        for img in imgs:
            assert img.shape == imgs[0].shape, "images must be the same shape"

        # create the SLIC objects
        self.SLIC_objs = [SLIC(img, bin_grid, dist_metric, dist_metric_args)
                          for img in imgs]

        # store combo metric arguments and set attribute for later
        self.metric_args = combo_metric_args if combo_metric_args else []
        self._progress_bar = None

        # set the given combo_metric if given
        if callable(combo_metric):
            self.metric = combo_metric
        else:
            # check the sting matches one of the defualts
            assert combo_metric in self._default_combo_metrics.keys(), \
                "metric " + combo_metric + " not recognised"
            # set to the default combo metric
            self.metric = self._default_combo_metrics[combo_metric]


    def _combined_distance(self):
        """
        Find the combined distance for every pixel in every bin using the
        given combo_metric
        """

        # reset the distances
        combo_dist = []

        # collect the distances for every SLIC in every bin
        all_distances = [obj.distances for obj in self.SLIC_objs]

        # zip together the bins and loop over them
        for dists in zip(*all_distances):

            # calculate the combined distance for this bin using combo_metric
            dist_tensor = torch.stack(dists)
            combo_dist.append(self.metric(dist_tensor))

        return combo_dist


    def iterate(self, n_iter):
        "loop for n_iter iterations with progress bar"

        # create the progress bar
        self._progress_bar = progress_bar(n_iter)

        for i in range(n_iter):

            # iterate up to finding distances for each kmeans object
            for obj in self.SLIC_objs:
                obj._update_distances()

            # find the combined distance
            combo_dist = self._combined_distance()

            # finish the iteration for each SLIC
            for n_obj, obj in enumerate(self.SLIC_objs):

                # set the combined distance
                obj.distances = combo_dist

                # only calculate the new clusters once as this is same for all
                if n_obj == 0:
                    obj._update_clusters()
                else:
                    obj.vec_clusts = self.SLIC_objs[0].vec_clusts
                    obj._cluster_contense = self.SLIC_objs[0]._cluster_contense

                # update the centroid means
                obj._update_centroids()

            # print the progress bar
            self._progress_bar(i)


    def get_segmentation(self):
        """
        Returns the current segmentation mask as a numpy array
        """
        return self.SLIC_objs[0].get_segmentation()


    def plot(self, objs=[], option='default', axs=[None], path=''):
        """
        Calls obj.plot() on each of the SLIC objects specified with option
        given

        Parameters
        ----------

        obj_indexs : list of ints, optional
            which of the SLIC objects to call plot on. If left empty this
            will plot all SLIC objects.

        option : string, optional
            the plot option to be passed to obj.plot, refer to the docstrings
            there for available options

        axs : tuple of matplotlib axis, optional
            the axis to be plotted on, if not given a subplot for each of
            the SLIC objs is used.

        path : string, optional
            the path to save the figure on if wanted
        """

        # if the time plot is wanted use time plot on this wrapper
        if option == 'time':
            self._progress_bar.plot_time()
            return

        # if default objs set then replace will list of all indexs
        if not objs:
            objs = range(len(self.SLIC_objs))

        # set the axis if not given
        if not any(axs):
            _fig, axs = plt.subplots(len(objs), 1,
                                     figsize=[len(objs)*22, 22])
            axs = np.array([axs]).ravel() # force it be an array even if N = 1

        # call the plot routenes
        for i, _ax in zip(objs, axs):
            self.SLIC_objs[i].plot(option, ax=_ax)

        # save the figure if wanted
        if path:
            plt.savefig(path)


if __name__ == '__main__':
    # run an example of SLIC on two images, then MSLIC on both
    from tools import get_img
    
    # setup
    grid = [20, 20]
    img_white = get_img("images/example_white.tif")
    img_polar = get_img("images/example_polar.tif")

    # iterate SLIC with just the white image
    obj_white = SLIC(img_white, grid)
    obj_white.iterate(5)
    obj_white.plot()

    # iterate SLIC with just the polar image
    obj_polar = SLIC(img_polar, grid)
    obj_polar.iterate(5)
    obj_polar.plot()

    # iterate MSLIC with both white and polar images
    obj_both = MSLIC_wrapper([img_white, img_polar], grid)
    obj_both.iterate(5)
    obj_both.plot()
