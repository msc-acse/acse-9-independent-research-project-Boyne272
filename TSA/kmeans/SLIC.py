# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:28:20 2019
"""


import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from ..tools import progress_bar


class bin_base():
    """
    Class to hold methods for binning vectors into a regular grid.

    bin_base(bin_grid, dim_x, dim_y)

    Parameters
    ----------

        bin_grid : tuple
            The number of grid cordinates in the x and y directions. Must be
            length 2.

        dim_x, dim_y : ints
            The dimensions of the whole grid

    """

    def __init__(self, bin_grid, dim_x, dim_y):

        # validate input
        assert len(bin_grid) == 2, "bin grid must be 2d"
        assert isinstance(bin_grid[0], int), "grid must be integers"
        assert dim_x % bin_grid[0] == 0, \
            'x grid must be multiple of img x size'
        assert dim_y % bin_grid[1] == 0, \
            'x grid must be multiple of img x size'

        # store number of bin divisions in x and y and total divisions
        self._nx, self._ny = bin_grid
        self._nk = self._nx * self._ny

        # store the size of each bin in x and y
        self._bin_dx = dim_x / self._nx
        self._bin_dy = dim_y / self._ny

        # create adjasent bins list
        self._adj_bins = self._find_adjasent_bins()


    def _bin_vectors(self, vecs):
        """
        Bin vectors with cordinates in the first two axis (ie. x, y, ...)
        into the bin grid
        """
        x_bins = (vecs[:, 0] / self._bin_dx).floor()
        y_bins = (vecs[:, 1] / self._bin_dy).floor()
        output = y_bins * self._nx + x_bins
        return output.long()


    def _find_adjasent_bins(self):
        """
        Create a list of which bins are adjasent to which bins
        """
        adj_bins = []
        for i in range(self._nk):

            # find the cordinates of each bin in the bin grid
            x, y = self._index_to_cords(i)

            # find the neightbours of that cordinate
            cordinates = self._neighbours(x, y, self._nx, self._ny)

            # convert the cordinates back into an index
            indexs = [self._cords_to_index(x_, y_) for x_, y_ in cordinates]

            # store the indexs in tensor form
            adj_bins.append(torch.tensor(indexs))

        return adj_bins


    def _neighbours(self, x, y, x_max, y_max, r=1):
        """
        Find the neighbours in radius r to the x and y cordinate, here
        this includes itself as a neighbour as one needs to search
        all vectors in the current and neighbouring bins
        """
        return [(x_, y_)
                for x_ in range(x-r, x+r+1)
                for y_ in range(y-r, y+r+1)
                if ((0 <= x_ < x_max) and # not outside x range
                    (0 <= y_ < y_max))]   # not outside y range


    def _index_to_cords(self, i):
        "convert bin point's index to bin x, y cordinate"
        return i%self._nx, int(i/self._nx)


    def _cords_to_index(self, x, y):
        "convert bin point's x, y cordinates to bin index"
        return y * self._nx + x



class SLIC(bin_base):
    """
    Implements Kmeans clustering on an image in 5d color position space
    using localised bins on a regular grid to enforce locality.

    SLIC(img, bin_grid, dist_metric=None, dist_metric_args=[1.,])

    Parameters
    ----------

        img : 3d numpy array
            Image to be segmented by kmeans,
            expected in shape [x_dim, y_dim, rgb] with all values in the
            interval [0, 1], not 255.

        bin_grid : tuple length 2
            The number of initial partitions in x and y respectivly,
            therefore the number of segments is the product of these.
            Both must be a factor of the x, y dimensions for the whole
            image. This intial segmentation restrains the kmeans centers
            in space, forcing locality of segments and speeding up the
            algorithm.

        dist_metric : function, optional
            The method of calculating distance between vectors and cluster
            centers. This must have form f(vecs, clusts, *args) where vecs and
            clusts are rank 2 tensors with samples on the first axis. This
            should return a rank 2 tensor of distances with shape (clusters, vectors)
            such that taking the argmin along dim=1 gives an array of closest
            cluster centroids for each vector.

        dist_metric_args : tuple, optional
            arguments to passed into dist_metric function. By defualt this is
            a single parameter for the bia to color over distance.

    """

    def __init__(self, img, bin_grid, dist_metric=None,
                 dist_metric_args=None):

        # validate inputs
        assert img.ndim == 3, "image must be a 3d array"
        assert img.shape[-1] == 3, "image must be of form (x, y, rgb)"

        # store given parameters
        self.img = img
        self._dim_y, self._dim_x = img.shape[:2]  # store image dimensions
        self._np = self._dim_y * self._dim_x      # the number of pixels
        self.vectors = self._img_to_vectors()

        # bin_base parent which handels the locality of clusters and vectors
        bin_base.__init__(self, bin_grid, self._dim_x, self._dim_y)

        # tensor of which bin each vector belongs to
        self._vec_bins = self._bin_vectors(self.vectors)

        # list of which vectors are in each bin
        self._bins_list = [(self._vec_bins == i).nonzero().squeeze()
                           for i in range(self._nk)]

        # tensor of which cluster each vector belongs to
        self.vec_clusts = self._vec_bins.clone()

        # list of which vectors are in each cluster
        self._cluster_contense = [(self.vec_clusts == i).nonzero().squeeze()
                                  for i in range(self._nk)]

        # initialise the centroids
        self.centroids = torch.empty([self._nk, 5], dtype=torch.float)
        self._update_centroids()

        # create attributes for later
        self.distances = []
        self._progress_bar = None


        # defualt distance function
        def dist(vecs, clusts, *args):
            """
            Find the distance between every vector and every cluster center
            given.  Here this is done by scaling the distance space by width
            of the bin grid.

            vecs and clusts are 2 rank tensors with shape (samples, features).
            returns a rank 2 tensor of distances with shape (clusters, vectors)
            such that taking the argmin along dim=1 gives an array of closest
            cluster centroids for each vector
            """
            col_clust, col_vec = clusts[:, 2:], vecs[:, 2:]
            pos_clust, pos_vec = clusts[:, :2], vecs[:, :2]

            col_dist = (col_clust - col_vec[:, None]).norm(dim=2)
            pos_dist = (pos_clust - pos_vec[:, None]).norm(dim=2)

            return col_dist + (args[0] * pos_dist /
                               np.sqrt(self._np / self._nk))

        # use the dist_metric if given
        self.dist_func = dist_metric if dist_metric else dist
        self.dist_metric_args = dist_metric_args if dist_metric_args else [1.,]


    # ============================================
    # Iteration methods
    # ============================================


    def _update_centroids(self):
        """
        Find the new center of each cluster as the mean of its constituent
        elemtens
        """
        for i in range(self._nk):
            vecs_in_cluster = self._cluster_contense[i]
            assert vecs_in_cluster.numel() > 0, 'no cluser should be empty'
            self.centroids[i] = self.vectors[vecs_in_cluster].mean(dim=0)


    def _update_distances(self):
        """
        Find the ditance between every vector in a bin and every centroid
        in that or the neighbouring bins.
        """

        # bin the centroids
        cent_bins_tensor = self._bin_vectors(self.centroids)

        # reset the distances
        self.distances = []

        # for every bin grid (same as number of centroids)
        for i in range(self._nk):

            # get relevant centroids and vectors
            centroids_to_search = self._adj_bins[i]
            vecs_in_bin = self._bins_list[i]

            # find distance between vectors in this bin and local centroids
            vecs_tensors = self.vectors[vecs_in_bin]
            cents_tensors = self.centroids[centroids_to_search]
            dist = self.dist_func(vecs_tensors, cents_tensors,
                                  *self.dist_metric_args)

            # store this distance
            self.distances.append(dist)


    def _update_clusters(self):
        """
        Find which vectors belong to which cluster by taking the mimunum of
        the distances.
        """

        # for every bin grid (same as number of centroids)
        for i in range(self._nk):

            # get relevant centroids and vectors
            vecs_in_bin = self._bins_list[i]
            centroids_to_search = self._adj_bins[i]

            # find which centroids are the closest and update them
            min_indexs = torch.argmin(self.distances[i], dim=1)
            min_clusters = centroids_to_search[min_indexs]
            self.vec_clusts[vecs_in_bin] = min_clusters

        # re-allocate which vectors are in each cluster
        for i in range(self._nk):
            self._cluster_contense[i] = (self.vec_clusts == i).nonzero().squeeze()


    def iterate(self, n_iter):
        "loop for n_iter iterations with progress bar"

        # create the progress bar
        self._progress_bar = progress_bar(n_iter)

        for i in range(n_iter):
            self._update_distances()
            self._update_clusters()
            self._update_centroids()
            self._progress_bar(i)


    # ================================================
    # Utility methods
    # ================================================


    def get_segmentation(self):
        """
        Returns the current segmentation mask as a numpy array
        """
        return self._tensor_to_mask(self.vec_clusts)


    def _tensor_to_mask(self, tensor):
        return tensor.view(self._dim_y, self._dim_x).cpu().numpy()


    def _img_to_vectors(self):
        """
        Convert the given 3d image array (x, y, rgb) into an array of 5d vectors
        (x, y, r, g, b)
        """

        # the x and y cordinates
        X, Y = np.meshgrid(range(self._dim_x),
                           range(self._dim_y))

        # create the 5d vectors
        vecs = np.zeros([self._np, 5])
        vecs[:, 0] = X.ravel()
        vecs[:, 1] = Y.ravel()
        vecs[:, 2] = self.img[:, :, 0].ravel()
        vecs[:, 3] = self.img[:, :, 1].ravel()
        vecs[:, 4] = self.img[:, :, 2].ravel()

        return torch.from_numpy(vecs).float()


    # =======================================
    # Visulisation methods
    # =======================================


    def plot(self, option='default', ax=None):
        """
        Plot a one fo the following options on an axis if specified:
            - 'default' image and the segement outlines
            - 'setup' image, bin edges and cluster centers
            - 'segments' the sedment mask
            - 'edges' just the segment outlines in a transparent manner
            - 'img' just the orinal image
            - 'centers' each kmean centroid
            - 'bin_edges' the bin mesh used
            - 'bins' both 'img' and 'bin_edges'
            - 'time' the iterations vs time

        If path is given the image will be saved on that path.

        If no axis is given one will be made.
        """

        # validate opiton
        valid_ops = ['default', 'setup', 'edges', 'img', 'centers', 'bins',
                     'time', 'bin_edges', 'segments', 'setup']
        assert option in valid_ops, "option not recoginsed"

        # create an axis if not given
        if ax is None:
            _fig, ax = plt.subplots(figsize=[15, 15])

        # plot options
        if option == 'default':
            self.plot('img', ax)
            self.plot('edges', ax)
            ax.set(title='Image Segmentation (default)')

        elif option == 'setup':
            self.plot('img', ax)
            self.plot('bin_edges', ax)
            self.plot('centers', ax)
            ax.set(title='Image Segmentation (setup)')

        elif option == 'segments':
            mask = self._tensor_to_mask(self.vec_clusts)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (segments)')

        elif option == 'edges':
            mask = self._tensor_to_mask(self.vec_clusts)
            mask = self._outline(mask)
            mask = self._mask_to_rgba(mask)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (edges)')

        elif option == 'img':
            ax.imshow(self.img)
            ax.set(title='Image Segmentation (img)')

        elif option == 'centers':
            ax.plot(self.centroids[:, 0].cpu().numpy(),
                    self.centroids[:, 1].cpu().numpy(), 'm*', ms=20)
            ax.set(title='Image Segmentation (centers)')

        elif option == 'bin_edges':
            mask = self._tensor_to_mask(self._vec_bins)
            mask = self._outline(mask)
            mask = self._mask_to_rgba(mask)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (bin_edges)')

        elif option == 'bins':
            mask = self._tensor_to_mask(self._vec_bins)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (bins)')

        elif option == 'time':
            assert hasattr(self, '_progress_bar'), \
                'must call iterate to use this'
            plt.close()
            self._progress_bar.plot_time()
            ax.set(title='Image Segmentation (time)')


    def _mask_to_rgba(self, mask, color='r'):
        "Take a 2d mask and return a 3d rgba mask for imshow overlaying"

        # validate input
        assert isinstance(mask, type(np.array([]))), \
                          "mask must be a numpy array"
        assert mask.ndim == 2, "mask must be a 2d array"
        assert color in ['r', 'g', 'b'], 'color must be r, g or b'

        # create the rgba arrray
        zeros = np.zeros_like(mask)
        rgba = np.dstack([zeros, zeros, zeros, mask])

        # set the correct color to be true
        i = ['r', 'g', 'b'].index(color)
        rgba[:, :, i] = mask

        return rgba


    def _outline(self, mask):
        """
        Take a 2d mask and use a laplacian convolution to find the segment
        outlines
        """

        # validate input
        assert isinstance(mask, type(np.array([]))), \
                          "mask must be a numpy array"
        assert mask.ndim == 2, "mask must be a 2d array"

        # convolve laplacian with mask
        laplacian = np.array([[1., 1., 1.],
                              [1., -8., 1.],
                              [1., 1., 1.]])
        edges = sig.convolve2d(mask, laplacian, mode='valid')
        return (edges > 0).astype(float) # any non-zero is an edge


if __name__ == '__main__':
    # run an example
    from tools import get_img

    # setup
    image = get_img("images/example_white.tif")
    obj = SLIC(image, [20, 20])

    # plot the initial binning
    obj.plot("setup")
    plt.gca().set(title='Initial Grid')

    # iterate
    obj.iterate(10)

    # plot the resulting segmentation
    obj.plot('default')
    plt.gca().set(title='Segmentation after 10 Iterations')

    # plot the time taken
    obj.plot('time')
