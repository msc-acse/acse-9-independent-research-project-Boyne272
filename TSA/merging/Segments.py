# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Tue Aug 27 18:37:15 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import morphology
from ..tools import progress_bar


# -----------------------------------------------------------------------------


class Mask_utilities():

    """
    A collection of utilities for minipulating a mask, such as getting its
    outline or creating an rgba array for plotting


    Mask_utilities(mask, connected=True)


    Parameters
    ----------

    mask : 2d numpy array
        integer array to be used

    connected : bool, optional
        if true will enforce connectivity of each region in the mask
        via skimgs morphology package. Defaults to true.
    """

    # laplacian array
    _laplacian_8 = np.array([[1., 1., 1.],
                             [1., -8., 1.],
                             [1., 1., 1.]])
    _laplacian_4 = np.array([[0., 1., 0.],
                             [1., -4., 1.],
                             [0., 1., 0.]])
    _laplacian_h = np.array([[1., 2., 1.],
                             [0., 0., 0.],
                             [-1., -2., -1.]])
    _laplacian_v = np.array([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]])


    def __init__(self, mask, connected=True):

        # validate input
        assert mask.ndim == 2, 'Must be a 2d mask'
        assert mask.dtype == np.int64, 'Must be an integer array'

        # enforce connectivity if wanted
        if connected:
            mask = self._enforce_connectivity(mask)

        # store the original mask for refrence
        self.orig_mask = mask.copy()

        # store the mask
        self.mask = mask

        # store the dimensions
        self._ydim, self._xdim = mask.shape


    def _rgba(self, mask=None, color='r', opaqueness=1):
        """
        Take a 2d mask and return a 4d rgba mask for imshow overlaying
        with the given color and opaqueness.
        """

        # if no mask is given use the saved one
        if mask is None:
            mask = self.mask

        # validate input
        assert mask.ndim == 2, 'Must be a 2d mask'

        # create the transparent mask
        zeros = np.zeros_like(mask)
        rgba = np.dstack([zeros, zeros, zeros, mask*opaqueness])

        # set the correct color channel
        i = ['r', 'g', 'b'].index(color)
        rgba[:, :, i] = mask

        return rgba


    def _outline(self, diag=True, original=False, multi=True):
        """
        Take the stored mask and use a laplacian convolution to find the
        outlines for plotting. diag decides if diagonals are to be
        included or not, original decides if the original mask should
        be used or not.

        multi is an option to do horizontal, vertical and multi_directional
        laplacians and combine them. This is a safer method as particular
        geometries can trick the above convolution.
        """
        # select the correcy arrays based on the options given
        lap = self._laplacian_8 if diag else self._laplacian_4
        mask = self.mask if not original else self.orig_mask

        # do the convolution to find the edges
        conv = convolve2d(mask, lap, mode='valid').astype(bool)

        if multi:
            conv2 = convolve2d(mask, self._laplacian_h,
                               mode='valid').astype(bool)
            conv3 = convolve2d(mask, self._laplacian_v,
                               mode='valid').astype(bool)
            conv = (conv + conv2 + conv3).astype(bool)


        # pad back boarders to have same shape as original image
        conv = np.pad(conv, 1, 'edge')

        return conv.astype(float)


    def _enforce_connectivity(self, mask):
        """
        Partition the given mask so that no section is disjoint
        """
        return morphology.label(mask, neighbors=8)


# -----------------------------------------------------------------------------


class segment_group(Mask_utilities):
    """
    Object that handles a feature extraction and merging of a segment mask
    though the use of segment instances. Upon initalisation these segments
    locate their neighbours, edges and enforce connectivity of each segment.
    Note the neighbour identification is a costly process.


    segment_group(mask)


    Parameters
    ----------

    mask : 2d numpy array
        integer array of a mask governing a segmentation

    """

    def __init__(self, mask):

        # parent validates input and creates the mask, orig_mask attributes
        Mask_utilities.__init__(self, mask, True)

        # create the segment objects and the directory which tracks merges
        self.seg_dict = self._create_segments(np.unique(self.mask), {})
        self._directory = {n:n for n in self.seg_dict.keys()}

        # store the group of each segment
        self.seg_clusters = {}


    def _create_segments(self, seg_ids, segs):
        """
        Initalise the segment objects from the list of ids is passed with the
        current mask. When updating then only pass segments to be recreated
        and give the segment dictionary to store the newly made segement.
        """

        # verbose
        N = len(seg_ids)
        prog_bar = progress_bar(N)
        print('Initalising %i segments' % N)

        # calculate edges once
        mask_edges = self._outline()

        # for every segment to create
        for n, i in enumerate(seg_ids):

            # update progress bar
            prog_bar(n)

            # find segments pixel cords
            bool_arr = self.mask == i
            cords = np.array(np.where(bool_arr)[::-1]).T
            # [::-1] as np.where returns y, x format
            # T as we want in form (N, 2)

            # find the segments edges (note * works as the & operator)
            bool_arr = bool_arr * mask_edges
            edges = np.array(np.where(bool_arr)[::-1]).T

            # create the segments
            segs[i] = (segment(cords, edges, i, self))

        # leave a space after the progress bar
        print('\n')

        return segs


    def _merge_segments(self, to_merge):
        """
        To merge needs to be an array of id pairs that are to be merged.
        For every pair the 2nd id is replace with the 1st one in self.mask.
        Then only the relevant segment objects are removed or recaluclated as
        required.

        Caution is taken to keep track of who has merged so that merging with
        a segment which has already been merged is handelled appropirately.
        """

        # stores which segments have changed
        to_check = set()

        # for each merhing pair
        for seg_id1, seg_id2 in to_merge:

            # select the up to date ids
            id1 = self._directory[seg_id1]
            id2 = self._directory[seg_id2]

            # update the working mask
            self.mask[self.mask == id2] = id1

            # update the segment directory
            for key, val in self._directory.items():
                if val == id2:
                    self._directory[key] = id1

            # track which segments to recalculate
            to_check.update(self.seg_dict[id1].neighbours +
                            self.seg_dict[id2].neighbours)


        # stores which segments exist but need recalculating
        to_recalc = []

        # delete the old segment objects
        for seg_id in to_check:
            if self._directory[seg_id] != seg_id:
                del self.seg_dict[seg_id]
            else:
                to_recalc.append(seg_id)

        # print exactly how many segments were removed
        print(len(to_check) - len(to_recalc), 'segments merged')

        # recreate those that still exist
        self._create_segments(to_recalc, self.seg_dict)


    def enforce_size(self, min_size=50):
        """
        Merge small sections below the given area with their largest neighbour
        """

        # stores which segments are to be merged
        to_merge = []

        # for every segment
        for _id, seg in self.seg_dict.items():

            # if too small
            if len(seg.cords) < min_size:

                # find the biggest neighbour
                neigh_sizes = [len(self.seg_dict[n_id].cords)
                               for n_id in seg.neighbours]
                max_index = np.argmax(neigh_sizes)
                max_id = seg.neighbours[max_index]

                # record the need to merge
                to_merge.append((max_id, _id))

        # carry out the mergers
        self._merge_segments(to_merge)


    def merge_by_cluster(self, edge_present=1.):
        """
        After a clustering has been assigned this merges segments that are
        adjasent and of the same group.

        If an edge_confidence has been calculated for the segments than the
        extra condition that the edge value is below the edge_present
        parameter is requiered.
        """

        # ensure clusters have been assigned
        assert len(self.seg_clusters) == len(self.seg_dict), \
            "Must assign clusters for the current segments"

        # stores which segments are to be merged
        to_merge = []

        # for every segment neighbour pair
        for _id, seg in self.seg_dict.items():
            for n_id in seg.neighbours:
                neighbour = self.seg_dict[n_id]

                # average the edge confidence between the two segments
                edge_conf = np.mean([seg.conf_dict[n_id],
                                     neighbour.conf_dict[_id]])
                # IF THE CODE BREAKS HERE then seg and neighbour disagree
                # on wether they are neighbours (asserting this is too much
                # wasted computation)

                # if they are of the same cluster and there is no edge
                if (self.seg_clusters[_id] == self.seg_clusters[n_id] and
                        edge_conf < edge_present):

                    # record the need to merge
                    to_merge.append((n_id, _id))

        # carry out the mergers
        self._merge_segments(to_merge)


    def merge_by_edge(self, edge_absent=-1.):
        """
        After an edge_confidence hsa been calculated merge any adjasent pairs
        with edge value less than the edge_absent parameter.
        """
        to_merge = []

        # for every segment neighbour pair
        for _id, seg in self.seg_dict.items():
            for n_id in seg.neighbours:
                neighbour = self.seg_dict[n_id]

                # average the edge confidence between the two segments
                edge_conf = np.mean([seg.conf_dict[n_id],
                                     neighbour.conf_dict[_id]])

                # if there is no apparent edge
                if edge_conf < edge_absent:

                    # record the need to merge
                    to_merge.append((n_id, _id))

        # carry out the mergers
        self._merge_segments(to_merge)


    def feature_extraction(self, extract_func, func_vars):
        """
        Loop over every segment with the extract function to obtain the
        feature vectors for each segement.


        Parameters
        ----------

        extract_func : callable
            callabel which accepts (X_cords, Y_cords, *func_vars) where
            the cordinates are of a single segment and returns a tuple of
            features of that segment. Note that X,Y is  given in the visible
            sense (i.e. an array is indexed [y, x])

        func_vars : tuple of vairables
            vairable to be passed into the fucntion to evaluate the segments
            feaures. This is often some filtered versions of the original
            image to be indexed at the segments cordinates


        Returns
        -------

        feats : 2d array
            array of segment features with shape (n segments, n features)

        """

        # vaidate input
        assert callable(extract_func), "extract_func must be a function"
        assert isinstance(func_vars, (list, tuple)), \
            "func_vars must be a list or tuple"

        # stores the segments features
        feats = []

        # for every segment
        for seg in self.seg_dict.values():

            # find its features
            feats.append(extract_func(seg.cords[:, 0],
                                      seg.cords[:, 1],
                                      *func_vars))

        return np.array(feats)


    def edge_confidence(self, confidence_func, func_vars):
        """
        Loop over every segment with the confidence function to obtain the
        edge confidence between every neighbour pair. This confidence value
        is then stored on each segment and used in the merging functions.

        Parameters
        ----------

        confidence_func : callable
            callabel which accepts (X_cords, Y_cords, *func_vars) where
            cordinates are of the segment edge with one neighbour and
            returns a single value of the confidence of an edges presence.
            By default >1 is consider to be an edge and <-1 is considered
            to be no edge, however this can be overwritten in the mergeing
            functions. Note that X,Y is given in the visible sense
            (i.e. an array is indexed [y, x])

        func_vars : tuple of vairables
            vairable to be passed into the fucntion to evaluate the segments
            edge confidence. This is often some edge detected versions of the
            original image to be indexed at the segments cordinates

        """

        # for every segment
        for seg in self.seg_dict.values():
            for neigh in seg.neighbours:

                # find the edge confidence and store on the segment
                X, Y = seg.edge_dict[neigh][:, 0], seg.edge_dict[neigh][:, 1]
                seg.conf_dict[neigh] = confidence_func(X, Y, *func_vars)


    def assign_clusters(self, clusters):
        """
        Store the clusters of each segment for use in merge_by_cluster. This
        is done seperately so that the segments of each cluster can be plotted
        before the merging is done.
        """

        # valiate input
        assert len(clusters) == len(self.seg_dict), \
            'must have a group for each segment'

        # remove the old clusters
        self.seg_clusters = {}

        # set the new clusters
        for seg_id, cluster in zip(self.seg_dict.keys(), clusters):
            self.seg_clusters[seg_id] = cluster


    def get_cluster_mask(self):
        """
        Returns a mask of the cluster of each segment
        """
        # create output mask
        output = np.full_like(self.mask, np.nan)

        # loop of segments to fill it
        for seg_id, clust in self.seg_clusters.items():
            output[self.mask == seg_id] = clust

        return output


    def plot(self, option='default', ax=None, **kwargs):
        """
        Plot the specified option on the axis if given. All of these are
        designed to be plotted over the original image, either by passing axis
        with this image already on it or by passing the image array as kwarg
        'back_img'.

        Valid options are:

        - 'both' or 'default'
            shows the merged segments edges and the original segment edges in a
            different color so what has been merged is clear.

        - 'orig_edges'
            shows the original segments edges.

        - 'merged_edges'
            shows the merged segments edges.

        - 'cluster'
            plots a mask that is transparent only where segments belong to the
            cluster specified by the kwarg 'cluster'.

        - 'cluster_all'
            plots every cluster as above on a multi axis figure. Must give
            back_img kwarg so that it can be put on each subplot.

        - 'segment'
            plots a mask that shades in the segment id given with the kwarg
            'segment' a slighly translucent red.

        - 'segment_edge'
            plots the edges of the segment id given with the kwarg
            'segment'. These are colored and labeled different for each
            different neighbouring segment (if a pixel has more than one
            neighbour segment adjasent it is random which is indicated)

        - 'edge_conf'
            plot every segment edge in a color that represents its edge
            confidence value.

        """

        # if no axis given create one
        if not ax:
            fig, ax = plt.subplots(figsize=[15, 15])

        # plot background image if given
        if 'back_img' in kwargs.keys():
            ax.imshow(kwargs['back_img'])

        # base options
        if option == 'orig_edges':
            outline = self._outline(original=True)
            ax.imshow(self._rgba(outline, color='r'))
            ax.set(title='original mask outline')

        elif option == 'merged_edges':
            outline = self._outline(original=False)
            ax.imshow(self._rgba(outline, color='g'))
            ax.set(title='original mask outline')

        # composite option
        elif option in ['both', 'default']:
            self.plot('orig_edges', ax=ax)
            self.plot('merged_edges', ax=ax)
            ax.set(title='pre and post merging edges')

        # cluster plots
        elif option == 'cluster':

            # validate the given cluster
            assert 'cluster' in kwargs.keys(), 'must specify which cluster'
            assert kwargs['cluster'] in self.seg_clusters.values(), \
                'cluster must exist'

            # make an translucent mask
            rgba = np.zeros([self._ydim, self._xdim, 4])
            rgba[:, :, 3] = .75

            # for every segment in this cluster make the mask transparent
            for seg in self.seg_dict.values():
                if self.seg_clusters[seg.id] == kwargs['cluster']:
                    rgba = seg.fill_mask(rgba, alpha=.0)

            ax.imshow(rgba)
            ax.set(title='Cluster ' + str(kwargs['cluster']))

        elif option == 'cluster_all':

            # validate the background image is present
            assert 'back_img' in kwargs.keys(), 'must give the background image'

            # setup figure
            plt.close()
            clusts = set(list(self.seg_clusters.values()))
            n = int((1 + len(clusts)) / 2)
            fig, axs = plt.subplots(n, 2, figsize=[20, n*10])

            # plot each cluster
            for _ax, clust in zip(axs.ravel()[:len(clusts)], clusts):
                self.plot('cluster', cluster=clust,
                          back_img=kwargs['back_img'], ax=_ax)

            # delete unused plots
            for _ax in axs.ravel()[len(clusts):]:
                fig.delaxes(_ax)
            plt.draw()

        # single segment plot
        elif option == 'segment':

            # validate the given segment
            assert 'segment' in kwargs.keys(), 'must specify which segment'
            assert kwargs['segment'] in self.seg_dict.keys(), \
                'segment must exist'

            # color in the segment in slighly translucent red
            rgba = np.zeros([self._ydim, self._xdim, 4])
            seg = self.seg_dict[kwargs['segment']]
            rgba = seg.fill_mask(rgba, alpha=.8, color=[1., 0., 0.])

            ax.imshow(rgba)
            ax.set(title='Segment ' + str(kwargs['segment']))

        elif option == 'segment_edge':

            # validate the given segment
            assert 'segment' in kwargs.keys(), 'must specify which segment'
            assert kwargs['segment'] in self.seg_dict.keys(), \
                'segment must exist'

            # plot just this segments edges
            self.seg_dict[kwargs['segment']].edge_plot(ax)


        # confidence in the existsence of each edge
        elif option == 'edge_conf':

            # use a nan array to be transparaent on imshow
            conf = np.full([self._ydim, self._xdim], np.nan)

            # fill every segement edge pixel with its confidence value
            for seg in self.seg_dict.values():
                seg.edge_mask(conf)

            # plot with color bar
            col = ax.imshow(conf, cmap='bwr')
            plt.colorbar(col)
            ax.set(title='Edge Confidences')


        # error option not recognised
        else:
            print('option not recognised, allowed options are:')
            for s in ('orig_edges', 'merged_edges', 'both', 'cluster',
                      'cluster_all', 'segment', 'segment_edge', 'edge_conf'):
                print('\t-' + s)


# -----------------------------------------------------------------------------


class segment():
    """
    A single segment which holds its constituent pixels, edge pixels and
    figures out its neighbours. This is to be used by the segment_group obj.


    segment(cords, edges, index, group_obj)


    Parameters
    ----------

    cords : 2d numpy array
        The cordinates of this segments pixels in shape (N, 2) where N is the
        number of pixels and the second axis is the X, Y cordinates

    edges : 2d numpy array
        The cordinates of this segments edge pixels in the same shape as cords

    index : int
        The id of this segment used in the segment_group.seg_dict

    group_obj : segment_group obj
        The segment_group which holds all segment objects. This is used to
        access the locations of other segments in order to find neighbours

    """

    def __init__(self, cords, edges, index, group_obj):

        # store the given params
        self.id = index
        self.cords = cords
        self.edges = edges
        self._group_obj = group_obj

        # find which edges are with each segments
        self.edge_dict = self._identify_edges()
        self.neighbours = list(self.edge_dict.keys())

        # create the dictionary for edge confidence to be used later
        self.conf_dict = {n:0 for n in self.neighbours}


    def _identify_edges(self):
        """
        For every edge pixel use the segment_group's mask to find what other
        segments neighbour it. Then store the neighbours in a dictionary
        {neighbour_id : [edge_cordinates]} so that all pixels on a particular
        border can be indexed.
        """

        # dictionary for {neighbour_id:[edge_cordinates]}
        edge_dict = {}

        # for every edge pixel
        for x, y in self.edges:

            # find its neighbour pixels and their segments
            adj_cords = self._pixel_neighbours(x, y)
            adj_segs = self._group_obj.mask[adj_cords[:, 1], adj_cords[:, 0]]
            # indexs backwards as mask has y, x format

            # remove duplicates
            unique_segs = np.unique(adj_segs)

            # store this cordinate in edge_dict of every other segment found
            for i in unique_segs:
                if i != self.id:

                    # if neighbour segment already in edge_dict append to it
                    # else create it as a new entry
                    edge_dict.setdefault(int(i), []).append((x, y))

        # set every edge entry to an array rather than a list of lists
        for key, lst in edge_dict.items():
            edge_dict[key] = np.array(lst)

        return edge_dict


    def _pixel_neighbours(self, x, y):
        """
        Find the neighbours to the x and y pixel, ensuring that these pixels
        are not outside the image by using the dimensions on self._group_obj.
        """
        return np.array([(x_, y_)
                         for x_ in range(x-1, x+2) # for each adj x
                         for y_ in range(y-1, y+2) # for each adj y
                         if ((x != x_ or y != y_) and   # not itself
                             (0 <= x_ < self._group_obj._xdim) and # not out x
                             (0 <= y_ < self._group_obj._ydim))])  # not out y


    def fill_mask(self, rgba, color=None, alpha=0.4):
        """
        Fill the given 3d rgba array with this segents cordinates using the given
        color values and given alpha (opaqueness).
        """

        # validate input
        assert len(rgba.shape) == 3, 'arr must be 3d'
        assert rgba.shape[-1] == 4, 'arr must have shape (X, Y, 4)'

        # if no color given pick a random one
        if color is None:
            color = np.random.rand(3)
        elif len(color) != 3:
            raise ValueError # must give 3 color values

        # set each channel appropirately
        rgba[self.cords[:, 1], self.cords[:, 0], 0] = color[0]
        rgba[self.cords[:, 1], self.cords[:, 0], 1] = color[1]
        rgba[self.cords[:, 1], self.cords[:, 0], 2] = color[2]
        rgba[self.cords[:, 1], self.cords[:, 0], 3] = alpha

        return rgba


    def edge_mask(self, arr):
        """
        Fill the given 2d arr at this segments edge values with this segments
        edge confidence at that boarder.
        """

        # validate input
        assert len(arr.shape) == 2, 'arr must be 2d'

        # for each neighbour and its border pixels fill arr with edge_conf
        for n_id, n_cords in self.edge_dict.items():
            arr[n_cords[:, 1], n_cords[:, 0]] = self.conf_dict[n_id]

        return arr


    def edge_plot(self, ax):
        """
        Plot this pixles edges on the given axis with a different color for
        each edge and label with a legend.
        """

        # for each neighbour and its border pixels plot with a label
        for n_id, n_cords in self.edge_dict.items():
            ax.plot(n_cords[:, 0], n_cords[:, 1], 'x', ms=5, label=n_id)

        # add the legend
        ax.legend()


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Create a regular dummy mask
    example_mask = np.arange(100).reshape([10, 10]).repeat(10, axis=0).repeat(10, axis=1)
    example_mask[12, 12] = 25 # add an iregular center
    example_mask[12, 17] = 25 # add an iregular center

    # create the segment object
    example_obj = segment_group(example_mask)

    # define a clustering for every segment after 35
    example_clust = np.arange(102)
    example_clust[35:] = 36

    # assign this clustering and implement the merging
    example_obj.assign_clusters(example_clust)
    example_obj.merge_by_cluster()

    # plot the outcome (lower side of the image is a single cluster)
    example_obj.plot(back_img=example_mask)
