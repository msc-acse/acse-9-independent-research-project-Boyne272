# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Tue Aug 27 18:23:14 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from ..tools import progress_bar


class AGNES():
    """
    Clusters the given feature vectors by the AGNES algorithm


    AGNES(features, join_by='max')


    Parameters
    ----------
    features : 2d numpy array
        The features to cluster by with dimensions (n_samples, n_features).
        It is advised to input features with normalised distributions
        as they are not scaled here and this is a distance based clustering
        algoithm

    join_by : string (optional)
        The method by which to recalculate distance to a new group after
        merging occures. Allowed options are:
            - 'max' takes the new clusters distance as the maximum of
              either of the joined vectors distances
            - 'min' same as above but with minimum distance
    """


    def __init__(self, features, join_by='max'):

        # logs which track merges and there distances
        self.dist_log = []
        self.merge_log = []

        # vectors
        self._vecs = features.copy()
        self._N = features.shape[0]
        self._group_count = self._N

        # find distances between every vector
        self._dists = np.full((self._N, self._N), np.inf)
        for _id in range(self._N):
            self._find_distance(_id)

        # option of distance recaulculation
        if join_by == 'max':
            self._join = self._join_by_max
        elif join_by == 'min':
            self._join = self._join_by_min
        else:
            raise ValueError # join_by not recoginsed


    def _find_distance(self, _id):
        "Calculte the distance between this vector and all other vectors"

        # find distance to every point before this id
        self._dists[_id, :_id] = la.norm(self._vecs[_id] - self._vecs[:_id],
                                         axis=1)

        # find the distance to every pstdoint after this id
        self._dists[_id+1:, _id] = la.norm(self._vecs[_id] - self._vecs[_id+1:],
                                           axis=1)


    def _join_by_min(self, id1, id2):
        """
        Find the new distance from every vector to the group formed by
        joining id1 and id2. This is taken as the min distance to either of
        these vectors. id1 stores the new distance and id2 is set to inf
        so as to not be considered in future min distance calculations.
        """

        # find the minimum of the two distances
        dists_id1 = np.hstack([self._dists[id1, :id1], self._dists[id1:, id1]])
        dists_id2 = np.hstack([self._dists[id2, :id2], self._dists[id2:, id2]])
        dists_min = np.minimum(dists_id1, dists_id2)

        # make merging points inf distance apart
        dists_min[id1] = dists_min[id2] = np.inf

        # set the id1 distances to the min_distances
        self._dists[id1, :id1] = dists_min[:id1]
        self._dists[id1:, id1] = dists_min[id1:]

        # set the id2 distances to inf
        self._dists[id2, :id2] = self._dists[id2:, id2] = np.inf


    def _join_by_max(self, id1, id2):
        """
        Find the new distance from every vector to the group formed by
        joining id1 and id2. This is taken as the max distance to either of
        these vectors. id1 stores the new distance and id2 is set to inf
        so as to not be considered in future min distance calculations.
        """

        # find the minimum of the two distances
        dists_id1 = np.hstack([self._dists[id1, :id1], self._dists[id1:, id1]])
        dists_id2 = np.hstack([self._dists[id2, :id2], self._dists[id2:, id2]])
        dists_max = np.maximum(dists_id1, dists_id2)

        # set the id1 distances to the min_distances
        self._dists[id1, :id1] = dists_max[:id1]
        self._dists[id1:, id1] = dists_max[id1:]

        # set the id2 distances to inf
        self._dists[id2, :id2] = self._dists[id2:, id2] = np.inf


    def iterate(self):
        """
        Merge closest pairs until all datapoints are connected. This has to be
        done before any clustering may occure.
        """

        iterate = self._group_count - 1 # should be n - 1 mergers
        prog_bar = progress_bar(iterate) # verbose

        for i in range(iterate):
            assert self._group_count > 1, 'graph already fully merged'

            # find the two closed groups to merge
            id1, id2 = np.unravel_index(self._dists.argmin(),
                                        self._dists.shape)

            # log this merger
            self.dist_log.append(self._dists[id1, id2])
            self.merge_log.append((id1, id2))
            self._group_count -= 1

            self._join(id1, id2) # implement merger

            prog_bar(i)  # verbose
        print('\n') # verbose


    def cluster_by_derivative(self, n_std=3., plot=True):
        """
        Find the grouping by allowing all mergers up to a cutoff, here
        here determined by a vairation in the second derivative over the given
        numbers of standard deviation.

        plot bool will plot the grouping projection in 2d.
        """

        # find the second deriative cutoff
        y = np.array(self.dist_log)
        dy2 = y[:-2] - 2*y[1:-1] + y[2:] # central
        cutoff = np.std(dy2) * n_std

        index = np.argmax(dy2 > cutoff) + 1 + 1
        # argmax gives the first instance greater than cutoff
        # +1 because central differencing scheme used, hence the point where
        # the distance jumps is actually one point further on
        # +1 again because dy2 not calculated for first merger

        # verbose
        print('Clustering up to 2nd derivative', cutoff,
              ' distance ', y[index])

        groupings = self.cluster_by_index(index, plot)

        return groupings


    def cluster_by_index(self, index, plot=True):
        """
        Find the grouping by allowing all mergers up to a given index,
        There will be N-index clusters where N is the number of samples
        (-1 from there being N-1 maximum joins is cancled by +1 from indexs
        starting at 0).
        Plot will plot the grouping projection in 2d.
        """

        # this directory tracks what group a vector belongs to
        directory = {n:n for n in range(self._N)}
        # { vector id : group id }

        n_clusters = self._N - index
        print('Clustering into', n_clusters, 'segments')

        # for every merge up to the requiered index
        for i in range(index):
            new_group, old_group = self.merge_log[i]

            # update every vector that had this group
            for vec_id, group in directory.items():
                if group == old_group:
                    directory[vec_id] = new_group

        # rebase the group indexs to start from 1 and be consecutive
        groupings = np.unique(list(directory.values()), return_inverse=True)[1]

        # plot with color bar if wanted
        if plot:
            _fig, ax = plt.subplots(figsize=[12, 8])
            col = ax.scatter(*self._vecs[:, :2].T, c=groupings)
            ax.set(title=str(groupings.max() + 1) +
                   ' Clusters up to Index ' + str(index))
            plt.colorbar(col)

        return groupings


    def cluster_by_distance(self, cutoff_dist=1., plot=True):
        """
        Find the grouping by allowing all mergers up to a given join distance.
        Plot will plot the grouping projection in 2d.
        """

        print('Clustering up to distance', cutoff_dist)
        index = np.array([d < cutoff_dist for d in self.dist_log]).sum()
        groupings = self.cluster_by_index(index, plot)

        return groupings


    def cluster_distance_plot(self, option='all', ax=None, last_few=True):
        """
        Plot the option vs iterations on the given axis

        Parameters
        ---------
        option : string (optional)
            What to plot aginst iteration, choose from:
            - 'dists' plots the join distance
            - '1st' plots the numerical first derivative
            - '2nd'plots the numerical second derivative
            - 'all' (default) plots plots all of the above
                (ignores ax and last few)

        ax : Matplotlib axis (optional)
            The axis to plot on, one is created if not given. If 'all' chosen
            then this setting is ignored

        last_few : bool (optional)
            Plot only the last 25 points (or all if less than 25 in total).
            If 'all' chosen then this setting is ignored
        """

        # set axis
        if not ax:
            _fig, ax = plt.subplots(figsize=[12, 10])

        # working values
        n = 25 if last_few else self._N
        y = np.array(self.dist_log)

        # base options
        if option == 'dists':
            x = np.arange(0, self._N-1)
            ax.plot(x[-n:], y[-n:], '-o')
            ax.set(title='Join Distances', xlabel='Iteration',
                   ylabel='Distance')

        elif option == '1st':
            x = np.arange(1, self._N-1)
            dy = y[1:] - y[:-1] # forward
            ax.plot(x[-n:], dy[-n:], '-o')
            ax.set(title='First Diff', xlabel='Iteration',
                   ylabel='Distance 1st Derivative')

        elif option == '2nd':
            x = np.arange(1, self._N-2)
            dy2 = y[:-2] - 2*y[1:-1] + y[2:] # central
            ax.plot(x[-n:], dy2[-n:], '-o')
            ax.set(title='Second Diff', xlabel='Iteration',
                   ylabel='Distance 2nd Derivative')

        # composit option
        elif option == 'all':
            plt.close()
            _fig, axs = plt.subplots(2, 2, figsize=[15, 15])
            self.cluster_distance_plot('dists', ax=axs[0, 0])
            self.cluster_distance_plot('dists', ax=axs[0, 1], last_few=False)
            self.cluster_distance_plot('1st', ax=axs[1, 0])
            self.cluster_distance_plot('2nd', ax=axs[1, 1])

        # option not recognised
        else:
            plt.close()
            raise ValueError


if __name__ == '__main__':

    # generate dummy 2d data
    np.random.seed(10)
    rand_feats = np.random.normal(size=[500, 2])
    rand_feats[50:, :] += 5
    rand_feats[100:, :] += 5
    rand_feats[150:, 0] += 5
    rand_feats[250:300, 1] += 10
    rand_feats[350:400, 0] += 10
    rand_feats[-1, -1] += 20

    # plot the dummy data
    plt.figure(figsize=[5, 5])
    plt.scatter(*rand_feats.T)
    plt.title("dummy data")

    # cluster it and plot the clusters
    obj = AGNES(rand_feats)
    obj.iterate()
    obj.cluster_distance_plot('all')
    obj.cluster_by_derivative(n_std=3., plot=True)
