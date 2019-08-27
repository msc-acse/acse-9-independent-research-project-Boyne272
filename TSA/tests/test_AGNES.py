import unittest
import numpy as np
import matplotlib.pyplot as plt
from TSA.merging import AGNES


class Test_AGNES(unittest.TestCase):
    
    def test_2d_dummy(self):
        "Generate 2d dummy data and test the whole functionality"
        
        # generate dummy data with 6 regions
        np.random.seed(10)
        features = np.random.normal(size=[500, 2])
        features[50:, :] += 4
        features[100:, :] += 4
        features[150:, 0] += 4
        features[250:300, 1] += 10
        features[350:400, 0] += 10

        # create object
        obj = AGNES(features)

        # check obj initalised correctly
        assert (obj._dists > 0).all(), "can not be negaive distances"
        exp_size = 0.5 * (obj._dists.size - obj._dists.shape[0])
        assert np.isfinite(obj._dists).sum() == exp_size, "distances should be lower traingular"

        # iterate
        obj.iterate()

        # check obj post iteration is correct
        assert len(obj.merge_log) == len(set(obj.merge_log)), \
            'no pair should be joined more than once'
        assert np.isinf(obj._dists).all(), \
            'every pair should be infinitely far apart once finished'

        # check the plot functions run
        obj.cluster_distance_plot('all')

        # check the grouping is as expected
        n_clusters = obj.cluster_by_derivative(n_std=3.).max() + 1
        assert n_clusters == 6, 'derivative clustering not as expected'

        # check grouping by distance works too
        # (using a distance just under what is stated aboive)
        n_clusters = obj.cluster_by_distance(cutoff_dist=9.41).max() + 1
        assert n_clusters == 6, 'cluster_by_distance clustering not as expected'
        
        # check grouping by distance works too
        # (using a distance just under what is stated aboive)
        n_clusters = obj.cluster_by_index(500-6).max() + 1
        assert n_clusters == 6, 'index clustering not as expected'
        
        # prevent figure build up
        plt.close('all')
        
        
    def test_2d_dummy_plus_outlier(self):
        """
        Generate 2d dummy data and test the whole functionality with
        an additional outlier
        """
        
        # generate dummy data with 6 regions
        np.random.seed(10)
        features = np.random.normal(size=[500, 2])
        features[50:, :] += 4
        features[100:, :] += 4
        features[150:, 0] += 4
        features[250:300, 1] += 10
        features[350:400, 0] += 10
        
        features[-1, -1] += 20

        # create object
        obj = AGNES(features)

        # check obj initalised correctly
        assert (obj._dists > 0).all(), "can not be negaive distances"
        exp_size = 0.5 * (obj._dists.size - obj._dists.shape[0])
        assert np.isfinite(obj._dists).sum() == exp_size, "distances should be lower traingular"

        # iterate
        obj.iterate()

        # check obj post iteration is correct
        assert len(obj.merge_log) == len(set(obj.merge_log)), \
            'no pair should be joined more than once'
        assert np.isinf(obj._dists).all(), \
            'every pair should be infinitely far apart once finished'

        # check the plot functions run
        obj.cluster_distance_plot('all')

        # check the grouping is as expected
        n_clusters = obj.cluster_by_derivative().max() + 1
        assert n_clusters == 7, 'clustering not as expected'

        # check grouping by distance works too
        # (using a distance just under what is stated aboive)
        n_clusters = obj.cluster_by_distance(cutoff_dist=9.41).max() + 1
        assert n_clusters == 7, 'cluster_by_distance clustering not as expected'
        
        # check grouping by distance works too
        # (using a distance just under what is stated aboive)
        n_clusters = obj.cluster_by_index(500-7).max() + 1
        assert n_clusters == 7, 'index clustering not as expected'
        
        
        # prevent figure build up
        plt.close('all')
        

    def test_higher_dimensions(self):
        "Exact same as test_2d_dummy_plus_outlier but on higer dimensional data"
        
        # generate dummy data with 6 groupings
        np.random.seed(10)
        features = np.random.normal(size=[500, 5])
        features[50:, :] += 4
        features[100:, :] += 4
        features[150:, 0] += 4
        features[250:300, 1] += 10
        features[350:400, 0] += 10
        features[-1, -1] += 20

        # create object
        obj = AGNES(features)

        # check obj initalised correctly
        assert (obj._dists > 0).all(), "can not be negaive distances"
        exp_size = 0.5 * (obj._dists.size - obj._dists.shape[0])
        actual_size = np.isfinite(obj._dists).sum()
        assert actual_size == exp_size, "distances should be lower traingular"

        # iterate
        obj.iterate()

        # check obj post iteration is correct
        assert len(obj.merge_log) == len(set(obj.merge_log)), \
            'no pair should be joined more than once'
        assert np.isinf(obj._dists).all(), \
            'every pair should be infinitely far apart'

        # check the plot functions run
        obj.cluster_distance_plot('all')

        # check the grouping is as expected
        n_clusters = obj.cluster_by_derivative().max() + 1
        assert n_clusters == 7, 'clustering not as expected'
        
        # check grouping by distance works too
        # (using a distance just under what is stated aboive)
        n_clusters = obj.cluster_by_distance(cutoff_dist=10.4).max() + 1
        assert n_clusters == 7, 'cluster_by_distance clustering not as expected'
        
        # check grouping by distance works too
        # (using a distance just under what is stated aboive)
        n_clusters = obj.cluster_by_index(500-7).max() + 1
        assert n_clusters == 7, 'index clustering not as expected'

        # prevent figure build up
        plt.close('all')
        
        
if __name__ == '__main__':
    
    # run all the tests if this is script is run independently
    tmp = Test_AGNES()
    tmp.test_2d_dummy()
    tmp.test_2d_dummy_plus_outlier()
    tmp.test_higher_dimensions()
    print('all tests passed')

#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()