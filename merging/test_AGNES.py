import unittest
import numpy as np
import matplotlib.pyplot as plt
# from AGNES import AGNES
from tools import progress_bar

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
        n_clusters = obj.cluster_by_derivative().max() + 1
        assert n_clusters == 6, 'clustering not as expected'

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
        assert n_clusters == 6, 'clustering not as expected'

        # prevent figure build up
        plt.close('all')
        

    def test_higher_dimensions(self):
        "Exact same as above but on higer dimensional data"
        
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

        # prevent figure build up
        plt.close('all')
        
        
if __name__ == '__main__':
#     unittest.main()
    tmp = Test_AGNES()
    tmp.test_2d_dummy()
    tmp.test_2d_dummy_plus_outlier()
    tmp.test_higher_dimensions()