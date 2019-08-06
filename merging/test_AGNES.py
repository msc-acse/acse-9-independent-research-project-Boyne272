import unittest
import numpy as np
import matplotlib.pyplot as plt
from AGNES import AGNES
from tools import progress_bar

class Test_AGNES(unittest.TestCase):
    
    def test_2d_dummy(self):
        "Generate 2d dummy data and test the whole functionality"
        
        # generate dummy data
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
        assert np.isinf(obj._dists).all(), 'every pair should be infinitely far apart'

        # check the plot functions run
        obj.cluster_distance_plot('all')

        # check the grouping is as expected
        assert len(np.unique(obj.deriv_clustering(5))) == 6, 'clustering not as expected'

        # prevent figure build up
        plt.close('all')
        

    def test_higher_dimensions(self):
        "Exact same as above but on higer dimensional data"
        
        # generate dummy data
        np.random.seed(10)
        features = np.random.normal(size=[500, 5])
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
        assert np.isinf(obj._dists).all(), 'every pair should be infinitely far apart'

        # check the plot functions run
        obj.cluster_distance_plot('all')

        # check the grouping is as expected
        assert len(np.unique(obj.deriv_clustering())) == 7, 'clustering not as expected'

        # prevent figure build up
        plt.close('all')
        
        
if __name__ == '__main__':
#     unittest.main()
    tmp = Test_AGNES()
    tmp.test_2d_dummy()
    tmp.test_higher_dimensions()