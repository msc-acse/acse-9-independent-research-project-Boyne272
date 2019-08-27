import unittest
import numpy as np
import matplotlib.pyplot as plt

from Segments import segment_group
from Segments import Mask_utilities

# from TSA.merging.Segments import segment_group
# from TSA.merging.Segments import Mask_utilities


# -----------------------------------------------------------------------------


class Test_Mask_utilities(unittest.TestCase):
    "Some simple tests for the Mask utilites section"
    
    def test_initalisation(self):
        "Unit Test the init function breaks when it should"
        
        # check it breaks for non-int arrays
        try:
            mask = np.arange(25).reshape([5, 5]).astype('float')
            obj = Mask_utilities(mask)
            return 'failed'
        except:
            pass
        
        # check it breaks for non-2d arrays
        try:
            mask = np.arange(25).astype('int')
            obj = Mask_utilities(mask)
            return 'failed'
        except:
            pass
        
        
    def test_rgba(self):
        "Unit Test rgba to ensure it returns masks that make sense"
        
        # create a simple mask for testing
        mask = np.zeros([10, 10]).astype('int')
        mask[:5, :5] = mask[5:, 5:] = 1
        obj = Mask_utilities(mask)
        
        # check red color channel maskes sense
        red_rgba = obj._rgba(mask, color='r')
        assert red_rgba.ndim == 3, 'must be 3d'
        assert red_rgba.shape[-1] == 4, 'must have 3 color channels and alpha'
        assert red_rgba.shape[:2] == mask.shape, \
            'must have same x,y dimensions as mask'
        assert (red_rgba[:, :, 0]).any(), 'red must have non-zero values'
        assert (red_rgba[:, :, 3]).any(), 'alpha must have non-zero values'
        assert not (red_rgba[:, :, 1:3]).any(), \
            'green and blue must have all zero values'
        
        # check it works with the default mask, color and alpha
        green_rgba = obj._rgba(color='g', opaqueness=.5)
        assert green_rgba.ndim == 3, 'must be 3d'
        assert green_rgba.shape[-1] == 4, 'must have 3 color channels and alpha'
        assert green_rgba.shape[:2] == mask.shape, \
            'must have same x,y dimensions as mask'
        assert (green_rgba[:, :, 1]).any(), 'green must have non-zero values'
        assert (green_rgba[:, :, 3]).any(), 'alpha must have non-zero values'
        assert not ((green_rgba[:, :, 0]).any() and
                    (green_rgba[:, :, 2]).any()), \
            'red and blue must have all zero values'
        assert green_rgba[-1, -1, 3] == .5, 'alpha must be .5'
        
        
    def test_outline(self):
        "Unit Test outline to ensure it returns masks that make sense"
        
        # create a simple mask for testing
        mask = np.zeros([10, 10]).astype('int')
        mask[:5, :5] = mask[5:, 5:] = 1
        obj = Mask_utilities(mask)
        
        # check the outline is as expected
        outline1 = obj._outline()
        assert outline1.sum() == 36., 'should be 36 detected edges'
        
        # check changing the mask, outline still works
        obj.mask = np.zeros([10, 10]).astype('int')
        obj.mask[5, 5] = 1
        outline2 = obj._outline()
        outline3 = obj._outline(original=True)
        assert not (outline2 == outline1).all(), \
            'Should now be a different array'
        assert (outline3 == outline1).all(), \
            'Should be the same array'
        
        # check the diag option works as intended
        outline4 = obj._outline(diag=False, multi=False)
        assert outline4.sum() == 5, 'expected a cross of size 5'
        assert outline2.sum() == 9, 'expected a box of size 9'
        
        
        
if __name__ == '__main__':
    
    # run all the tests if this is script is run independently
    test_obj = Test_Mask_utilities()
    test_obj.test_initalisation()
    test_obj.test_rgba()
    test_obj.test_outline()
    print('all tests passed')
    
#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()


# -----------------------------------------------------------------------------

class Test_Segments(unittest.TestCase):
    """
    This test class is for both segment_group and segments as neither can
    operate without the other
    """
    
    def test_segments_consistency(self):
        """
        ensure the segment objs are consistent with eachother by agreeing
        with neighbours and never having impossible configureations like
        more edges than cordinates.
        """

        # Create a regular dummy mask
        mask = np.arange(100).reshape([10,10]).repeat(10, axis=0).repeat(10, axis=1)
        mask[12,12] = 25 # add an iregular center
        mask[12,17] = 25 # add an iregular center

        # create segment obj
        obj = segment_group(mask)

        # for every segment
        for _id, seg in obj.seg_dict.items():

            # check neighbours agree on being a neighbour
            for n_id in seg.neighbours:
                neigh = obj.seg_dict[n_id]
                assert _id in neigh.neighbours, \
                    'neighbour must contain self as neighbour'

            # check there are not more edges than cordinates
            assert len(seg.edges) <= len(seg.cords), \
                'must be less edges than cordinates'
            
    
    def test_neighbours(self):
        """
        unit test the neighbour and edge detection by generating a dummy mask
        with expected bahviour and ensuring several sample instances with
        different expected numbers of neighbours are as expected
        """

        # Create a regular dummy mask
        mask = np.arange(25).reshape([5,5]).repeat(5, axis=0).repeat(5, axis=1)
        mask[12,12] = 25 # add an iregular center
        mask[12,17] = 25 # add an iregular center

        # make the object with this mask
        obj = segment_group(mask)
        
        print(obj.seg_dict, np.unique(obj.mask))
        
        # verify group initalisation with contiunity enforced
        assert len(obj.seg_dict) == np.unique(obj.mask).size, \
            'should be the same segemnts as in the mask'
        assert len(obj.seg_dict) == 27, "there should be 27 clusters"

        # verify the neighbours of a few specific clusters
        clust = obj.seg_dict[0]
        assert len(clust.neighbours) == 3, 'This segment should have x neghbours'
        clust = obj.seg_dict[1]
        assert len(clust.neighbours) == 5, 'This segment should have x neghbours'
        clust = obj.seg_dict[7]
        assert len(clust.neighbours) == 8, 'This segment should have x neghbours'
        clust = obj.seg_dict[12]
        assert len(clust.neighbours) == 9, 'This segment should have x neghbours'
        clust = obj.seg_dict[15] # iregular (enforcing continuiny relabels clusters)
        assert len(clust.neighbours) == 1, 'This segment should have x neghbours'


        # verify the number of edge pixels for a few specific clusters
        clust = obj.seg_dict[0]
        assert len(clust.edges) == 9, \
            'This segment has 2 edges of len 5 with 1 shared pixels'
        clust = obj.seg_dict[1]
        assert len(clust.edges) == 13, \
            'This segment has 3 edges of len 5 with 3 shared pixels'
        clust = obj.seg_dict[7]
        assert len(clust.edges) == 16, \
            'This segment has 4 edges of len 5 with 4 shared pixels'
        clust = obj.seg_dict[12]
        assert len(clust.edges) == 24, \
            'This segment same as previous but wth center dot (+8)'
        clust = obj.seg_dict[15] # iregular (enforcing continuiny relabels clusters)
        assert len(clust.edges) == 1, \
            'This segment has only 1 pixel'


        # verify the edge pixels with any particular neighbour
        clust = obj.seg_dict[0]
        assert len(clust.edge_dict[1]) == 5, 'These segments share 5 edge pixels'
        assert len(clust.edge_dict[5]) == 5, 'These segments share 5 edge pixels'
        clust = obj.seg_dict[1]
        assert len(clust.edge_dict[0]) == 5, 'These segments share 5 edge pixels'
        assert len(clust.edge_dict[2]) == 5, 'These segments share 5 edge pixels'
        assert len(clust.edge_dict[6]) == 5, 'These segments share 5 edge pixels'
        clust = obj.seg_dict[12]
        assert len(clust.edge_dict[15]) == 8, 'These segments share 8 edge pixels'
        clust = obj.seg_dict[15] # iregular (enforcing continuiny relabels clusters)
        assert len(clust.edge_dict[12]) == 1, 'This segment has only 1 pixel'

        
    def test_feature_extraction(self):
        """
        test feature extraction by making the feature function the
        average of segments pixels in the mask array. Hence each segment
        should have a single feature equal to its id 
        """

        def extraction(Xs, Ys, mask):
            return mask[Ys, Xs].mean()

        # Create a regular dummy mask
        mask = np.arange(25).reshape([5,5]).repeat(5, axis=0).repeat(5, axis=1)
        mask[12,12] = 25 # add an iregular center

        # make the object with this mask
        obj = segment_group(mask)

        # test the feature extraction
        feats = obj.feature_extraction(extraction, [obj.mask])
        for f, seg in zip(feats, obj.seg_dict.values()):
            assert f == seg.id, 'the feature should just be the seg id'


    def test_edge_confidence(self):
        """
        test the edge confidence calculation and assignment by making the
        edge cofindence the average of edge pixels in a mask where all edge
        pixels are 1. Hence every edge confidence value should be 1, else
        a non edge pixel has been used in the caluclation.
        """
        
        def extraction(Xs, Ys, mask):
            return mask[Ys, Xs].mean()

        # Create a regular dummy mask
        mask = np.arange(25).reshape([5,5]).repeat(5, axis=0).repeat(5, axis=1)
        mask[12,12] = 25 # add an iregular center

        # make the object with this mask
        obj = segment_group(mask)

        # get the mask with only edges
        edge_mask = obj._outline()
#         plt.imshow(edge_mask)

        # assign the edge confidences
        obj.edge_confidence(extraction, [edge_mask])

        # test they are all one
        for seg in obj.seg_dict.values():
            for edge_val in seg.conf_dict.values():
                assert (edge_val == 1.), 'every edge pixel averaged should just be 1'


    def test_cluster_merging(self):
        """
        test the cluster merging works by merging the lower half of a regular
        grid and ensuring the segments are then correct agter that
        """

        # Create a regular dummy mask
        mask = np.arange(100).reshape([10,10]).repeat(10, axis=0).repeat(10, axis=1)
        mask[12,12] = 25 # add an iregular center
        mask[12,17] = 25 # add an iregular center

        # create the segment object
        obj = segment_group(mask)

        # define a clustering for every segment after 35
        clust = np.arange(102)
        clust[35:] = 36

        # assign this clustering and implement the merging
        obj.assign_clusters(clust)
        obj.merge_by_cluster()

        # check the expected number of clusters exist
        assert len(obj.seg_dict) == 36, 'there should now be 36 clusters'

        # check the large semgment created has the correct number of neighbours
        big_seg_id = list(obj.seg_dict.keys())[-1]
        assert len(obj.seg_dict[big_seg_id].neighbours) == 11,\
            'the big cluster should have 11 neighbours'
        assert len(obj.seg_dict[big_seg_id].cords) == 6700,\
            'the big cluster should have 6700 cordinate in it'
            
            
if __name__ == '__main__':

    # run all the tests if this is script is run independently
    test_obj = Test_Segments()
    test_obj.test_segments_consistency()
    test_obj.test_neighbours()
    test_obj.test_feature_extraction()
    test_obj.test_edge_confidence()
    print('all tests passed')

#     # unittest.main does not work in google colab, but should work elsewhere
#     unittest.main()