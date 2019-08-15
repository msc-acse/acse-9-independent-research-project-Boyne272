import unittest
import numpy as np
import matplotlib.pyplot as plt

from Segments import segment_group
from Segments import Mask_utilities


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
        outline4 = obj._outline(diag=False)
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

########################### make me ###########################################
# class Test_Segments(unittest.TestCase):
#     """
#     This test class is for both segment_group and segments as neither can
#     operate without the other
#     """
    
#     def test_initalisation(self):
        
        
        
        
# if __name__ == '__main__':
    
#     # run all the tests if this is script is run independently
#     test_obj = Test_Segments()
#     test_obj.test_initalisation()
#     print('all tests passed')
    
# #     # unittest.main does not work in google colab, but should work elsewhere
# #     unittest.main()