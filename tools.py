# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:20:45 2019

@author: Richard Bonye
"""

import sys

def percent_print(i, i_max, interval=1):
    """
    Print a progress bar or percentage value
    Parameters
    ----------
    i : int
        The current iteration number
    i_max : int
        The total number of iterations to complete
    interval : int, optional
        The frequency of updating the progress bar (set high for long
        simulations)
    Returns
    -------
    updated : bool
        Whether the update bar was updated
    """

    if i % interval == 0:

        # print percent
        # sys.stdout.write("\r %.1f %% Done" % (100*i/i_max))

        # print progress bar
        m = int(50 * i/i_max) + 1
        n = 50 - m
        sys.stdout.write("\rProgress |" + "#"*m + " "*n + "|")

        # update the string on screen
        sys.stdout.flush()
        return True

    return False
	
	
from numpy import array
from PIL import Image

def get_img(path):
    "Load an image as a numpy array"
    img = Image.open(path)
    arr = array(img.convert('RGB')).astype(float)/255.
    return arr