import torch
import matplotlib.pyplot as plt
import numpy as np
from SLIC import SLIC
from tools import set_seed, get_img, progress_bar


class MSLIC_wrapper():

    def __init__(self, imgs, bin_grid, combo_metric='max',
                 dist_metric='default', **kwargs):
        """
        Wrapper around SLIC to run multiple instances at once, using a the
        combined of each instance to assign clusters in every iteration
        
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
            
        combo_metric : string, optional
            The metric by which the distances between vectors and centroids
            in each image are to be combined. Allowed options are:
            - 'max' maximum of the distances in each image
            - 'min' minimum of the distances in each image
            - 'mean' average of the distances in each image
            - 'sum' total of the distances in each image
            - 'custom' custom function must be passed as 'metric_func' kwarg
                which takes a 3d tensor of distances with shape
                (image, vector, cluster) and returns a 2d tensor of distances
                with shape (vector, cluster)

        dist_metric : string, optional (same as SLIC)
            Choose the of calculating distance between two vectors:
            - 'normal' finds the normal distance without scaling either
                position or color space
            - 'default' finds the normal distance with scaling the position
                space by the bin widths. Can also pass the kwarg 'factor'
                with an additional scaling factor for the position space.
            - 'custom' pass a fucntion to be used in the kwarg 'dist_func'
            
        kwargs : optional
            Other than those described above for combo_metric and dist_metric
            these are passed into the SLIC objects, refer to it's documentaiton.
        """
        
        # validate all given images
        for img in imgs:
            assert img.shape == imgs[0].shape, "images must be the same shape"
        
        # create the SLIC objects
        self.SLIC_objs = [SLIC(img, bin_grid, dist_metric, **kwargs)
                          for img in imgs]
        
        # set the combine metric
        if combo_metric == 'max':
            self.metric = lambda t:t.max(dim=0)[0]

        elif combo_metric == 'min':
            self.metric = lambda t:t.min(dim=0)[0]

        elif combo_metric == 'mean':
            self.metric = lambda t:t.mean(dim=0)

        elif combo_metric == 'sum':
            self.metric = lambda t:t.sum(dim=0)
            
        elif combo_metric == 'custom':
            assert 'metric_func' in kwargs.keys(), \
                "Must pass 'metric_func' as key word argument"
            self.metric = metric_func
        
        else:
            assert False, "metric " + combo_metric + " not recognised"
                
    
    def combined_distance(self):
        "Find the combined distance for every pixel using the specified metric"
        
        # reset the distances
        self.combo_dist = []
        
        # for all distance values in every bin
        all_distances = [obj.vc_dists for obj in self.SLIC_objs]
        for dists in zip(*all_distances):
            
            # calculate the combined distance using the distance metric
            dist_tensor = torch.stack(dists)
            self.combo_dist.append(self.metric(dist_tensor))
    
            
    def iterate(self, n_iter):
        "loop for n_iter iterations with progress bar"
        
        # create the progress bar and store on each of the objects
        self.progress_bar = progress_bar(n_iter)
        for obj in self.SLIC_objs:
            obj.progress_bar = self.progress_bar
        
        for i in range(n_iter):
            
            # iterate up to finding distances for each kmeans object
            for obj in self.SLIC_objs:
                obj.update_distances()
                
            # find the combined distance
            self.combined_distance()
            
            # finish the iteration for each SLIC
            for n, obj in enumerate(self.SLIC_objs):
                
                # set the combined distance
                obj.vc_dists = self.combo_dist
                
                # only calculate the new clusters once
                if n == 0:
                    obj.update_clusters()
                else:
                    obj.cluster_tensor = self.SLIC_objs[0].cluster_tensor
                    obj.cluster_list = self.SLIC_objs[0].cluster_list
                    
                # update the centroid means
                obj.update_centroids()
            
            # print the progress bar
            self.progress_bar(i)
    
    
    def plot(self, obj_indexs, option='default', axs=[None], path=None):
        """
        Calls obj.plot() on each of the SLIC objects given with option
        
        Parameters
        ----------
        
        obj_indexs : list of ints
            which of the given objects to be plotted
            
        option : string, optional
            the plot option to be passed to obj.plot, refer to the docstrings
            there for available options 
            
        axs : tuple of matplotlib axis, optional
            the axis to be plotted on, if not given a subplot for each of
            the SLIC objs is used
        """
        
        # set the axis if not given
        if not any(axs):
            N = len(obj_indexs)
            fig, axs = plt.subplots(N, 1, figsize=[N*22, 22])
            axs = np.array([axs]).ravel() # force it be an array even if N = 1
            
        # call the plot routenes
        for i, ax in zip(obj_indexs, axs):
            self.SLIC_objs[i].plot(option, ax=ax, path=path)
            
            
if __name__ == '__main__':
    set_seed(10)

    grid = [40, 40]

    img_white = get_img("images/TX1_white_cropped.tif")
    obj_white = kmeans_local(img_white, grid)

    img_polar = get_img("images/TX1_polarised_cropped.tif")
    obj_polar = kmeans_local(img_polar, grid)

    # plot the initial binning 
    obj_white.plot("setup")
    plt.gca().set(title='Initial Grid')

    multi_obj = MSLIC_wrapper([obj_white, obj_polar])

    # iterate
    multi_obj.iterate(2)
    
    # plot
    multi_obj.plot([0,1])
    