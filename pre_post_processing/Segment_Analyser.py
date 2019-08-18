import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist
from tools import progress_bar

class Segment_Analyser():
    """
    Takes in a segmentation and its clustering and get the user to label
    each cluster. Then the appropirate distriubtions of the labelled clusters
    can be calculated.
    
    
    Segment_Analyser(img, mask, clusters)
    
    
    Parameters
    ----------
    
    img
    
    
    mask
    
    
    clusters
    
    """
    
    def __init__(self, img, mask, clusters):
        
        # store the passed parameters
        self.img = img.copy()
        self.mask = mask.copy()
        self.clusters = clusters.copy()
        
        # dictionary stores clusters labels
        self.labels = dict([(str(c), c) for c in np.unique(clusters)])
#         self.set_labels()
        
        
    def set_labels(self):
        """
        Prompt the user for every cluster to give label for it. If the same
        label is given twice these clusters are merged.
        """
        
        # store the new labels
        label_dict = {}
        
        for label, clust in self.labels.items():
            
            # plot this cluster
            self.plot_custer(label)
            plt.show()
            
            # get user input
            print('Currently labelled ', label ,
                  '\nGive a new label (leave blank to unchange):')
            new_label = input()
            
            # set the new label if given
            if new_label:
                label = new_label
            
            # if the label has been given before merge the clusters
            if label in label_dict.keys():
                self.clusters[self.clusters == clust] = label_dict[label]
            # else store this label for this cluster
            else:
                label_dict[label] = clust
            
            # close that cluster plot for next iteration
            plt.close()
        
        # print the current labels and store them
        print('Current Labels:', list(label_dict.keys()))
        self.labels = label_dict
        
        
    def plot_custer(self, label, ax=None):
        """
        Plot the original image and mask all but the cluster with the
        given label. Plots on ax if given.
        """
        
        # create an axis if not given
        if not ax:
            fig, ax = plt.subplots()
            
        # plot the original image
        ax.imshow(self.img)
        
        # create a plot the cluster mask
        overlay = np.full_like(self.mask, 0.5, dtype='float')
        overlay[self.clusters == self.labels[label]] = np.NaN
        ax.imshow(overlay, cmap='binary')
        
        
    def plot_clusters(self, alpha=.4, ax=None):
        """
        Plot the original image and highlight each different cluster with a
        random colored shade. Plots on ax if given.
        """
              
        # create an axis if not given
        if not ax:
            fig, ax = plt.subplots()
            
        # plot the original image
        ax.imshow(self.img)
        
        # creaete an empty rgba mask
        rgba = np.zeros([*self.mask.shape, 4])
        
        # for every cluster
        for clust in self.labels.values():
            
            # mask of this cluster
            bool_arr = self.clusters == clust
            
            # pick random colors
            color = np.random.rand(3)
            
            # set each channel appropirately
            rgba[:, :, 0][bool_arr] = color[0]
            rgba[:, :, 1][bool_arr] = color[1]
            rgba[:, :, 2][bool_arr] = color[2]
            rgba[:, :, 3][bool_arr] = alpha
        
        ax.imshow(rgba)
        
        
    def get_composition(self, return_arr=False):
        
        # calculate the sizes of each cluster 
        fracs = [(self.clusters == clust).mean()
                 for clust in self.labels.values()]
        
        # plot this
        labs = np.array(list(self.labels.keys()))
        xs = range(len(self.labels))
        plt.figure()
        plt.bar(xs, fracs, tick_label=labs)
        plt.gca().set(ylabel = 'Fractional Composition',
                      title = 'Clustering Composition')
        
        # print values in a table
        print('Tabel of Compositions')
        for l, f in zip(labs, fracs):
            print(round(f*100, 2), '%\t', l)  
        
        if return_arr:
            return fracs
        
        
    def get_grain_count(self, return_arr=False):
        
        # count the number of segments in each cluster
        n_grains = []
        for clust in self.labels.values():
            segs = np.unique(self.mask[self.clusters == clust])
            n_grains.append(len(segs))

        # plot this
        labs = np.array(list(self.labels.keys()))
        xs = range(len(self.labels))
        plt.figure()
        plt.bar(xs, n_grains, tick_label=labs)
        plt.gca().set(ylabel = 'Count',
                      title = 'Cluster Grain Counts')
        
        # print values in a table
        print('Tabel of Grain Count')
        for l, n in zip(labs, n_grains):
            print(n, '%\t', l)    
        
        if return_arr:
            return n_grains
    

    def get_gsd(self, label, return_arr=False):
        
        # identify segments in this cluster
        clust = self.labels[label]
        segs = np.unique(self.mask[self.clusters == clust])
        
        # distribution stores
        sizes, perimeters, ratios = [np.empty(len(segs)) for _ in range(3)]
        # verbose
        bar = progress_bar(len(segs))
        
        # for every segment
        for i, seg in enumerate(segs):
            bar(i)
            bool_arr = self.mask == seg
            
            sizes[i] = bool_arr.sum()
            edges = self._get_edges(bool_arr) * bool_arr
            perimeters[i] = edges.sum()
            ratios[i] = sizes[i] / perimeters[i]
            
        # plot histograms
        for arr, title in zip([sizes, perimeters, ratios],
                              ['sizes', 'perimeters', 'ratios']):
            plt.figure()
            plt.hist(arr)
            plt.gca().set(title = label + ' ' + title + ' dist',
                          xlabel='Pixels', ylabel='Count')

        if return_arr:
            return np.vstack([sizes, perimeters, ratios]) 
        
        
    def get_span(self, label, return_arr=False):
        
        # identify segments in this cluster
        clust = self.labels[label]
        segs = np.unique(self.mask[self.clusters == clust])
        
        spans = np.empty(len(segs))
        
        # verbose
        bar = progress_bar(len(segs))
        
        # for every segment
        for i, seg in enumerate(segs):
            bar(i)
            bool_arr = self.mask == seg
            edges = self._get_edges(bool_arr) * bool_arr
            spans[i] = pdist(edges).max()
        
        # plot the histogram
        plt.figure()
        plt.hist(spans)
        plt.gca().set(title = label + ' span dist', xlabel='Pixels',
                      ylabel='Count')
        
        if return_arr:
            return spans
        
    def _get_edges(self, mask):
        """
        Take the mask and use a laplacian convolution to find the outlines.
        (same as ouline on segment_group)
        """
        # do the convolution to find the edges
        lap = np.array([[1., 1., 1.],
                        [1., -8., 1.],
                        [1., 1., 1.]])
        conv = convolve2d(mask, lap, mode='valid')
        
        # pad back boarders to have same shape as original image
        conv = np.pad(conv, 1, 'edge')
        
        # return where there is not zero gradient
        return 1. - np.isclose(conv, 0)
