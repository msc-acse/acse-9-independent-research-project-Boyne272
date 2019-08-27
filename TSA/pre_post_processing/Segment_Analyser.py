import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist
from ..tools import progress_bar

class Segment_Analyser():
    """
    Takes in a segmentation and its clustering. Gets user inputs to label
    each cluster. Then the appropirate distriubtions of the labelled clusters
    are calculated.
    
    
    Segment_Analyser(img, mask, clusters)
    
    
    Parameters
    ----------
    
    img : 2d or 3d numpy array
        An rgb, rgba, or grey scale array of the image. Used for visulisation
        only 
    
    mask : 2d numpy array
        Segmentation mask for analysing distibuions
    
    clusters : 2d numpy array
        Clustering mask for identifying different materials
    
    """
    
    def __init__(self, img, mask, clusters):
        
        # store the passed parameters
        self.img = img.copy()
        self.mask = mask.copy()
        self.clusters = clusters.copy()
        
        # dictionary stores clusters labels
        self.labels = dict([(str(c), c) for c in np.unique(clusters)])
        
        
    def set_labels(self):
        """
        Prompt the user for every cluster to give label for it. If the same
        label is given twice these clusters are merged.
        """
        
        # store the new labels
        label_dict = {}
        
        for label, clust in self.labels.items():
            
            # plot this cluster
            self.plot_cluster(label)
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
        
        
    def plot_cluster(self, label, ax=None):
        """
        Plot the original image and mask all but the cluster with the
        given label. Plots on ax if given.
        """
        
        # create an axis if not given
        if not ax:
            fig, ax = plt.subplots(figsize=[12, 12])
            
        # plot the original image
        ax.imshow(self.img)
        
        # mask all pixels translucently except for this clusters pixels
        overlay = np.zeros([self.mask.shape[0], self.mask.shape[1], 4])
        overlay[:, :, 3] = .75
        overlay[:, :, 3][self.clusters == self.labels[label]] = 0.
        
        # plot mask and label axis
        ax.imshow(overlay, cmap='binary')
        ax.set(title='Cluster ' + label)
        ax.axis('off')
        
        
    def plot_clusters(self):
        """
        Plots the indevidual clusters via plot_cluster for every cluster
        present.
        """
        # setup figure
        n_lab = len(self.labels)
        n_row = int((1 + n_lab) / 2)
        fig, axs = plt.subplots(n_row, 2, figsize=[20, n_row*10])
        
        # plot each cluster
        for label, ax in zip(self.labels.keys(), axs.ravel()[:n_lab]):
            self.plot_cluster(label, ax=ax)

        # delete unused plots
        for ax in axs.ravel()[n_lab:]:
            fig.delaxes(ax)
        plt.draw()
        
        
        
    def get_composition(self, return_arr=False):
        """
        Plots a bar graph of fractional composition for each label.
        
        These vaues are also printed in a table.
        
        If return_arr is true then a 1d relative compositions array is returned.
        """
        
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
        """
        Plots the number of segments in each cluster on a bar graph.
        
        These vaues are also printed in a table.
        
        If return_arr is true then a 1d grain count array is returned.
        """
        
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
            print(n, '\t', l)    
        
        if return_arr:
            return n_grains
    

    def get_gsd(self, label, span=True, return_arr=False):
        """
        Plots the distribution of segment areas, perimeters and the ratio of
        the two (i.e. the grain size distribution) of the cluster with the 
        given label.
        
        If span is True this will also plot the span of this clusters segments
        (can be lengthy calculation).
        
        If return_arr is true then a 2d array array is returned with 
        (size, perimeter, ratio) on the first axis and segment on the second.
        """
        
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
        fig, axs = plt.subplots(2, 2, figsize=[16, 16])
        for arr, title, ax in zip([sizes, perimeters, ratios],
                                  ['sizes', 'perimeters', 'ratios'],
                                  axs.ravel()[:3]):
            ax.hist(arr)
            ax.set(title = label + ' ' + title + ' dist',
                   xlabel='Pixels', ylabel='Count')
        
        if span:
            # calculate the span if wanted
            print('calculating span\n')
            span_arr = self._get_span(label, ax=axs.ravel()[3],
                                      return_arr=True)
            # return span as well as others if wanted
            if return_arr:
                return np.vstack([sizes, perimeters, ratios, span_arr]) 
            
        else:                    
            # delete the unused figure
            fig.delaxes(axs.ravel()[3])
            
            # return arrays wanted
            if return_arr:
                return np.vstack([sizes, perimeters, ratios]) 
        
        
    def _get_span(self, label, return_arr=False, ax=None):
        """
        Plot the span distribtuion of cluster with the given label where
        span is defined as the longest distance between two points in the
        segment.
        
        If return_arr is true then a 1d segment span array is returned.
        If ax is passed it will plot on that axis
        """
        
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
            edge_arr = self._get_edges(bool_arr) * bool_arr
            edges = np.vstack(np.where(edge_arr)).T
            spans[i] = pdist(edges).max()
        
        # create an axis if not given
        if ax == None:
            fig, ax = plt.subplots(figsize=[15, 15])
        
        # plot the histogram
        ax.hist(spans)
        ax.set(title = label + ' span dist', xlabel='Pixels', ylabel='Count')
        
        if return_arr:
            return spans
                                
        
    def _get_edges(self, mask):
        """
        Take the mask and use a laplacian convolution to find the outlines.
        (same as segment_group._ouline)
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

    
if __name__ == '__main__':
    
    # create a dummy mask
    mask_upper = np.array([[0,0,0,0],
                           [1,1,2,2],
                           [3,3,4,5],
                           [6,7,8,9]]).repeat(5, axis=0).repeat(10, axis=1)
    mask_lower = np.array([[10,11],
                           [12,13]]).repeat(10, axis=0).repeat(20, axis=1)
    mask = np.vstack((mask_upper, mask_lower))

    # create a dummy clustering
    cluster = np.zeros_like(mask)
    cluster[20:, :] = 1

    # create the analysis object
    example_obj = Segment_Analyser(mask, mask, cluster)
    # example_obj.set_labels() # prompt the user to set the labels
    example_obj.labels = {'upper':0, 'lower':1} # set the labels manually 

    # plot the generated segments and clusters
    example_obj.plot_cluster('upper')
    example_obj.plot_cluster('lower')

    # # observe the different properties
    example_obj.get_composition()
    example_obj.get_grain_count()
    example_obj.get_gsd('upper')
    example_obj.get_gsd('lower')
