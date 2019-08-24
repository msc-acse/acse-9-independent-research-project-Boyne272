import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from image_filters import Image_Filters

class recombination_wrapper():
    
    valid_kwargs = [''] # for validating kwarg input ##############
    
    def __init__(self, img, mask, filts=None, **kwargs):
        
        # validate image and mask
        assert img.shape[:2] == mask.shape, \
            'image and mask must have same x,y dimensions'
        assert len(img.shape) == 3 and img.shape[2] == 3, \
            'image must be in shape x, y, rgb'
        
        # set tuneable parameters
        self._init_params(**kwargs)
        
        # store input arrays
        self.img = img
        self.grey_img = img.mean(axis=2)
        self.orig_mask = mask
        self.work_mask = mask.copy()
        self._ydim, self._xdim = mask.shape
        
        # create derived arrays
        if filts == None:
            self.filts = Image_Filters()
            print("Calcuating filtered images ", end='')
            self.orig_edges = self.filts.mask_outine(mask)
            print(".", end='')
            self.sobel_array = self.filts.sobel_edge_detection(self.grey_img)
            print(".", end='')
            gabor_filts = self.filts.gabor_bank(13, 5, 20, gamma=1.)
            self.gabor_array = self.filts.apply_bank(gabor_filts, self.grey_img)
            print(".", end='')
            self.laws_array = self.filts.laws_texture_features(self.grey_img,
                                                               self._law_size)
            print(".")
        else: # this is to save time during code development 
            self.filts = filts
            self.orig_edges = filts.img_store['orig_edges']
            self.sobel_array = filts.img_store['sobel_array']
            self.gabor_array = filts.img_store['gabor_array']
            self.laws_array = filts.img_store['laws_array']
        
        # create the segment objects
        self.seg_objs = self._create_segments()
        self.seg_directory = dict([(n, n) for n in self.seg_objs.keys()])
        # { original id : merged id }, if key != val the segment was merged
        self._to_merge = []
        
        # placeholder for basic_heirachical_clustering obj
        self.cluster_obj, self.clusters = self._cluster()
        
    
    def _init_params(self, **kwargs):
        """
        Parameters that can be tuned are set here and can be overridden
        by passing in the correct kwarg, giving flexibility without the
        need to significantly complicate later code.
        """
        
        # the number of bins used in binning each color channel
        self._n_color_bins = 10
        
        # the number of bins used in binning the gabor image
        self._n_gabor_bins = 10
        
        # scale the color or gabor features in clustering
        self._color_scale = 1.
        self._gabor_scale = 1.
        self._laws_scale = 1.
        
        # scale of closeness in clustering by texture/color (2nd deriv, std scaled)
        self._cluster_scale = 3.
        
        # the minimum size a segment can be
        self._min_seg_size = 50
        
        # size of the laws texture filters to convolve (x5)
        self._law_size = 1
        
        # threshold for a edge to be considered present (std)
        self._bound_present = 2.
        self._bound_absent = -2
        
        # let keyword arguments override these defaults 
        for key, val in kwargs.items():
            exec('self._' + key + '=' + str(val))
        
    
    def _create_segments(self):
        """
        Initalise the segment objects from the original mask. If a list of
        ids is passed in update then only those segments are recreated
        provided they have not been removed yet
        """
        
        seg_ids = np.unique(self.orig_mask)
        segments = {}
        
        # verbose
        N = len(seg_ids)
        bar = progress_bar(N)
        print('Initalising %i segments' % N)
        
        for n, i in enumerate(seg_ids):
            bar(n) # update progress bar
            
            bool_arr = self.orig_mask==i # segments pixel cords
            cords = np.array(np.where(bool_arr)[::-1]).T
            # [::-1] as np.where returns y, x format
            # T as we want in form (N, 2)
            
            bool_arr = bool_arr * self.orig_edges # segments edge pixels
            edges = np.array(np.where(bool_arr)[::-1]).T
            
            segments[i] = (segment(cords, edges, i, self))
            
        print('\n')
        return segments
    
    
    def _update_segments(self, seg_ids):
        """
        Initalise the segment objects from the original mask. If a list of
        ids is passed in update then only those segments are recreated
        provided they have not been removed yet
        """
                
        # verbose
        N = len(seg_ids)
        bar = progress_bar(N)
        print('Updating %i segments' % N)
        
        for n, _id in enumerate(seg_ids):
            # kill all non_existent segments objects
            if self.seg_directory[_id] != _id:
                del self.seg_objs[_id]
            
            # recreate changed segment objects
            else:
                bool_arr = self.work_mask==_id # segments pixel cords
                cords = np.array(np.where(bool_arr)[::-1]).T
                # [::-1] as np.where returns y, x format
                # T as we want in form (N, 2)

                bool_arr = bool_arr * self.orig_edges # segments edge pixels
                edges = np.array(np.where(bool_arr)[::-1]).T

                self.seg_objs[_id] = (segment(cords, edges, _id, self))
            
            bar(n)

            
        print('\n')
        return 
    
        
    def _merge_segments(self):
        """
        For every pair in the to_merge list set the working mask to replace
        the 2nd id with the 1st one. 
        
        Caution is taken to keep track of who has merged so that merging with
        a segment which has already been merged is handelled appropirately. 
        This is done using the seg_directory.
        
        Any changed segment and its neighbours are tracked in order to only
        update segments which have been changed
        """
        to_recalculate = set()
        
        print('Merging %i segments\n' % len(self._to_merge))
        for seg_id1, seg_id2 in self._to_merge:
            
            # select the up to date ids
            id1 = self.seg_directory[seg_id1]
            id2 = self.seg_directory[seg_id2]
            
            # update the working mask
            self.work_mask[self.work_mask == id2] = id1
            
            # update the segment directory
            for key, val in self.seg_directory.items():
                if val == id2:
                    self.seg_directory[key] = id1
                    
            # track which segments to recalculate
            to_recalculate.update([*self.seg_objs[id1].neighbours, 
                                   *self.seg_objs[id2].neighbours])
        
        # clear the merging list
        self._to_merge = []
        return to_recalculate
    
    
    def _size_scan(self):
        """
        Merge small sections with their largest neighbour 
        """
        
        # identify segements to be merged
        for seg1_id, seg1 in self.seg_objs.items():
                
            if seg1.metric_dict['size'] < self._min_seg_size:
                max_index = np.argmax([self.seg_objs[_id].metric_dict['size']
                                       for _id in seg1.neighbours])
                to_merge_id = seg1.neighbours[max_index]
                self._to_merge.append((to_merge_id, seg1_id))
        
        # merge segments then recalculate
        recalc = self._merge_segments()
        self._update_segments(recalc)
        self.cluster_obj, self.clusters = self._cluster()
        
        
    def _texture_scan(self):
        """
        Scan for adjasent segments to merge if there is no apparent edge
        between them
        """
        
        scanned = []
        
        for seg in self.seg_objs.values():
            for neigh_id in seg.neighbours:
                if neigh_id not in scanned:
                    neigh = self.seg_objs[neigh_id]
                    avg_bound_val = (seg.metric_dict['boundaries_dict'][neigh_id] + 
                                     neigh.metric_dict['boundaries_dict'][seg.id]) / 2
                    if seg.group == neigh.group and not avg_bound_val < self._bound_present:
                        self._to_merge.append((seg.id, neigh_id))
                    
            scanned.append(seg.id)

        # merge segments then recalculate
        recalc = self._merge_segments()
        self._update_segments(recalc)
        self.cluster_obj, self.clusters = self._cluster()
        
        
    def _edge_scan(self):
        
        scanned = []
        
        for seg in self.seg_objs.values():
            for neigh_id in seg.neighbours:
                if neigh_id not in scanned:
                    neigh = self.seg_objs[neigh_id]
                    avg_bound_val = (seg.metric_dict['boundaries_dict'][neigh_id] + 
                                     neigh.metric_dict['boundaries_dict'][seg.id]) / 2
                    if avg_bound_val < self._bound_absent:
                        self._to_merge.append((seg.id, neigh_id))
                    
            scanned.append(seg.id)
            
        # merge segments then recalculate
        recalc = self._merge_segments()
        self._update_segments(recalc)
        self.cluster_obj, self.clusters = self._cluster()
        
        
    def _cluster(self):
        """
        """
        # create the features and the clustering object
        feat_vecs = np.vstack([seg.feat_vec
                               for seg in self.seg_objs.values()])
        obj = basic_heirachical_clustering(feat_vecs)
        
        print('Clustering segments')
        obj.iterate()
        
        cluster = obj.deriv_clustering(scale=self._cluster_scale, plot=False)
        print('\nNumber of clusters %i \n' % len(set(cluster)))
        
        # tell each segment what group it is in
        for grp, seg in zip(cluster, self.seg_objs.values()):
            seg.group = grp
        
        return obj, np.unique(cluster)
    
            
    def plot(self, option='default', ax=None, **kwargs):
        
        # if no axis given create one
        if not ax:
            fig, ax = plt.subplots(figsize=[22, 11])
        
        
        # base options
        if option == 'img':
            ax.imshow(self.img)
            ax.set(title='original image')
            
        elif option == 'img_grey':
            ax.imshow(self.grey_img, cmap='gray')
            ax.set(title='original image greyscaled')
        
        elif option == 'orig_mask':
            ax.imshow(self.orig_mask)
            ax.set(title='original mask')
            
        elif option == 'orig_edges':
            ax.imshow(self.filts.rgba(self.orig_edges, color='r'))
            ax.set(title='original mask outline')
            
        elif option == 'merged_mask':
            ax.imshow(self.work_mask)
            ax.set(title='current merged mask')
            
        elif option == 'merged_edges':
            edges = self.filts.mask_outine(self.work_mask)
            ax.imshow(self.filts.rgba(edges, color='g'))
            ax.set(title='original mask outline')
            
            
        # composite options
        elif option == 'both' or option == 'default':
            self.plot('img', ax=ax)
            self.plot('orig_edges', ax=ax)
            self.plot('merged_edges', ax=ax)
            ax.set(title='pre and post merging edges')
            
        elif option == 'merged':
            self.plot('img', ax=ax)
            self.plot('merged_edges', ax=ax)
            ax.set(title='merged segments')
            
        elif option == 'original':
            self.plot('img', ax=ax)
            self.plot('orig_edges', ax=ax)
            ax.set(title='original segments')
            
            
        # clustering options
        elif option == 'shade_clusters':
            self.plot('img', ax=ax)
            colors = np.random.rand(len(self.clusters)+1, 3)
            rgba = np.zeros([self._ydim, self._xdim, 4])
            for seg in self.seg_objs.values():
                rgba = seg.fill_mask(rgba, color=colors[seg.group])
            ax.imshow(rgba)
            ax.set(title='Texture/Color Clustering')
        
        elif option == 'one_cluster':
            assert 'cluster' in kwargs.keys(), 'must specify which cluster'
            self.plot('img', ax=ax)
            rgba = np.zeros([self._ydim, self._xdim, 4])
            rgba[:, :, 3] = 1
            for seg in self.seg_objs.values():
                if seg.group == kwargs['cluster']:
                    rgba = seg.fill_mask(rgba, alpha=0)
            ax.imshow(rgba)
            
        elif option == 'all_cluster':
            plt.close()
            n_clusters = len(self.clusters)
            y = int(n_clusters/2) + 1
            fig, axs = plt.subplots(y, 2, figsize=[24, 6*y])
            
            for ax, i in zip(axs.ravel()[:n_clusters],
                             self.clusters):
                self.plot('one_cluster', ax=ax, cluster=i)
                ax.set(title='Cluster %i'%i)
            
            for ax in axs.ravel()[n_clusters:]:
                fig.delaxes(ax) # remove unused axis
        
        
        # indevidual segment options
        elif option == 'seg_edge':
            assert 'segment' in kwargs.keys(), "specifiy which segment to plot"
            self.seg_objs[kwargs['segment']].edge_plot(ax=ax)
            
        elif option == 'seg_fill':
            assert 'segment' in kwargs.keys(), "specifiy which segment to plot"
            rgba = np.zeros([self._ydim, self._xdim, 4])
            rgba = self.seg_objs[kwargs['segment']].fill_mask(rgba, alpha=.5)
            ax.imshow(rgba)
            ax.set(title='Segment ' + str(kwargs['segment']))
            
            
        # confidence in the existsence of each edge
        elif option == 'edge_confidence':
            mask = np.full(self.work_mask.shape, np.nan)
            for seg in self.seg_objs.values():
                for neigh_id in seg.neighbours:
                    Xs, Ys = [*seg.edge_dict[neigh_id].T]
                    mask[Ys, Xs] = seg.metric_dict['boundaries_dict'][neigh_id]
                    
            self.plot('img', ax=ax)
            col = ax.imshow(mask, cmap='RdYlGn')
            plt.colorbar(col, fraction=0.046/2, pad=0.04)
            ax.set(title='Confidence of edge existence')
            
        elif option == 'edge_hist':
            plt.close()
            bins = kwargs['bins'] if 'bins' in kwargs.keys() else 10
            confidences = []
            for seg in self.seg_objs.values():
                for val in seg.metric_dict['boundaries_dict'].values():
                    confidences.append(val)
            print(confidences)
            plt.hist(confidences, bins)
            plt.title('Histogram of edge confidences')
            
            
        # option not known
        else:
            print('option not recognised, allowed options are:')
            for s in ('default', 'both', 'merged', 'original', 'img',
                      'img_grey', 'orig_mask', 'orig_edges', 'merged_mask',
                      'merged_edges', 'seg_edge', 'seg_fill',
                      'edge_confidence', 'edge_hist'): 
                print('\t-' + s)


class segment():
    
    def __init__(self, cords, edges, index, wrapper, **kwargs):
        """
        
        """
        # store the given params and some convinient values
        self.id = index
        self.cords = cords
        self.edges = edges
        self._wrapper = wrapper

        # store this segments group id
        self.group = None
        
        # find which edges are with each segments
        self.edge_dict = self._identify_edges()
        self.neighbours = list(self.edge_dict.keys())
        
        # find this segments metrics
        self.metric_dict, self.feat_vec = self._calculate_metrics()
    
    
    def _identify_edges(self):
        """
        For every edge use the wrapper mask to find what other segments
        share this edge and store them in a dictionary so that all pixels
        on a particular border can be indexed.
        """
        
        # dictionary for neighbour_id:[edge_cordinates]
        edge_dict = {}
        
        for x, y in self.edges:
            adj_cords = self._pixel_neighbours(x, y)
            # indexs backwards as mask has y, x format
            adj_segs = self._wrapper.work_mask[adj_cords[:, 1], adj_cords[:, 0]]
            unique_segs = np.unique(adj_segs)
            
            # store cordinates that boarder each neighbour appropirately
            for i in unique_segs:
                if i != self.id:
                    
                    # if neighbour segment already in edge_dict append to it
                    # else create it as a new entry
                    edge_dict.setdefault(int(i),[]).append((x, y))
                        
        # set every edge entry to an array
        for key, lst in edge_dict.items():
            edge_dict[key] = np.array(lst)

        return edge_dict
    

    def _calculate_metrics(self):
        """
        Calculate various properties of this segment which will be used
        in the merging decision process
        """
        
        mets = {}
        
        # shape feats
        mets['size'] = self.cords.shape[0]
        mets['perimeter'] = self.edges.shape[0]
        
        # color feats
        opts = {'bins':self._wrapper._n_color_bins, 'range':(0.,1.),
                'density':True}
        rgbs = self._wrapper.img[self.cords[:, 1], self.cords[:, 0]]
        mets['color_hist_red'] = np.histogram(rgbs[:, 0], **opts)[0] / self._wrapper._n_color_bins
        mets['color_hist_grn'] = np.histogram(rgbs[:, 1], **opts)[0] / self._wrapper._n_color_bins
        mets['color_hist_blu'] = np.histogram(rgbs[:, 2], **opts)[0] / self._wrapper._n_color_bins
        
        mets['color_avg'] = rgbs.mean(axis=0)
        mets['color_std'] = rgbs.std(axis=0)

        # boundary feats
        mets['boundaries_dict'] = {}
        for n in self.neighbours:
            Xs, Ys = self.edge_dict[n][:, 0], self.edge_dict[n][:, 1]
            lst = [self._wrapper.sobel_array[Ys, Xs, i].mean()**2
                   for i in range(4)] # one for each filter direction
            mets['boundaries_dict'][n] = np.log(np.mean(lst)) #############################
        
        # texture features
        gabs = self._wrapper.gabor_array[self.cords[:, 1], self.cords[:, 0]]
        mets['gabor_hist'] = np.histogram(gabs, bins=self._wrapper._n_gabor_bins,
                                          range=(0.,1.), density=True)[0] / self._wrapper._n_gabor_bins
        mets['gabor_avg'] = gabs.mean()
        mets['gabor_std'] = gabs.std()
        
        laws = self._wrapper.laws_array[self.cords[:, 1], self.cords[:, 0], :]
        mets['laws_avg'] = laws.mean(axis=0)
        mets['laws_stds'] = laws.std(axis=0)
           
        # feature vector
        feat = np.hstack((
#                           mets['color_hist_red'] * self._wrapper._color_scale,
#                           mets['color_hist_grn'] * self._wrapper._color_scale,
#                           mets['color_hist_blu'] * self._wrapper._color_scale,
                          mets['color_avg'] * self._wrapper._color_scale,
                          mets['color_std'] * self._wrapper._color_scale,
#                           mets['gabor_hist'] * self._wrapper._gabor_scale,
                          mets['gabor_avg'] * self._wrapper._gabor_scale,
                          mets['gabor_std'] * self._wrapper._gabor_scale,
#                           mets['laws_avg'] * self._wrapper._laws_scale,
#                           mets['laws_stds'] * self._wrapper._laws_scale
                        ))
            
        return mets, feat
       
        
    def _pixel_neighbours(self, x, y):
        """
        Find the neighbours to the x and y pixel. This method is stored on the
        wrapper as it knows the dimensions of the whole image.
        """
        return np.array([(x_, y_) for x_ in range(x-1, x+2)
                                  for y_ in range(y-1, y+2)
                         if ((x != x_ or y != y_) and   # not the center
                             (0 <= x_ < self._wrapper._xdim) and # not outside x range
                             (0 <= y_ < self._wrapper._ydim))])  # not outside y range
    
    
    def fill_mask(self, arr, color=[], alpha=0.4):
        "build up an rgba mask without large memory consumption"
        
        assert len(arr.shape) == 3, 'arr must be 3d'
        assert arr.shape[-1] == 4, 'arr must have shape (X, Y, 4)'
        
        if len(color) == 0:
                color = np.random.rand(3)
        arr[self.cords[:,1], self.cords[:,0], 0] = color[0]
        arr[self.cords[:,1], self.cords[:,0], 1] = color[1]
        arr[self.cords[:,1], self.cords[:,0], 2] = color[2]
        arr[self.cords[:,1], self.cords[:,0], 3] = alpha
        return arr
    
    
    def edge_plot(self, ax):
        ""        
        for n_id, n_cords in self.edge_dict.items():
            ax.plot(n_cords[:,0], n_cords[:,1], 'x', ms=5, label=n_id)
        ax.legend()
                