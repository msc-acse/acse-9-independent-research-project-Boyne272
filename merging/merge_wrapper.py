import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

class merge_wrapper():
    
    # edge filters
    _S0_3 = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    _S90_3 = _S0_3.T
    _S45_3 = np.array([[0 , 1, 2],
                      [-1, 0, 1],
                      [-2,-1, 0]])
    _S135_3 = _S45_3.T
    _S0_5 = np.array([[1, 2, 3, 2, 1],
                     [2, 3, 5, 3, 2],
                     [0, 0, 0, 0, 0],
                     [-2,-3,-5,-3,-2],
                     [-1,-2,-3,-2,-1]])
    _S90_5 = _S0_5.T
    _S45_5 = np.array([[0 , 2, 3, 2, 1],
                      [-2, 0, 3, 5, 2],
                      [-3,-3, 0, 3, 3],
                      [-2,-5,-3, 0, 2],
                      [-1,-2,-3,-2, 0]])
    _S135_5 = _S45_5.T
    _Laplacian_3 = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])
    _Laplacian_5 = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, -24, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]])
    _avg_7 = np.ones([7,7])/49.
    _avg_5 = np.ones([5,5])/25.
    _avg_3 = np.ones([3,3])/9.

    
    def __init__(self, img, mask, **kwargs):
        
        # validate image and mask
        assert img.shape[:2] == mask.shape, \
            'image and mask must have same x,y dimensions'
        
        # set the tuneable parameters
        self._initalise_params(**kwargs)
        
        # store input arrays
        self.img = img 
        self.orig_mask = mask
        self.work_mask = mask.copy()
        self._ydim, self._xdim = mask.shape
        
        # create derived arrays
        self.orig_edges = self._outline(mask)
        self.work_edges = self.orig_edges.copy()
        self.bound_array = self._edge_detection()

        # create the segment objects
        self.segments, self.seg_ids = self.make_segments()
        
        # create working vairables
        self.to_merge = []
        self.directory = dict([(n, n) for n in self.seg_ids])
    
    
    def _initalise_params(self, **kwargs):
        """
        All parameters that can be tuned are set here and can be overridden
        by passing in the correct kwarg. This gives flexibility without the
        need to add many if statments to other methods.
        """
        
        self._merge_size_rel = 0.02 # relative size to merge if one neighbour
        self._merge_size_cutoff = 200 # relative size maximum to merge

        self._color_nbins = 20
        self._color_sim_thresh = .2 # should be below to merge
        self._color_dif_thresh = .5

        self._edge_present = 2., 5., 7. # above edge is present
        self._edge_absent = .1, .3, .5 # below edge is not present
        
        self._confidence_thresh = 0 # what measure for confidence of merging
        self._max_passes = 10
        
        # let keyword arguments override these defaults 
        for key, val in kwargs.items():
            exec('self._' + key + '=' + str(val))
    
    
    def _edge_detection(self):
        """
        """
        
        zeros = np.zeros_like(self.orig_mask)
        edge_det = np.dstack([zeros, zeros, zeros, zeros])
        grey_img = self.img.mean(axis=2)
        for i, filt in enumerate([self._S0_5, self._S45_5,
                               self._S90_5, self._S135_5]):
            edge_det[:, :, i] = sig.convolve2d(grey_img, filt, mode='same')

#           # verbose
#         for i in range(4):
#             plt.figure(figsize=[22,22])
#             plt.imshow(edge_det[:,:,i])
        
        return edge_det.astype(float)


    def make_segments(self, seg_ids=[]):
        """
        Create the segment objects from the current working mask. If seg_ids
        is passed then only those segments
        """
        
        # setup
        segments = {} # holds {int id : obj segment}
        if len(seg_ids) == 0:
            seg_ids = np.unique(self.work_mask)
        else:
            # if we are updating segements just edit the pre-existsing dict
            segments = self.segments
        
        # verbose
        N = len(seg_ids)
        bar = progress_bar(N)
        print('Initalising %i segments' % N)
        
        for n, i in enumerate(seg_ids):
            bar(n) # update progress bar
            
            bool_arr = self.work_mask==i # segments pixel cords
            cords = np.array(np.where(bool_arr)[::-1]).T
            # [::-1] as np.where returns y,x format
            # T as we want in form (N, 2)
            
            bool_arr = bool_arr * self.work_edges # segments edge pixels
            edges = np.array(np.where(bool_arr)[::-1]).T
            
            if len(cords) == 0: # if we are updating a dead segment
                ########################### fix me Im hideous
                del segments[i]
            else:
                segments[i] = (segment(cords, edges, i, self))
            
        print('\n')
        return segments, seg_ids
    
        
    def neighbours(self, x, y):
        """
        Find the neighbours to the x and y pixel. This method is stored on the
        wrapper as it knows the dimensions of the whole image.
        """
        return np.array([(x_, y_) for x_ in range(x-1, x+2)
                                  for y_ in range(y-1, y+2)
                         if ((x != x_ or y != y_) and   # not the center
                             (0 <= x_ < self._xdim) and # not outside x range
                             (0 <= y_ < self._ydim))])  # not outside y range
    
    
    def scan(self):
        """
        Scan over the whole mask and compare every segment pair, being
        careful to only compare each pair once. These are stored in the
        to_merge list so that the order of comparison is not significnt
        to the specific paring
        """
        
        already_scanned = [] # stores fully scanned segments
        
        for seg in self.segments.values():
            for neigh_id in seg.neighbours: # for every segment pair
                if neigh_id not in already_scanned:
                    if (seg.compare(self.segments[neigh_id]) > 
                        self._confidence_thresh):
                        self.to_merge.append((seg.id, neigh_id))

            already_scanned.append(seg.id)
                
    
    def merge(self):
        """
        For every pair in the to_merge list set the working mask to replace
        the 2nd id with the 1st one. Caution is taken to keep track of
        who has merged so that merging with a segment which has already been
        merged is still possible.
        """
        to_update = []
        for seg_id1, seg_id2 in self.to_merge:
            
            # select the up to date ids
            seg_id1 = self.directory[seg_id1]
            seg_id2 = self.directory[seg_id2]
            
            # update the working mask
            self.work_mask[self.work_mask == seg_id2] = seg_id1
            
            # update the directory which tracks merged segments incase one
            # trys to merge with a segment that no longer exists
            for key, val in self.directory.items():
                if val == seg_id2:
                    self.directory[key] = seg_id1
        
            to_update += [*self.segments[seg_id1].neighbours,
                          *self.segments[seg_id2].neighbours]
        
        # clear the merging list
        to_update = np.unique(to_update)
        self.to_merge = []
        return to_update
        
                    
    def iterate(self):
        "The loop to be carried out until the new mesh is made"
        counter = 1
        while (counter-1) < self._max_passes:
            
            print("Starting Iteration ", counter)
            
            self.scan() # find which segments need to be merged
            if not self.to_merge: # if to_merge is empty stop
                print('No segments to merge, terminating\n')
                break
            print("merging ", len(self.to_merge), " segments\n")
            to_update = self.merge() # merge all nessesary segments
            
            # recreate the segments to represent the new mask
            print(to_update)
            self.segments, self.seg_ids = self.make_segments(to_update)
            counter += 1
            
            
    def plot(self, option='default', ax=None, **kwargs):
        
        # validate input
        all_options = ['default', 'compare1', 'compare2', 'both', 'merged',
                       'original', 'img', 'img_grey', 'orig_mask', 'orig_edges',
                       'merged_mask', 'merged_edges',
                       'seg_fill', 'seg_edge', 'confidence']
        assert option in all_options, "option " + option + " not recognised"
        
        # if no axis given create one
        if not ax:
            fig, ax = plt.subplots(figsize=[22, 22])
        
        
        # plot combined options
        if option == 'default' or option == 'compare1':
            self.plot('img', ax=ax)
            self.plot('merged_edges', ax=ax)
            
            # fill in every segment that no longer exists
            post_merge_indexs = list(self.directory.values())
            NLE = [i for i in np.unique(self.orig_mask) # no longer exists
                   if i not in post_merge_indexs
                   or post_merge_indexs.count(i) > 1]
            NLE_mask = np.isin(self.orig_mask, NLE)
            NLE_rgba = self._rgba(NLE_mask, opaqueness=.5)
            ax.imshow(NLE_rgba)
            ax.set(title='merging comparison 1')

        if option == 'compare2':
            self.plot('img', ax=ax)
            self.plot('orig_edges', ax=ax)
            
            # fill in every segment that was merged
            post_merge_indexs = list(self.directory.values())
            changed_indexs = list(set([i for i in post_merge_indexs if
                                       post_merge_indexs.count(i) > 1]))
            for index in changed_indexs:
                self.segments[index].plot(self.orig_mask.shape, ax=ax, 
                                          opt='fill')
            ax.set(title='merging comparison 2')
            
        elif option == 'both':
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
            
            
        # plot base options
        elif option == 'img':
            ax.imshow(self.img)
            ax.set(title='original image')
            
        elif option == 'img_grey':
            ax.imshow(self.img.mean(axis=2), cmap='gray')
            ax.set(title='original image greyscaled')
        
        elif option == 'orig_mask':
            ax.imshow(self.orig_mask)
            ax.set(title='original mask')
            
        elif option == 'orig_edges':
            edges = self._outline(self.orig_mask)
            rgba = self._rgba(edges, color='r')
            ax.imshow(rgba)
            ax.set(title='original mask outline')
            
        elif option == 'merged_mask':
            ax.imshow(self.work_mask)
            ax.set(title='current merged mask')
            
        elif option == 'merged_edges':
            edges = self._outline(self.work_mask)
            rgba = self._rgba(edges, color='g')
            ax.imshow(rgba)
            ax.set(title='original mask outline')
        
        
        # kwarg requiered options
        elif option == 'seg_edge' or option == 'seg_fill':
            assert 'seg' in kwargs.keys(),\
                "must specifiy which segment to plot"
            self.segments[kwargs['seg']].plot(ax=ax, opt=option[4:])
            ax.set(title='segment ' + str(kwargs['seg']))
            
        elif option == 'confidence':
            assert 'sec' in kwargs.keys(),\
                "must specifiy list of color, edge, size to be used"
            assert len(kwargs['sec']) == 3,\
                "must give exactly 3 options"
            
            mask = np.zeros_like(self.work_mask)
            mask[:,:] = np.nan
            
            for seg in self.segments.values():
                for neigh_id in seg.neighbours:
                    val = seg.compare(self.segments[neigh_id], *kwargs['sec'])
                    Xs, Ys = [*seg.edge_dict[neigh_id].T]
                    mask[Ys, Xs] = val

            self.plot('img_grey', ax=ax)
            col = ax.imshow(mask, cmap='RdYlBu')
            plt.colorbar(col, fraction=0.046/2, pad=0.04)
            ax.set(title='Confidence on merging using sec='+str(kwargs['sec']))
            
        
    def _rgba(self, mask, color='r', opaqueness=1):
        "Take a 2d mask and return a 4d rgba mask for imshow overlaying"
        
        # create the transparent mask
        zeros = np.zeros_like(mask)
        rgba = np.dstack([zeros, zeros, zeros, mask*opaqueness])
        
        # set the correct color channel
        i = ['r', 'g', 'b'].index(color)
        rgba[:, :, i] = mask
            
        return rgba
    
    
    def _outline(self, mask, opt='full'):
        """
        Take a 2d mask and use a laplacian convolution to find the segment 
        outlines for plotting. Option decides if all directions are to be
        included 'full' or just horizontal and vertical ones 'edge'
        """
        laplacian = np.ones([3, 3])
        
        if opt == 'full':
            laplacian[1, 1] = -8
            
        elif opt == 'edge':
            laplacian[1, 1] = -4
            laplacian[[0, 2, 0, 2], [0, 0, 2, 2]] = 0
            
        conv = sig.convolve2d(mask, laplacian, mode='valid')
        conv = np.pad(conv, 1, 'edge') # ignore edges
        not_edge = np.isclose(conv, 0)
        return 1. - not_edge
class segment():
    
    def __init__(self, cords, edges, index, wrapper, **kwargs):
        """
        
        """
        # store the given params and some convinient values
        self.id = index
        self.cords = cords
        self.edges = edges
        self._rgbs = wrapper.img[cords[:, 1], cords[:, 0]]
        self._wrapper = wrapper

        # store this segments group id
#         self.group = index
        
        # find which edges are with each segments
        self.edge_dict = self._identify_edges()
        self.neighbours = list(self.edge_dict.keys())
        
        # find this segments metrics
        self.metrics = self._calculate_metrics()
    
    
    def _identify_edges(self):
        """
        For every edge use the wrapper mask to find what other segments
        share this edge and store them in a dictionary so that all pixels
        on a particular border can be indexed.
        """
        
        # dictionary for neighbour_id:[edge_cordinates]
        edge_dict = {}
        
        for x, y in self.edges:
            adj_cords = self._wrapper.neighbours(x, y)
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
        
        mets['size'] = self.cords.shape[0]
        mets['perimeter'] = self.edges.shape[0]

        hists = [np.histogram(self._rgbs[:, i],
                              bins=self._wrapper._color_nbins,
                              range=(0.,1.), 
                              density=True)[0] for i in range(3)]
        cum_hists = [np.cumsum(h) for h in hists]
        mets['col_hist'] = np.array([ch/ch[-1] for ch in cum_hists])
        
        mets['boundaries_dict'] = {}
        for n in self.neighbours:
            Xs, Ys = self.edge_dict[n][:, 0], self.edge_dict[n][:, 1]
            lst = [self._wrapper.bound_array[Ys, Xs, i].mean()**2
                   for i in range(4)] # one for each filter direction
            mets['boundaries_dict'][n] = np.mean(lst)
            
        return mets
       
    
    def compare(self, other, size=True, edge=True, col=True):
        """
        Compare this and other segments, and return an integer value which is
        the confidence that they should be merged. Every tuneable parameter
        used here (stored on the wrapper object) these can be set by passing 
        the kwarg when creating the wrapper.
        """
        m1, m2 = self.metrics, other.metrics
        confidence = 0
        
        # merge segments if one is enclosed by the other and is small (0 or 6)
        if size:
            if self.neighbours == [other.id] or other.neighbours == [other.id]:
                if (m1['size'] < m2['size'] * self._wrapper._merge_size_rel or
                    m2['size'] < m1['size'] * self._wrapper._merge_size_rel) and \
                   (m1['size'] < self._wrapper._merge_size_cutoff or 
                    m2['size'] < self._wrapper._merge_size_cutoff):
                    return 6 # arbitarily positive

        # KS test the color spectra (-3, 3)
        if col:
            diff = abs(m1['col_hist'] - m2['col_hist'])
            diff_max = diff.max(axis=1)
            sim_confid = sum(diff_max < self._wrapper._color_sim_thresh)
            dif_confid = sum(diff_max > self._wrapper._color_dif_thresh)
            confidence += sim_confid - dif_confid

        # boundary test (-3, 3)
        if edge:
            boundary_val = (m1['boundaries_dict'][other.id] +
                            m2['boundaries_dict'][self.id])/2
            # average the measure too look at the combined edges
            exists_confid = (boundary_val > np.array(self._wrapper._edge_present)).sum()
            doesnt_confid = (boundary_val < np.array(self._wrapper._edge_absent)).sum()
            confidence += doesnt_confid - exists_confid

        return confidence
    
    def plot(self, ax, opt='edge'):
        """
        Create a rgba matrix for this segments with the option to plot the:
            - 'edge' just give the outline of the image
            - 'fill' the entire segment, translucent to see the image beneath
            - 'vertices' the corders of each segement (REMOVED)
        """
        zeros = np.zeros(self._wrapper.orig_mask.shape)
        rgba = np.dstack([zeros, zeros, zeros, zeros])
        
        if opt == 'edge':
            for n_id, n_cords in self.edge_dict.items():
                ax.plot(n_cords[:,0], n_cords[:,1], 'x', ms=5, label=n_id)
            ax.legend()
                
        elif opt == 'fill':
            red, green, blue = np.random.rand(3) / 2
            rgba[self.cords[:,1], self.cords[:,0], 0] = red
            rgba[self.cords[:,1], self.cords[:,0], 1] = green
            rgba[self.cords[:,1], self.cords[:,0], 2] = blue
            rgba[self.cords[:,1], self.cords[:,0], 3] = 0.6
            ax.imshow(rgba)
                