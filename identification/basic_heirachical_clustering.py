import numpy as np
import matplotlib.pyplot as plt
from tools import progress_bar

class basic_heirachical_clustering():
    
    def __init__(self, features, join_by='max'):
        
        # logs which track merges and there distances
        self.dist_log = []
        self.merge_log = []
        
        # vectors
        self._vecs = features.copy()
        self._N = features.shape[0]
        self._group_count = self._N
        
        # find distances between every vector
        self._dists = np.full((self._N, self._N), np.inf)
        for _id in range(self._N):
            self._find_distance(_id)
                               
        # option of distance recaulculation
        if join_by == 'max':
            self._join = self._join_by_max
        elif join_by == 'min':
            self._join = self._join_by_min
        else:
            raise(ValueError)
    
        
    def _calc_dist(self, vec, vec_array):
            return np.sqrt(((vec - vec_array)**2).sum(axis=1))

        
    def _find_distance(self, _id):
        "Calculte the distance between this vector and all others"
        
        # find distance to every point before this id
        self._dists[_id, :_id] = self._calc_dist(self._vecs[_id],
                                                 self._vecs[:_id])
        
        # find the distance to every pstdoint after this id
        self._dists[_id+1:, _id] = self._calc_dist(self._vecs[_id],
                                                   self._vecs[_id+1:])
        return
        
        
    def _join_by_min(self, id1, id2):
        
        # find the minimum of the two distances
        dists_id1 = np.hstack([self._dists[id1, :id1], self._dists[id1:, id1]])
        dists_id2 = np.hstack([self._dists[id2, :id2], self._dists[id2:, id2]])
        dists_min = np.minimum(dists_id1, dists_id2)
        
        # make merging points inf distance apart
        dists_min[id1] = dists_min[id2] = np.inf
        
        # set the id1 distances to the min_distances
        self._dists[id1, :id1] = dists_min[:id1]
        self._dists[id1:, id1] = dists_min[id1:]
        
        # set the id2 distances to inf
        self._dists[id2, :id2] = self._dists[id2:, id2] = np.inf
    
    
    def _join_by_max(self, id1, id2):
        
        # find the minimum of the two distances
        dists_id1 = np.hstack([self._dists[id1, :id1], self._dists[id1:, id1]])
        dists_id2 = np.hstack([self._dists[id2, :id2], self._dists[id2:, id2]])
        dists_max = np.maximum(dists_id1, dists_id2)
        
        # set the id1 distances to the min_distances
        self._dists[id1, :id1] = dists_max[:id1]
        self._dists[id1:, id1] = dists_max[id1:]
        
        # set the id2 distances to inf
        self._dists[id2, :id2] = self._dists[id2:, id2] = np.inf
        
        
    def iterate(self):
        
        iterate = self._group_count-1
        
        bar = progress_bar(iterate) # verbose
        
        for i in range(iterate):
            
            if self._group_count == 1:
                print('\nFully merged graph\n')
                break
            
            id1, id2 = np.unravel_index(self._dists.argmin(),
                                        self._dists.shape)
            
            self.dist_log.append(self._dists[id1, id2])
            self.merge_log.append((id1, id2))
            self._group_count -= 1
            
            self._join(id1, id2)
             
            bar(i) # verbose
            
            
    def deriv_clustering(self, scale=3., plot=True):
        
        directory = dict([(n,n) for n in range(self._N)])
        # { vector id : group id }
        
        # find the second deriative cutoff
        y = np.array(self.dist_log)
        dy2 = y[:-2] - 2*y[1:-1] + y[2:] # central
        std = np.std(dy2)
        cutoff = (dy2 < (scale * std)).sum() - 1 # dont do the merge that was too far
        
        for i in range(cutoff):
            id1, id2 = self.merge_log[i]
            for key, val in directory.items():
                if val == id2:
                    directory[key] = id1
        
        groupings = np.unique(list(directory.values()), return_inverse=True)[1]
        n = len(np.unique(list(directory.values())))
        
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(*self._vecs[:, :2].T, c=groupings)
            ax.set(title= str(n) + ' cluster projection by derivative scale ' 
                   + str(scale))
        
        return groupings

    
    def dist_clustering(self, cutoff=1.):
        
        directory = dict([(n,n) for n in range(self._N)])
        # { vector id : group id }
        
        for dist, pair in zip(self.dist_log, self.merge_log):
            if dist < cutoff:
                for key, val in directory.items():
                    if val == pair[1]:
                        directory[key] = pair[0]
        
        groupings = np.unique(list(directory.values()), return_inverse=True)[1]
        n = len(np.unique(list(directory.values())))
                
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(*self._vecs[:, :2].T, c=groupings)
            ax.set(title= str(n) + ' cluster projection by distance cutoff '
                   + str(cutoff))
        
        return groupings

    
    
    def dist_plot(self, option, ax=None, last_few=True):

        if not ax:
            fig, ax = plt.subplots(figsize=[12, 10])
        n = 20 if last_few else self._N
        y = np.array(self.dist_log)
        
        if option == 'dists':
            x = np.arange(0, self._N-1)
            ax.plot(x[-n:], y[-n:], '-o')
            ax.set(title='Join Distances')
            
        elif option == '1st':
            x = np.arange(1, self._N-1)
            dy = y[1:] - y[:-1] # forward
            ax.plot(x[-n:], dy[-n:], '-o')
            ax.set(title='First Diff')
            
        elif option == '2nd':
            x = np.arange(1, self._N-2)
            dy2 = y[:-2] - 2*y[1:-1] + y[2:] # central
            ax.plot(x[-n:], dy2[-n:], '-o')
            ax.set(title='Second Diff')
            
        # composit option
        elif option == 'all':
            plt.close()
            fig, axs = plt.subplots(2,2, figsize=[15,15])
            self.dist_plot('dists', ax=axs[0,0])
            self.dist_plot('dists', ax=axs[0,1], last_few=False)
            self.dist_plot('1st', ax=axs[1,0])
            self.dist_plot('2nd', ax=axs[1,1])
            
        else:
            plt.close()
            print('opt not known')
            
            
if __name__ == '__main__':
    
    # generate dummy 3d data
    np.random.seed(10)
    features = np.random.normal(size=[500, 2])
    features[50:, :] += 4
    features[100:, :] += 4
    features[150:, 0] += 4
    features[250:300, 1] += 10
    features[350:400, 0] += 10
    features[-1, -1] += 20
    
#     plt.figure(figsize=[10, 10])
    plt.scatter(*features.T)
    plt.title("dummy data")

    obj = basic_heirachical_clustering(features)
    obj.iterate()
    obj.dist_plot('all')
    obj.deriv_clustering(3, True)
    