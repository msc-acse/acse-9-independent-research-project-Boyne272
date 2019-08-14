import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import skimage.feature as skf
from scipy import ndimage as ndi


class Image_processor():
    
    def __init__(self, img=np.array([]), path=''):
        """
        A class to wrap all image feature extraction and preprocessing.
        Use path to load an image from file.
        """

        if path:
            img = ski.io.imread(path)[:, :, :3] # exclude any alpha channels
        else:
            assert img.size, 'Must either pass a path or image'
            
        assert type(img) == type(np.array([])), 'img must be an array'
        assert img.ndim == 3, 'image must have form x, y, color'
        assert img.shape[-1] == 3, 'image must have 3 color channels'
        
        float_img = ski.util.img_as_float(img)
        self.imgs = {'original':float_img, 'curr':float_img.copy()}
        self.dims = np.array(img.shape[:2])
        return

    # ==========================================
    # general purpose functions
    # ==========================================
    
    def save(self, path, key='curr'):
        ski.io.imsave(path, self.imgs[key])
        return
    
    
    def plot(self, key='curr', ax=None, **kwargs):
        "Plot an image"
        if not ax:
            size = 10 * self.dims[::-1] / self.dims[0]
            fig, ax = plt.subplots(figsize=size)
            
        col = ax.imshow(self.imgs[key])
        
        if self.imgs[key].ndim == 2:
            plt.colorbar(col)
    
    def reset(self, key='original'):
        """
        Reset the working image.
        """
        self.imgs['curr'] = self.imgs[key]
        
        
    def rebase(self):
        "mask the working image the new base"
        self.imgs['original'] = self.imgs['curr'].copy()

        
    def set(self, array, key):
        "add a given array to the imgs"
        self.imgs[key] = array
        

    # ==========================================
    # morphing functions
    # ==========================================

    def normalise(self, key='curr', std=1.):
        "Take single channel image and normalise it to have mean=0, given std"
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        self.imgs[key] = std * (self.imgs['curr'] - self.imgs['curr'].mean()) / self.imgs['curr'].std()
        return self.imgs[key]
    
    
    def dilation(self, key='curr', size=3):
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        selem = np.ones([size, size])
        self.imgs[key] = ski.morphology.dilation(self.imgs['curr'], selem)
        return self.imgs[key]
    
    
    def erosion(self, key='curr', size=3):
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        selem = np.ones([size, size])
        self.imgs[key] = ski.morphology.erosion(self.imgs['curr'], selem)
        return self.imgs[key]
    
    
    def median(self, key='curr'):
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        self.imgs[key] = ski.filters.median(self.imgs['curr'])
        return self.imgs[key]
    
    
    # ==========================================
    # colorspace functions
    # ==========================================
    
    
    def grey_scale(self, key='curr'):
        self.imgs[key] = ski.color.rgb2gray(self.imgs['curr'])
        return self.imgs[key]
    
    
    def hsv(self, key='curr'):
        self.imgs[key] = ski.color.rgb2hsv(self.imgs['curr'])
        return self.imgs[key]
        
        
    def select_channel(self, channel, key='curr'):
        assert self.imgs['curr'].ndim ==3, 'curr must be a multi-channel image'
        self.imgs[key] = self.imgs['curr'][:, :, channel]
        return self.imgs[key]
    
        
    # ==========================================
    # image filters functions
    # ==========================================

        
    def gabor_filters(self, frequency, n_angs, key='curr'):
        
        self.grey_scale(key='grey')
        thetas = np.linspace(0, np.pi, n_angs)
        gabor_imgs = []
        
        for t in thetas:
            kernel = np.real(ski.filters.gabor_kernel(frequency, theta=t))
            gabor_imgs.append(ndi.convolve(self.imgs['grey'],
                                           kernel, mode='reflect'))    
        self.imgs[key] = np.dstack(gabor_imgs).mean(axis=-1)
        return self.imgs[key]
        
    
    def gauss(self, sigma=1, key='curr'):
        self.imgs[key] = ski.filters.gaussian(self.imgs['curr'], sigma=sigma,
                                             multichannel=True)
        return self.imgs[key]
    
    
    def sobel(self, key='curr'):
        sobs = [ski.filters.sobel(self.imgs['curr'][:, :, i])
                for i in range(3)]
        self.imgs[key] = np.dstack(sobs)
        return self.imgs[key]
        
        
    def scharr(self, key='curr'):
        schs = [ski.filters.scharr(self.imgs['curr'][:, :, i])
                for i in range(3)]
        self.imgs[key] = np.dstack(schs)
        return self.imgs[key]
            
        
    def laplace(self, size=3, key='curr'):
        self.imgs[key] = ski.filters.laplace(self.imgs['curr'], ksize=size)
        return self.imgs[key]
            
        
    # ==========================================
    # complext skimage functions
    # ==========================================    
        
        
    def lbp(self, radius=3, method='uniform', key='curr'):
        self.grey_scale(key='grey')
        self.imgs[key] = skf.local_binary_pattern(self.imgs['grey'],
                                                  radius*8,
                                                  radius,
                                                  method)
        return self.imgs[key]
        
        
    def hog(self, key='curr'):
        self.imgs[key] = ski.feature.hog(self.imgs['curr'], visualize=True)[1]
        return self.imgs[key]
    
    
    def canny(self, key='curr'):
        self.imgs[key] = ski.feature.canny(self.imgs['curr']).astype(float)
        return self.imgs[key]
    
    
    def prewitt(self, key='curr'):
        assert self.imgs['curr'].ndim == 2, 'curr image must be single channel'
        self.imgs[key] = ski.filters.prewitt(self.imgs['curr']).astype(float)
        return self.imgs[key]
    
        
if __name__ == '__main__':
    
    # examples
    IP_obj = Image_processor(path='/content/images/TX1_white_cropped.tif')
    
    # do a blured laplacian
    IP_obj.reset()
    IP_obj.gauss(sigma=1)
    IP_obj.gauss(sigma=1)
    IP_obj.laplace(size=3)
    IP_obj.grey_scale('curr')
    IP_obj.plot()
    
    # do an edge detection
    IP_obj.reset()
    IP_obj.grey_scale()
    IP_obj.prewitt()
    IP_obj.plot()
 