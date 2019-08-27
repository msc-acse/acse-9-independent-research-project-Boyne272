import cv2
import numpy as np
import matplotlib.pyplot as plt

class preprocessing():
    
    def __init__(self, path, **kwargs):
        
        # load the image
        self._orig_path = path
        self.img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.log = 'Orig Img'
        
        # set defualt parameters and user choices 
        self.params = {'figsize':[22, 22], 'sig_col':75, 'sig_space':75,
                       'kernal_guass':(5, 5), 'kernal_median':5,
                       'kernal_bilateral':5, 'erode_iterations':1,
                       'kernal_erode':np.ones([5,5], dtype=np.uint8),
                       'dilate_iterations':1,
                       'kernal_dilate':np.ones([5,5], dtype=np.uint8)}
        self.set_param(**kwargs)
        
        
    def set_param(self, **kwargs):
        # set any custom params
        for key, val in kwargs.items():
            if key in self.params.keys():
                self.params[key] = val
                print(key, ' set to ', val)
        
        
    def save_as_img(self, path):
        cv2.imwrite(path, self.img)
        
        
    def save_as_array(self, new_path):
        np.save(path, self.img)
        
        
    def normalise(self):
        self.img = cv2.normalize(self.img, None, alpha=0, beta=1,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        
    def show(self):
        plt.figure(figsize=self.params['figsize'])
        plt.imshow(self.img)
        plt.title(self.log)
        
        
    def change_color_space(self, flag=cv2.COLOR_BGR2HSV):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.log += ', C Space Transform'
        
        
    def apply(self, *args):
        
        # options in order
        for arg in args:
            if 'guass' == arg:
                k = self.params['kernal_guass']
                self.img = cv2.GaussianBlur(self.img, k, 0)
                self.log += ', guass (' + str(k) + ')'

            if 'median' == arg:
                k = self.params['kernal_median']
                self.img = cv2.medianBlur(self.img, k)
                self.log += ', median (' + str(k) + ')'

            if 'bilateral' == arg:
                k = self.params['kernal_bilateral']
                self.img = cv2.bilateralFilter(self.img, k, self.params['sig_col'],
                                               self.params['sig_space'])
                self.log += ', bilateral (' + str(k) + ')'

            if 'erode' == arg:
                self.img = cv2.erode(self.img, self.params['kernal_erode'],
                                     iterations=self.params['erode_iterations'])
                self.log += ', erode'

            if 'dilate' == arg:
                self.img = cv2.dilate(self.img, self.params['kernal_dilate'],
                                      iterations=self.params['dilate_iterations'])
                self.log += ', dilate'