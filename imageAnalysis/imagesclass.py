# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:54:24 2022

@author: geert
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import trackpy as tp
from .calculatesann import calculateSANN
import os

#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class realspace:
    
    def __init__(self, img):
        """
        several operations on realspace images

        Parameters
        ----------
        img : 3d numpy array (M, N, 3) dtype = numpy.uint8
            Image in BGR format.

        Returns
        -------
        None.

        """
        self.img = img
        self.size = img.shape
        self.original = img.copy()
        self._isbinary = False
        if len(self.size) ==  2:
            self._isgrey = True
        else:
            self._isgrey = False
        print('initiazation of image complete')
        
    def reset(self):
        """
        Reset the image to the original input. Also resets some qualifiers

        Returns
        -------
        None.

        """
        self.img = self.original.copy()
        self.size = self.img.shape
        self._isbinary = False        
        if len(self.size) ==  2:
            self._isgrey = True
        else:
            self._isgrey = False
    
    def show(self, windowname = "rescaledimage", scale = 0.2):
        """
        Resizes and shows image
    
        Parameters
        ----------
        windowname : STR, optional
            The default is "rescaledimage".
        scale : FLOAT, optional
            The default is 0.2.
    
        Returns
        -------
        None, but opens a window and waits for a keypress.
    
        """
            
        newwidth = int(self.size[0] * scale)
        newheight = int(self.size[1] * scale)
        newshape = (newheight, newwidth)
        resized = cv2.resize(self.img, newshape, interpolation = cv2.INTER_AREA)
        
        cv2.imshow(windowname, resized) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def difference(self, img2, scaleself = -1):
        """
        Take out the background
        
        Parameters
        ----------
        background : 1d or 3d numpy array (M, N, (1 or 3)) dtype = float 
            The background of the image. Needs to have the same dimensions as
            self.img
        scaleself : FLOAT, optional
            How to rescale self.img. Use -1 to rescale by the mean value of
            self.img. The default is 0.2.

        Returns
        -------
        None.

        """
        
        if scaleself == -1:
            scaleself = 1/self.img.mean()
        img1 = self.img * scaleself
        temp = img1 - img2
        temp -= temp.min()
        temp *= 255/temp.max()
        
        self.img = temp.astype('uint8')
        
        

        
    def makegrayscale(self):
        """
        Make self.img grayscale using the cv2 COLOR_BGR2GRAY cvtColor method

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a grayscale image')
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = self.gray
        self.size = self.img.shape
        self._isgrey = True
        
    def takeblue(self):
        """
        Takes the blue channel from self.img and uses it as a greyscale image

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a grayscale image')
        self.blue = self.img[:,:,0]
        self.img = self.blue
        self.size = self.img.shape
        self._isgrey = True
    def takegreen(self):
        """
        Takes the green channel from self.img and uses it as a greyscale image

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a grayscale image')
        self.green = self.img[:,:,1]
        self.img = self.green
        self.size = self.img.shape
        self._isgrey = True
    def takered(self):
        """
        Takes the red channel from self.img and uses it as a greyscale image

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a grayscale image')
        self.red = self.img[:,:,2]
        self.img = self.red
        self.size = self.img.shape
        self._isgrey = True

    def discreteFourier(self):
        """
        Takes the greyscale image and perferoms discrete Fourier transform on it

        Returns
        -------
        fourtrans: Fourier instance
            an instance of the Fourier class with the Fourier transform of
            self.img as data

        """
        if not self._isgrey:
            raise ValueError('please first convert to greyscale image')
        rows, cols = self.size
        m = cv2.getOptimalDFTSize( rows )
        n = cv2.getOptimalDFTSize( cols )
        padded = cv2.copyMakeBorder(self.img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
        planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
        complexI = cv2.merge(planes)
        cv2.dft(complexI, complexI)
        fourtrans = Fourier(complexI)
        return(fourtrans)

    def convolve(self, kernel):
        """
        Convolves self.img using the given kernel.

        Parameters
        ----------
        kernel : 2D array (M, N)
            Kernel to be used in the convolution. M and N should both be odd.

        Returns
        -------
        None.
        """
        self.convimg = cv2.filter2D(self.img, -1, kernel)
        self.img = self.convimg

    def blur(self, blursize):
        """
        Blurs self.img using a sqaure kernel of the given size

        Parameters
        ----------
        blursize : INT
            Sidelength of the kernel

        Returns
        -------
        None.

        """
        self.blurimg = cv2.blur(self.img, (blursize, blursize))
        self.img = self.blurimg
        
    def locate(self, size, minMass = None, maxSize = None, dark = False):
        '''
        Determine the particle coordinates of the image using the trackpy
        tp.locate function, made using version 0.5.0
        http://soft-matter.github.io/trackpy/v0.5.0/generated/trackpy.locate.html#trackpy.locate

        Parameters
        ----------
        size : odd INT or TUPLE of odd INTs
            Passed into 'diameter' parameter of tp.locate;
            This may be a single number or a tuple giving the feature’s extent
            in each dimension, useful when the dimensions do not have equal
            resolution (e.g. confocal microscopy). The tuple order is the same
            as the image shape, conventionally (z, y, x) or (y, x). The
            number(s) must be odd integers. When in doubt, round up.
        minMass : FLOAT, optional
            Passed into 'minmass' parameter of tp.locate. The default is None.
            minmass:
            The minimum integrated brightness. This is a crucial parameter for
            eliminating spurious features. Recommended minimum values are 100
            for integer images and 1 for float images. Defaults to 0 (no filtering).
        maxSize : FLOAT, optional
            Passed into 'maxsize' parameter of tp.locate. The default is None.
            maxsize:
            maximum radius-of-gyration of brightness, default None
        dark : BOOL, optional
            Set to True if features are darker than background. The default is
            False.
            Note; the tp.locate option used to achieve this will be deprecated.

        Returns
        -------
        coords : instance of calculateSANN
            Detected particle coordinates in the calculateSANN class.

        '''
        if not self._isgrey:
            raise ValueError('please first convert to greyscale image')
        self.locdata = tp.locate(self.img, size, minmass = minMass, maxsize = maxSize, invert = dark)
        N = len(self.locdata)
        self.locations = np.zeros((N,2), dtype = 'float')
        for index, (_, loc) in enumerate(self.locdata.T.iteritems()):
            place = [dim for _, dim in loc.iloc[:2].iteritems()]
            self.locations[index] = place
        self.coords = calculateSANN(2, N, (self.size[1], self.size[0]), self.locations.T[[1,0]].T, diameters = [size])
        return(self.coords)
        
        
    def binarize(self, thresh = 'auto'):
        """
        Splits self.img in 2 groups based on a treshold. Alternatively the 
        treshhold can be determined automatically

        Parameters
        ----------
        thresh : INT or STR, optional
            The value for the threshold to use. If a string is given as input 
            a threshold is automatically chosen. The default is 'auto'

        Returns
        -------
        None.

        """
        if not self._isgrey:
            raise ValueError('please first convert to greyscale image')
        if type(thresh) == str:
            thresh, _ = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        _, self.binary = cv2.threshold(self.img, thresh, 255, cv2.THRESH_BINARY)
        self.img = self.binary
        self._isbinary = True
        
    

    def components(self):
        """
        Finds all separate chunks that have the same intensity value. Makes a 
        list of all (non-zero) chunks with alle coordinates belonging to the
        chunk.

        Returns
        -------
        None.

        """
        if not self._isbinary:
            raise ValueError('please first convert to binary')
        N, places = cv2.connectedComponents(self.img)
        N -= 1
        self.particlelist = [None]*N
        for i, goal in enumerate(self.particlelist):
            goal = (places == i + 1).nonzero()
            self.particlelist[i] = goal
            
    
    def cutter(self, xmin, xmax, ymin, ymax):
        """
        Cuts the image to a smaller size

        Parameters
        ----------
        xmin : INT
            Leftmost part of the new image (inclusive).
        xmax : INT
            Rightmost part of the new image (exclusive).
        ymin : INT
            Topmost part of the new image (inclusive).
        ymax : INT
            Bottommost part of the new image (exclusive).

        Returns
        -------
        None.

        """
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > self.size[1]: xmax = self.size[1] 
        if ymax > self.size[0]: ymax = self.size[0]
        
        self.img = self.img[ymin:ymax,xmin:xmax]
        self.size = self.img.shape
        
    
    
    def save(self, filename):
        """
        Saves current image under filename

        Parameters
        ----------
        filename : STR
            Filename under which image will be saved. Needs an extension.

        Returns
        -------
        None.

        """
        cv2.imwrite(filename, self.img)
    
    
class realspacefromfile(realspace):
    def __init__(self, filename):
        """
        several operations on realspace images

        Parameters
        ----------
        img : STR
            Image filename (and location if in separate folder) including extension.

        Returns
        -------
        None.

        """
        self.filename = filename
        self.img = cv2.imread(filename)
        realspace.__init__(self,self.img)
        
        
        
        
        



        
        
class Fourier:
    def __init__(self, data):
        """
        Several operations on a (discrete) Fourier transform

        Parameters
        ----------
        data : numpy array (N, M, 2) dtype = numpy.float32
            Original Fourier transform. Real part in [:,:,0], imaginary part in [:,:,1].

        Returns
        -------
        None.

        """
        self.data = data
        self.original = data.copy()
        self.size = data.shape
        self.mask = np.ones(self.size)
        
    def reset(self):
        """
        Reset self.data to the original input data

        Returns
        -------
        None.

        """
        self.data = self.original.copy()
        self.size = self.data.shape
        
    def show(self,  translate = True, logarithm = True, windowname = "Fourier Transform", scale = 0.2):
        """
        Rescales and shows the magnitude of the Fourier transform
    
        Parameters
        ----------
        complexI : 3d numpy array (>=n, >=m, 2)
            an array with the real and complex parts of the Fourier Transform of 
            an image
        translate : BOOLEAN, optional
            Translate to the middle of the image? The default is True.
        logarithm : BOOLEAN, optional
            Rescale values logarithmically? The default is True.
        windowname : STR, optional
            The default is "Fourier Transform".
        scale : FLOAT, optional
            Add value for rescaling. The default is 0.2.
    
        Returns
        -------
        None, but shows Fourier transform
    
        """
        real = np.array(self.data[:,:,0])
        compl = np.array(self.data[:,:,1])
        magnitude = np.sqrt(real**2 + compl**2)
        shape = magnitude.shape
    
        if translate:
            magnitude = self.reorient(magnitude)

        if logarithm:
            templ = np.ones(shape, dtype = magnitude.dtype)
            logmag = np.log(templ + magnitude)
            magnitude = logmag
        
        normalization = magnitude.astype('uint8')
        cv2.normalize(normalization, normalization, 0, 255, cv2.NORM_MINMAX)
            
        newwidth = int(shape[0] * scale)
        newheight = int(shape[1] * scale)
        newshape = (newheight, newwidth)
        resized = cv2.resize(normalization, newshape, interpolation = cv2.INTER_AREA)
        
        cv2.imshow(windowname, resized) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #return(magnitude, normalization, resized)
    
    
    def save(self, filename, translate = True, logarithm = True):
        """
        Rescales and saves the magnitude of the Fourier transform
    
        Parameters
        ----------
        complexI : 3d numpy array (>=n, >=m, 2)
            an array with the real and complex parts of the Fourier Transform of 
            an image
        translate : BOOLEAN, optional
            Translate to the middle of the image? The default is True.
        logarithm : BOOLEAN, optional
            Rescale values logarithmically? The default is True.
        windowname : STR, optional
            The default is "Fourier Transform".
    
        Returns
        -------
        None, but shows Fourier transform
    
        """
        real = np.array(self.data[:,:,0])
        compl = np.array(self.data[:,:,1])
        magnitude = np.sqrt(real**2 + compl**2)
        shape = magnitude.shape
    
        if translate:
            magnitude = self.reorient(magnitude)

        if logarithm:
            templ = np.ones(shape, dtype = magnitude.dtype)
            logmag = np.log(templ + magnitude)
            magnitude = logmag
        
        normalization = magnitude.astype('uint8')
        cv2.normalize(normalization, normalization, 0, 255, cv2.NORM_MINMAX)
            
        cv2.imwrite(filename, normalization)
    
    def radialprofile(self, center = None, translate = True, logarithm = True):
        """
        Calculate the radial average of the magnitude of the transform

        Parameters
        ----------
        center : Tuple of 2 ints, optional
            Center point of the circle. Defaults to origin of the Fourierspace.
            The default is None.
        translate : BOOLEAN, optional
            Translate to the middle of the image? The default is True.
        logarithm : BOOLEAN, optional
            Rescale values logarithmically? The default is True.

        Returns
        -------
        radialprofile : numpy array
            Array with radially averaged pixel values.

        """
        
        if center is None: # use the middle of the image
            center = (int(self.size[1]/2), int(self.size[0]/2))
            
        y, x = np.indices(self.size)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        real = np.array(self.data[:,:,0])
        compl = np.array(self.data[:,:,1])
        magnitude = np.sqrt(real**2 + compl**2)
        
        if translate:
            magnitude = self.reorient(magnitude)
        
        if logarithm:
            logmag = np.log(magnitude + 1)
            magnitude = logmag
        
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile 
    
        

    def inverseDiscreteFourier(self):
        """
        Do the inverse Fourier transform on a (already size-wise optimized) image
    
        Returns
        -------
        newimg: realspace instance
            Realspace instance that is the inverse Fourier transformed of self.data.
    
        """
        # rows, cols = self.shape[:2]
        # m = cv2.getOptimalDFTSize( rows )
        # n = cv2.getOptimalDFTSize( cols )
        # padded0 = cv2.copyMakeBorder(self.data[:,:,0], 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # padded1 = cv2.copyMakeBorder(self.data[:,:,1], 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        image = self.data.copy()
        
        cv2.dft(image, image, flags = (cv2.DFT_INVERSE) )
        
        real = np.array(image[:,:,0])
        compl = np.array(image[:,:,1])
        normalizor = 1
        for dim in self.size[:2]: normalizor *= dim
        magnitude = np.sqrt(real**2 + compl**2)/normalizor
        
        img = (np.round(magnitude)).astype('uint8')
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        
        newimg = realspace(img)
        
        return(newimg)
    

    def annulusmask(self, small, big, center = None, inversemask = False):
        """
        Make an anulus mask

        Parameters
        ----------
        small : int
            Radius of the inner edge.
        big : int
            Radius of the outer edge.
        center : Tuple of 2 ints, optional
            Center point of the circle. Defaults to origin of the Fourierspace.
            The default is None.

        Returns
        -------
        None.

        """
        if center is None: # use the middle of the image
            center = (int(self.size[1]/2), int(self.size[0]/2))
        
        self.circlemask(center, small)
        innermask = self.mask
        self.circlemask(center, big, False)
        outermask = self.mask
        

        self.mask =  innermask * outermask
        if inversemask:
            self.mask = 1 - self.mask


    def circlemask(self, center=None, radius=None, insideblack = True):
        """
        Make a circular mask.
    
        Parameters
        ----------
        center : tuple of 2 ints, optional
            center point of the circle. Defaults to origin of the Fourierspace.
            The default is None.
        radius : int, optional
            radius of the circle. The default is None.
        insideblack: Boolean, optional
            is the inside (True) or the outside (False) black. The default is True
    
        Returns
        -------
        None.
        
        """
    
        if center is None: # use the middle of the image
            center = (int(self.size[1]/2), int(self.size[0]/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], self.size[0]-center[1], self.size[1]-center[0])
        
        
        Y, X = np.ogrid[:self.size[0], :self.size[1]]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        if insideblack:
            temp = dist_from_center >= radius
        else:
            temp = dist_from_center <= radius
        
        self.mask = self.reorient(temp)
        
        
    def usemask(self):
        """
        Use the defined mask on the dataset

        Returns
        -------
        None.

        """
        temp0 =  self.data[:, :, 0] * self.mask
        temp1 =  self.data[:, :, 1] * self.mask
        self.data[:, :, 0] = temp0
        self.data[:, :, 1] = temp1
        
    

    @staticmethod
    def reorient(dataset):
        shape = dataset.shape
        temp = np.zeros(shape, dtype = dataset.dtype)
        
        center = [int(dim/2) for dim in shape[:2]]
        
        temp[:-center[0], :-center[1]] = dataset[center[0]:, center[1]:]
        temp[:-center[0], -center[1]:] = dataset[center[0]:, :center[1]]
        temp[-center[0]:, :-center[1]] = dataset[:center[0], center[1]:]
        temp[-center[0]:, -center[1]:] = dataset[:center[0], :center[1]]
        
        return(temp)
    
    
    
    
class imageframes:
    def __init__(self, path, namestart, skipfiles = 0):
        self.imgs = []
        self.filenames = []
        for file in os.listdir(path):
            if file.startswith(namestart):
                print(file[10:16])
                foo = cv2.imread(path + file)
                #print(type(foo))
                foo = realspace(foo)
                self.imgs.append(foo)
                self.filenames.append(path + file)
        self._isgrey = False
                
        
    def makegrayscale(self):
        if self._isgrey:
            raise ValueError('please first convert to greyscale image')
        for img in self.imgs:
            img.makegrayscale()
        self._isgrey = True
            
    def difference(self, src2):
        myback = cv2.imread(src2)
        if self._isgrey:
            myback = cv2.cvtColor(myback, cv2.COLOR_BGR2GRAY)
        myback = myback.astype('float')
        myback /= myback.mean()
        for img in self.imgs:
            img.difference(myback)
    
    
    def batchlocate(self, size, searchrange, dark = False, mem = 3):
        print('load locations')
        frames = [foo.img for foo in self.imgs]
        self.alllocs = tp.batch(frames, size, invert = dark)
        print('link particles')
        self.connected = tp.link(self.alllocs, searchrange, memory = mem)
        print('compute and remove drift')
        self.drift = tp.compute_drift(self.connected)
        self.trajectories = tp.subtract_drift(self.connected.copy(), self.drift)
        
    def showtraject(self):
        print('showing trajectories, might be somewhat slow')
        tp.plot_traj(self.trajectories)

    def msd(self, umperpx, fps, minimumlength = -1, show = True):
        if minimumlength < 0:
            minimumlength = len(self.imgs)
        temp = tp.filter_stubs(self.trajectories, minimumlength)
        self.ensemblemsd = tp.emsd(temp, umperpx, fps)
        if show:
            fig, ax = plt.subplots()
            ax.plot(self.ensemblemsd.index, self.ensemblemsd, 'o')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ ($\mu$m$^2$)', xlabel='lag time $t$ ($s$)')
    
    





