# -*- coding: utf-8 -*-
"""
    Copyright (C) 2023  Geert Schulpen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import numpy as np

import matplotlib.pyplot as plt
import cv2
import trackpy as tp

from imageanalysis.calculatesann import CalculateSANN


class Realspace:
    
    def __init__(self, img):
        """
        Several operations on realspace images.

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
        if len(self.size) == 2:
            self._isgrey = True
        else:
            self._isgrey = False
        print('initiazation of image complete')
        
    def reset(self):
        """
        Reset the image to the original input. Also reset some qualifiers.

        Returns
        -------
        None.

        """
        self.img = self.original.copy()
        self.size = self.img.shape
        self._isbinary = False        
        if len(self.size) == 2:
            self._isgrey = True
        else:
            self._isgrey = False
    
    def show(self, windowname="rescaledimage", scale=0.2):
        """
        Resize and show image.
    
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
        
    def difference(self, background, scaleself=-1):
        """
        Determine the difference between self.img and a provided image.
        
        Parameters
        ----------
        background : 2d or 3d numpy array (M, N, (1 or 3)) dtype = float 
            The background of the image. Needs to have the same dimensions as
            self.img
        scaleself : FLOAT, optional
            How to rescale self.img. Use -1 to rescale by the mean value of
            self.img. The default is -1

        Returns
        -------
        None.

        """
        
        if scaleself == -1:
            scaleself = 1/self.img.mean()
        img1 = self.img * scaleself
        temp = img1 - background
        temp -= temp.min()
        temp *= 255/temp.max()
        
        self.img = temp.astype('uint8')
        
        

        
    def makegrayscale(self):
        """
        Make self.img grayscale using the cv2 COLOR_BGR2GRAY cvtColor method.

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a'
                             ' grayscale image')
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = self.gray
        self.size = self.img.shape
        self._isgrey = True
        
    def takeblue(self):
        """
        Take the blue channel from self.img and use it as a greyscale image.

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a'
                             ' grayscale image')
        self.blue = self.img[:,:,0]
        self.img = self.blue
        self.size = self.img.shape
        self._isgrey = True
        
    def takegreen(self):
        """
        Take the green channel from self.img and use it as a greyscale image.

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a'
                             ' grayscale image')
        self.green = self.img[:,:,1]
        self.img = self.green
        self.size = self.img.shape
        self._isgrey = True
        
    def takered(self):
        """
        Take the red channel from self.img and use it as a greyscale image.

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Please do not call a grayscale maker on a'
                             ' grayscale image')
        self.red = self.img[:,:,2]
        self.img = self.red
        self.size = self.img.shape
        self._isgrey = True

    def discretefourier(self):
        """
        Perform discrete Fourier transform on a grayscale image.

        Returns
        -------
        fourtrans: Fourier instance
            an instance of the Fourier class with the Fourier transform of
            self.img as data

        """
        if not self._isgrey:
            raise ValueError('please first convert to greyscale image')
        rows, cols = self.size
        m = cv2.getOptimalDFTSize(rows)
        n = cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(self.img, 0, m - rows, 0, n - cols,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])

        planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
        complexI = cv2.merge(planes)
        cv2.dft(complexI, complexI)
        fourtrans = Fourier(complexI)
        return fourtrans

    def convolve(self, kernel):
        """
        Convolve self.img with the given kernel.

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
        Blur self.img using a sqaure kernel.

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
        
    def locate(self, size, minMass=None, maxSize=None, dark=False):
        """
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
            for integer images and 1 for float images. Defaults to 0 (no 
            filtering).
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
        coords : instance of CalculateSANN
            Detected particle coordinates in the CalculateSANN class.

        """
        if not self._isgrey:
            raise ValueError('please first convert to greyscale image')
        self.locdata = tp.locate(self.img, size, minmass=minMass,
                                 maxsize=maxSize, invert=dark)
        N = len(self.locdata)
        self.locations = np.zeros((N,2), dtype = 'float')
        for index, (_, loc) in enumerate(self.locdata.T.items()):
            place = [dim for _, dim in loc.iloc[:2].items()]
            self.locations[index] = place
        self.coords = CalculateSANN(2, N, (self.size[1], self.size[0]),
                                    self.locations.T[[1,0]].T, diameters=[size])
        return self.coords
        
        
    def binarize(self, thresh='auto'):
        """
        Split self.img in 2 groups based on a treshold.

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
        if isinstance(thresh, str):
            thresh, _ = cv2.threshold(self.img, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, self.binary = cv2.threshold(self.img, thresh, 255, cv2.THRESH_BINARY)
        self.img = self.binary
        self._isbinary = True
        
    

    def components(self):
        """
        Find all separate chunks that have the same intensity value. Makes a 
        list of all (non-zero) chunks with all coordinates belonging to the
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
            goal = (places == i+1).nonzero()
            self.particlelist[i] = goal
            
    
    def radialprofile(self, center=None):
        """
        Calculate the radial average of pixel magnitudes.

        Parameters
        ----------
        center : Tuple of 2 ints, optional
            Center point of the circle. Defaults to origin of the Fourierspace.
            The default is None.

        Returns
        -------
        radialprofile : numpy array
            Array with radially averaged pixel values.

        """
        
        if not self._isgrey:
            raise ValueError('please first convert to greyscale image')
        
        if center is None: # use the middle of the image
            center = (int(self.size[1]/2), int(self.size[0]/2))
            
        y, x = np.indices(self.size)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        
        tbin = np.bincount(r.ravel(), self.img.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile 
 
    
    def cutter(self, xmin, xmax, ymin, ymax):
        """
        Cut the image to a smaller size.

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
        
        self.img = self.img[ymin:ymax, xmin:xmax]
        self.size = self.img.shape
        
    
    
    def save(self, filename):
        """
        Save the current image.

        Parameters
        ----------
        filename : STR
            Filename under which image will be saved. Needs an extension.

        Returns
        -------
        None.

        """
        cv2.imwrite(filename, self.img)
    
    
class RealspaceFromFile(Realspace):
    def __init__(self, filename):
        """
        Several operations on realspace images, using a file  as input.

        Parameters
        ----------
        img : STR
            Image filename (and location if in separate folder) including
            extension.

        Returns
        -------
        None.

        """
        self.filename = filename
        self.img = cv2.imread(filename)
        Realspace.__init__(self, self.img)
        
        
        
        
        



        
        
class Fourier:
    def __init__(self, data):
        """
        Several operations on a (discrete) Fourier transform

        Parameters
        ----------
        data : numpy array (N, M, 2) dtype = numpy.float32
            Original Fourier transform. Real part in [:,:,0], imaginary part
            in [:,:,1].

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
        
    def show(self,  translate=True, logarithm=True,
             windowname="Fourier Transform", scale=0.2):
        """
        Rescale and show the magnitude of the Fourier transform
    
        Parameters
        ----------
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
            templ = np.ones(shape, dtype=magnitude.dtype)
            logmag = np.log(templ + magnitude)
            magnitude = logmag
        
        normalization = magnitude.astype('uint8')
        cv2.normalize(normalization, normalization, 0, 255, cv2.NORM_MINMAX)
            
        newwidth = int(shape[0] * scale)
        newheight = int(shape[1] * scale)
        newshape = (newheight, newwidth)
        resized = cv2.resize(normalization, newshape,
                             interpolation=cv2.INTER_AREA)
        
        cv2.imshow(windowname, resized) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    def save(self, filename, translate=True, logarithm=True):
        """
        Rescale and save the magnitude of the Fourier transform
    
        Parameters
        ----------
        complexI : 3d numpy array (>=n, >=m, 2)
            an array with the real and complex parts of the Fourier Transform
            of an image
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
            templ = np.ones(shape, dtype=magnitude.dtype)
            logmag = np.log(templ + magnitude)
            magnitude = logmag
        
        normalization = magnitude.astype('uint8')
        cv2.normalize(normalization, normalization, 0, 255, cv2.NORM_MINMAX)
            
        cv2.imwrite(filename, normalization)
        
    
    def radialprofile(self, center=None, translate=True, logarithm=True):
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
        
        real = np.array(self.data[:,:,0])
        compl = np.array(self.data[:,:,1])
        magnitude = np.sqrt(real**2 + compl**2)
        
        if translate:
            magnitude = self.reorient(magnitude)
        
        if logarithm:
            logmag = np.log(magnitude + 1)
            magnitude = logmag
            
        y, x = np.indices(magnitude.shape)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile 
    
        

    def inversediscretefourier(self):
        """
        Do the inverse Fourier transform on a (already size-wise optimized) image
    
        Returns
        -------
        newimg: Realspace instance
            Realspace instance that is the inverse Fourier transform of self.data.
    
        """
        
        image = self.data.copy()
        
        cv2.dft(image, image, flags=(cv2.DFT_INVERSE))
        
        real = np.array(image[:,:,0])
        compl = np.array(image[:,:,1])
        normalizor = 1
        for dim in self.size[:2]: normalizor *= dim
        magnitude = np.sqrt(real**2 + compl**2)/normalizor
        
        img = (np.round(magnitude)).astype('uint8')
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        
        newimg = Realspace(img)
        
        return newimg
    

    def annulusmask(self, small, big, center=None, inversemask=False):
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


    def circlemask(self, center=None, radius=None, insideblack=True):
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
            is the inside (True) or the outside (False) black.
            The default is True
    
        Returns
        -------
        None.
        
        """
    
        if center is None:
            center = (int(self.size[1]/2), int(self.size[0]/2))
        if radius is None:
            radius = min(center[0], center[1], self.size[0]-center[1],
                         self.size[1]-center[0])
        
        
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
        """
        Translate the first 2 axes of an array such that the [0,0] component
        becomes the middle component

        Parameters
        ----------
        dataset : numpy array (2 or more dimensions)
            Array to be translated.

        Returns
        -------
        temp : numpy array (same size as input)
            Array translated such that the [0,0] entry from the input is the 
            middle entry in the output

        """
        shape = dataset.shape
        temp = np.zeros(shape, dtype = dataset.dtype)
        
        center = [int(dim/2) for dim in shape[:2]]
        
        temp[:-center[0], :-center[1]] = dataset[center[0]:, center[1]:]
        temp[:-center[0], -center[1]:] = dataset[center[0]:, :center[1]]
        temp[-center[0]:, :-center[1]] = dataset[:center[0], center[1]:]
        temp[-center[0]:, -center[1]:] = dataset[:center[0], :center[1]]
        
        return temp
    
    
    
    
class ImageFrames:
    def __init__(self, path, namestart):
        """
        Several operations on a collection of images.
        
        The images should all be saved in the same folder. If there are other 
        files in the folder that should not be read the filenames that should
        be imported need to start with a common pattern, that is not matched by
        any of the other files.

        Parameters
        ----------
        path : STR
            Path to the folder in which the images are stored.
        namestart : STR
            The common pattern that the filenames that should be imported start
            with.

        Returns
        -------
        None.

        """
        self.imgs = []
        self.filenames = []
        for file in os.listdir(path):
            if file.startswith(namestart):
                #print(file[10:16])
                foo = cv2.imread(path + file)
                #print(type(foo))
                foo = Realspace(foo)
                self.imgs.append(foo)
                self.filenames.append(path + file)
        self._isgrey = False
                
        
    def makegrayscale(self):
        """
        Convert all loaded images to graysale.

        Raises
        ------
        ValueError
            If the images are grayscale already they cannot be reentered into
            this function.

        Returns
        -------
        None.

        """
        if self._isgrey:
            raise ValueError('Make sure to not input a grayscale image into'
                             ' a grayscale maker')
        for img in self.imgs:
            img.makegrayscale()
        self._isgrey = True
            
    def difference(self, src2):
        """
        Take the difference between each image and a commmon background.

        Parameters
        ----------
        src2 : STR
            Path, filename and extension of the image to be used as background.

        Returns
        -------
        None.

        """
        myback = cv2.imread(src2)
        if self._isgrey:
            myback = cv2.cvtColor(myback, cv2.COLOR_BGR2GRAY)
        myback = myback.astype('float')
        myback /= myback.mean()
        for img in self.imgs:
            img.difference(myback)
    
    
    def batchlocate(self, size, searchrange, dark=False, mem=3):
        """
        Locate and identify particles.
        
        Using the trackpy functions. Uses trackpy version 0.5.0
        http://soft-matter.github.io/trackpy/v0.5.0
        
        Parameters
        ----------
        size : odd INT or TUPLE of odd INTs
            Passed into 'diameter' parameter of tp.batch;
            This may be a single number or a tuple giving the feature’s extent
            in each dimension, useful when the dimensions do not have equal
            resolution (e.g. confocal microscopy). The tuple order is the same
            as the image shape, conventionally (z, y, x) or (y, x). The
            number(s) must be odd integers. When in doubt, round up.
        searchrange : INT or TUPLE of INTs
            The maximum distance features can move between frames.
        dark : BOOL, optional
            Set to True if features are darker than background. The default is
            False.
            Note; the tp.locate option used to achieve this will be deprecated.
        mem : TYPE, optional
            The maximum number of frames during which a feature can vanish,
            then reappear nearby, and be considered the same particle. The
            default is 3.

        Returns
        -------
        None.

        """
        
        print('load locations')
        frames = [foo.img for foo in self.imgs]
        self.alllocs = tp.batch(frames, size, invert=dark)
        print('link particles')
        self.connected = tp.link(self.alllocs, searchrange, memory=mem)
        print('compute and remove drift')
        self.drift = tp.compute_drift(self.connected)
        self.trajectories = tp.subtract_drift(self.connected.copy(), self.drift)
        
    def showtraject(self):
        """
        Show the trajectories of all the particles. Note; this is slow for
        images with many particles.

        Returns
        -------
        None.

        """
        print('showing trajectories, might be somewhat slow')
        tp.plot_traj(self.trajectories)

    def msd(self, umperpx, fps, minimumlength=-1, show=True):
        """
        Determine the mean squared displacement of the collective.

        Parameters
        ----------
        umperpx : float
            Scale of the image.
        fps : int
            Frames per second.
        minimumlength : int, optional
            minimum length a trajectory should be to include it in averaging.
            The default is -1, which includes only trajectories of particles
            that are present in every frame.
        show : BOOL, optional
            Wether to show a graph of the result. The default is True.

        Returns
        -------
        None.

        """
        if minimumlength < 0:
            minimumlength = len(self.imgs)
        temp = tp.filter_stubs(self.trajectories, minimumlength)
        self.ensemblemsd = tp.emsd(temp, umperpx, fps)
        if show:
            fig, ax = plt.subplots()
            ax.plot(self.ensemblemsd.index, self.ensemblemsd, 'o')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ ($\mu$m$^2$)',
                   xlabel='lag time $t$ ($s$)')
    
    





