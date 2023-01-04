# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:30:18 2022

@author: geert
"""

import numpy as np

class coordinateFileManipulation:
    def __init__(self, filename):
        '''
        Setup a class to work with particle locations

        Parameters
        ----------
        filename : STR
            Path & filename of file to read or write.

        Returns
        -------
        None.

        '''
        print('init')
        self.filename = filename
        self.N = 0
        self.xrange = 0.0
        self.yrange = 0.0
        self.zrange = 0.0
        self.box = (self.xrange, self.yrange, self.zrange)
        self.particlelist = []
        self.diameterlist = [1]
        self.colorlist = ['']
        
    def readFile(self):
        '''
        Read the file in self.filename. If needed changes the particle
        coordinates to all be positive numbers.

        Returns
        -------
        None.

        '''
        self.data1 = [i.strip().split() for i in open(self.filename).readlines()]
        self.N = int(self.data1[0][0])
        xmin = np.array(self.data1[1]).astype('float')[0]
        ymin = np.array(self.data1[2]).astype('float')[0]
        zmin = np.array(self.data1[3]).astype('float')[0]
        self.xrange = np.array(self.data1[1]).astype('float')[1]
        self.yrange = np.array(self.data1[2]).astype('float')[1]
        self.zrange = np.array(self.data1[3]).astype('float')[1]
        self.box = (self.xrange - xmin, self.yrange - ymin, self.zrange - zmin)
        dataset = np.array(self.data1[4:]).astype('float')
        self.particlelist = dataset[:,:3]
        self.particlelist[:,0] -= xmin
        self.particlelist[:,1] -= ymin
        self.particlelist[:,2] -= zmin
        self.diameterlist = dataset[:,3]
        if len(dataset[1]) > 4:
            self.colorlist = dataset[:,4:]
        else:
            self.colorlist = [''] * self.N
        
        
    def inputinfo(self, N, box, particlelist, diameterlist = [1], colorlist = ['']):
        '''
        Alternative way to input particle information

        Parameters
        ----------
        N : INT
            Total number of particles.
        box : TUPLE of M FLOATs
            Region in which particles might be. M is equal to the number of
            dimensions in the dataset
        particlelist : NP.ARRAY of N by M
            Particle locations.

        diameterlist : LIST of FLOATs (length 1 or N), optional
            Diameters of the particles. If length of is 1 it is
            extended to length N. The default is [1].
        colorlist : LIST (length 1 or N), optional
            Colors/additional data of the particles. If length is 1 it is
            extended to length N. The default is [''].

        Returns
        -------
        None.

        '''
        self.N = N
        self.xrange = box[0]
        self.yrange = box[1]
        try:
            self.zrange = box[2]
        except:
            self.zrange = 0
        self.box = (self.xrange, self.yrange, self.zrange)
        if particlelist.shape[1] == 2:
            z = np.zeros(particlelist.shape[0])
            particlelist = np.concatenate((particlelist, z[:,np.newaxis]), axis = 1)
        self.particlelist = particlelist
        if len(diameterlist) == 1:
            diameterlist = diameterlist * N
        self.diameterlist = diameterlist
        if len(colorlist) == 1:
            colorlist = colorlist * N
        self.colorlist = colorlist
        
        
        
    def writeFile(self, **kwargs):
        """
        Write the file to self.filename. All parameters are optional and can
        only be called by name, not by position. If parameters are not given
        the classvariable is used

        Parameters
        ----------
        box : TUPLE of M FLOATs
            Region in which particles might be. M is equal to the number of
            dimensions in the dataset
        particlelist : NP.ARRAY of shape (N,M)
            Particle locations, N is the number of particles (automatically
            determined)
        diameters : LIST of FLOATs (length 1 or N)
            Diameters of the particles. If length of is 1 it is
            extended to length N.
        colors : LIST (length 1 or N)
            Colors/additional data of the particles. If length is 1 it is
            extended to length N. Typically either a single integer or 3 rgb
            values, inputted as floats.

        Returns
        -------
        None.
        
        Writes
        ------
        File to self.filename with N + 4 lines, structured as follows. This
        file can be used directly by the 'viscol' code developed by Michiel
        Hermes, of which an in-browser version exists: 
            https://webspace.science.uu.nl/~herme107/viscol/#about
        
        Number_of_particles
        0    xbox
        0    ybox
        0    zbox
        x1    y1    z1    diam1    color1
        x2    y2    z2    diam2    color2
        ....

        """
        print('write')
        particlelist = kwargs.pop('particlelist', self.particlelist)
        if particlelist.shape[1] == 2:
            z = np.zeros(particlelist.shape[0])
            particlelist = np.concatenate((particlelist, z[:,np.newaxis]), axis = 1)
        box = kwargs.pop('box', self.box)
        if len(box) == 2:
            box = (box[0], box[1], 0.0)
        N = particlelist.shape[0]
        
        diameters = kwargs.pop('diameters', self.diameterlist)
        if len(diameters) == 1:
            diameters = diameters * N
        
        colors = kwargs.pop('colors', self.colorlist)
        if len(colors) == 1:
            colors = colors * N
        
        with open(self.filename, 'w') as f:
            f.write(str(N) + '\n')
            for dim in box:
                f.write("0.000000 \t " + str(dim) + "\n")
            for coords, diameter, color in zip(particlelist, diameters, colors):
                f.write(str(coords[0]) + " \t " + str(coords[1]) + " \t " + str(coords[2]))
                f.write(" \t " + str(diameter) + " \t " + str(color) + " \n")
                
                
                
                
    def removeParticles(self, requirement):
        """
        Remove particles that have 'requirement' as their color value.

        Parameters
        ----------
        requirement : TYPE (typically INT)
            Colorvalue to remove.

        Returns
        -------
        None, but changes N, particlelist, diameterlist and colorlist to
        exclude the particles

        """
        
        colors = self.colorlist.copy()
        colors = (colors == requirement)
        removal = np.sum(colors)
        N = self.N - removal
        particlelist = np.zeros((N,3), dtype = 'float')
        diameterlist = np.zeros(N, dtype = 'float')
        colorlist = np.zeros(N, dtype = 'int')
        
        pidout = 0
        pidin = 0
        while pidin < N and pidout < self.N:
            if pidin % 1000 == 0:
                print(f'we\'re working on {pidout} of {self.N} particles')
            if colors[pidout] == 1:
                pidout += 1
            else:
                particlelist[pidin] = self.particlelist[pidout]
                diameterlist[pidin] = self.diameterlist[pidout]
                colorlist[pidin] = self.colorlist[pidout]
                pidin += 1
                pidout += 1
        
        self.N = N
        self.particlelist = particlelist
        self.diameterlist = diameterlist
        self.colorlist = colorlist
        
        
    def rearrangecolumn(self, axis):
        """
        Invert the coordinates of one column. Calculates xnew = xbox - x

        Parameters
        ----------
        axis : INT (0, 1 or 2)
            Which axis to invvert. 0 for x axis, 1 for y and 2 for z axis.

        Returns
        -------
        None.

        """
        
        self.particlelist[:,axis] = self.box[axis] - self.particlelist[:,axis]
        
        
        
        
