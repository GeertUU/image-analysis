# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:22:42 2022

@author: geert
"""



from coordinatefilemaniputlations import coordinateFileManipulation
import numpy as np
from operator import itemgetter

from scipy.spatial import KDTree


    


class calculateSANN:
    
    cbrt2 = np.cbrt(2)
    root2 = np.sqrt(2)
    sinpi8 = np.sin(np.pi/8)
    cospi8 = np.cos(np.pi/8)
    
    amsin = np.arccos(0.5 * (1.0 - sinpi8));
    apsin = np.arccos(0.5 * (1.0 + sinpi8));
    amcos = np.arccos(0.5 * (1.0 - cospi8));
    apcos = np.arccos(0.5 * (1.0 + cospi8));
    msin = (1.0 - root2 - 2.0 * (cospi8 - sinpi8)) * amsin;
    psin = (1.0 - root2 + 2.0 * (cospi8 - sinpi8)) * apsin;
    mcos = (1.0 + root2 + 2.0 * (cospi8 + sinpi8)) * amcos;
    pcos = (1.0 + root2 - 2.0 * (cospi8 + sinpi8)) * apcos;
    pre0 = 0.25 * (mcos + msin + pcos + psin);
    
    msin = ( 2.0 * root2 + 9.0 * cospi8 - sinpi8) * amsin;
    psin = ( 2.0 * root2 - 9.0 * cospi8 + sinpi8) * apsin;
    mcos = (-2.0 * root2 - cospi8 - 9.0 * sinpi8) * amcos;
    pcos = (-2.0 * root2 + cospi8 + 9.0 * sinpi8) * apcos;
    pre1 = mcos + msin + pcos + psin;
    
    msin = ( -root2 - 12.0 * cospi8) * amsin;
    psin = ( -root2 + 12.0 * cospi8) * apsin;
    mcos = (  root2 + 12.0 * sinpi8) * amcos;
    pcos = (  root2 - 12.0 * sinpi8) * apcos;
    pre2 = 2.0 * (mcos + msin + pcos + psin);
    pre3 = 16.0 * (cospi8 * (amsin - apsin) + sinpi8 * (apcos - amcos));
    
    

    
    
    def __init__(self, NDIM, N, box, particles, **kwargs):
        '''
        Class to calculate the approximate 2d SANN nearest neighbors. Ask
        Apltug if you need actual 2d SANN. Extension to 3d SANN could be
        implemented.

        Parameters
        ----------
        NDIM : INT
            Number of dimensions. Should be 2 at the moment.
        N : INT
            Number of particles.
        box : TUPLE of FLOATs
            Region in which particles might be. M is equal to or greater
            than NDIM. Values at higher index are ignored.
        particles : NP.ARRAY of N by (at least) NDIM
            Particle locations. Values at higher index are ignored.
        **kwargs : DICT
            Extra input for coordinateFileManipulation.writeFile.

        Returns
        -------
        None.

        '''
        self.NDIM = NDIM
        self.N = N
        self.box = box
        self.particles = particles
        self.kwargs = kwargs
        self.tree = KDTree(self.particles)

        





    def verlet(self, cutoff):
        '''
        Partition particles using a simple cutoff. Makes a matrix with
        particle-partice distances and a matrix with particle id's sorted by
        distance, both of which are needed for SANN. Uses periodic boundaries.

        Parameters
        ----------
        cutoff : FLOAT
            Cutoff length.

        Returns
        -------
        None.

        '''
        
        self.neighborparticles = np.full((self.N, self.N), -1, dtype = 'int')
        self.distanceMatrix = np.zeros((self.N, self.N))
        alreadyfound = np.zeros(self.N, dtype = 'int')        
        
        for i, loci in enumerate(self.particles):
            counter = alreadyfound[i]
            if i % 1000 == 0:
                print("working on particle", i, "of", self.N, "found", counter, "neighbors so far")
            for j, locj in enumerate(self.particles[i + 1:]):
                j += i + 1
                deltaR2 = 0
                for dist1d, maxlenght in zip(loci-locj, self.box[:self.NDIM]):
                    dist1d -= maxlenght * int(2 * dist1d/ maxlenght )
                    deltaR2 += dist1d**2
                deltaRabs = np.sqrt(deltaR2)
                self.distanceMatrix[i, j] = deltaRabs
                self.distanceMatrix[j, i] = deltaRabs
                
                if deltaRabs < cutoff:
                    self.neighborparticles[i][counter] = j
                    counter += 1
                    self.neighborparticles[j][alreadyfound[j]] = i
                    alreadyfound[j] += 1
                    
                    
    
    def rmisolver(self, m, sumr, sumr2, sumr3):
        """
        Calculate the Rmi for 2d SANN
    
        Parameters
        ----------
        m : INT
            Number of particles currently under consideration.
        sumr : FLOAT
            Sum of distances to particles.
        sumr2 : FLOAT
            Sum of squared distances to particles.
        sumr3 : FLOAT
            Sum of cubed distances to particles.
    
        Returns
        -------
        rmi : FLOAT
            Rmi needed in 2d SANN determination.
    
        """
        a = calculateSANN.pre3 * sumr3
        b = calculateSANN.pre2 * sumr2
        c = calculateSANN.pre1 * sumr
        d = calculateSANN.pre0 * m - np.pi
        
        tbdmcc = 3*b*d-c*c
        inner = (9*b*c*d - 27*a*d*d - 2*c*c*c);
        sqrtpart = np.sqrt(inner*inner + 4 * tbdmcc*tbdmcc*tbdmcc )
        cuberootpart = np.cbrt(sqrtpart + inner)
        rmi = cuberootpart/(3 * calculateSANN.cbrt2 * d) - (calculateSANN.cbrt2 * tbdmcc)/(3 * d * cuberootpart) - c/(3.0 * d)
    
        return(rmi)
    
    
    
    def SANN2d(self):
        '''
        Does the main part of the SANN algorithm. Can only be called after
        self.verlet() has been called.

        Returns
        -------
        None, but fills self.numberNNlist with number of nearest neighbors for
        each particle and self.Rmilist with the Rmi for each particle.

        '''
        for i, (pids, dist) in enumerate(zip(self.neighborparticles, self.distanceMatrix)):
            datalist = self.dataarray[i] = []
            for j, particle in enumerate(pids):
                distance = dist[particle]
                if particle == -1:
                    amount = j
                    break
                else:
                    datalist.append((particle, distance))
            
            
            datalist.sort(key = itemgetter(1))
            if amount < 3:
                self.numberNNlist[i] = 0#-1
                self.Rmilist[i] = -1
                
            else:
                sumr = 0
                sumr2 = 0
                sumr3 = 0
                m = 0
                for m in range(3):
                    r = datalist[m][1]
                    sumr += r
                    sumr2 += r**2
                    sumr3 += r**3
                m += 1
                
                while (m < amount):
                    Rmi = self.rmisolver(m, sumr, sumr2, sumr3)
                    if Rmi < datalist[m][1]:
                        self.numberNNlist[i] = m
                        self.Rmilist[i] = Rmi
                        break
                    r = datalist[m][1]
                    sumr += r
                    sumr2 += r**2
                    sumr3 += r**3
                    m+=1
                else: #nobreak
                    Rmi = self.rmisolver(m, sumr, sumr2, sumr3)
                    self.numberNNlist[i] = 0#-2
                    self.Rmilist[i] = Rmi
    
    def SANNtree2d(self, amount = 25):
        """
        Calculates SANN using a kdtree. Uses no periodic boundary conditions.

        Parameters
        ----------
        amount : INT, optional
            The number of nearest neighbors to query from the kdTree. Increase
            if some particles are assigned no neighbors. Decrease to decrease
            runtime. The default is 25.

        Returns
        -------
        None, but fills self.numberNNlist with number of nearest neighbors for
        each particle and self.Rmilist with the Rmi for each particle.

        """
        for i, loci in enumerate(self.particles):
            dists, ids = self.tree.query(loci, k = amount)
            
            sumr = 0
            sumr2 = 0
            sumr3 = 0
            m = 0
            for m in range(1,4):
                r = dists[m]
                sumr += r
                sumr2 += r**2
                sumr3 += r**3
            
            while (m < amount):
                Rmi = self.rmisolver(m, sumr, sumr2, sumr3)
                m += 1
                if Rmi < dists[m]:
                    self.numberNNlist[i] = m - 1
                    self.Rmilist[i] = Rmi
                    break
                r = dists[m]
                sumr += r
                sumr2 += r**2
                sumr3 += r**3
            else: #nobreak
                Rmi = self.rmisolver(m, sumr, sumr2, sumr3)
                self.numberNNlist[i] = 0#-2
                self.Rmilist[i] = Rmi
    
    
    def SANN3d(self):
        '''
        to be filled with 3d SANN algorithm
        '''
        pass
    def SANNtree3d(self):
        '''
        to be filled with 3d SANN algorithm
        '''
        pass
    
    
    def SANNtree(self):
        '''
        Calculates SANN using kd tree, thus using no periodic boundaries.
        Automatically determines 2d or 3d, however, only 2d is implemented.
        Sets up self.numberNNlist and self.Rmilist to receive the output.
        '''
        self.numberNNlist = np.zeros(self.N, dtype = 'int')
        self.Rmilist = np.zeros(self.N, dtype = 'float')
        
        if self.NDIM == 2:
            self.SANNtree2d()
        if self.NDIM == 3:
            self.SANNtree3d()
    
    def SANN(self):
        '''
        Calculates SANN using periodic boundaries. Call only after calling
        self.verlet(). Automatically determines 2d or 3d, however, only 2d is
        implemented. Sets up self.numberNNlist and self.Rmilist.
        '''
        
        self.numberNNlist = np.zeros(self.N, dtype = 'int')
        self.Rmilist = np.zeros(self.N, dtype = 'float')
        self.dataarray = [0]*self.N
        
        if self.NDIM == 2:
            self.SANN2d()
        if self.NDIM == 3:
            self.SANN3d()
                        
            
    def printsann(self, filename, remove0 = False):
        '''
        Write a file with the SANN results stored as colors.

        Parameters
        ----------
        filename : STR
            Path & filename of file to write.
        remove0 : BOOL, optional
            Set to TRUE if you don't want to print the particles determined to
            have 0 neighbors. The default is False.

        Returns
        -------
        None.

        Writes
        ------
        File to filename with N + 4 lines, structured as follows. This
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

        '''
        self.myfile = coordinateFileManipulation(filename)
        self.myfile.inputinfo(self.N, self.box, self.particles, colorlist = self.numberNNlist)
        self.myfile.rearrangecolumn(1)
        if not remove0:
            self.myfile.writeFile(**self.kwargs)
        else:
            self.myfile.removeParticles(0)
            self.myfile.writeFile(**self.kwargs)

        





class fromfileSANN(calculateSANN):
    def __init__(self, myfile, **kwargs):
        '''
        Class to calculate the approximate 2d SANN nearest neighbors. Ask
        Apltug if you need actual 2d SANN. Extension to 3d SANN could be
        implemented.

        Parameters
        ----------
        myfile : STR
            path and filename of file to be used.
        **kwargs : DICT
            Extra input for coordinateFileManipulation.writeFile.

        Returns
        -------
        None.

        '''
        self.myfile = coordinateFileManipulation(myfile)
        self.myfile.readFile()
        
        self.N = self.myfile.N
        self.box = self.myfile.box
        self.particles = self.myfile.particlelist
        self.kwargs = kwargs
        
        self.findNDIM()
        self.tree = KDTree(self.particles)
        
        
    def findNDIM(self):
        """
        Find number of dimensions for a coordinate file.

        Returns
        -------
        None.

        """
        if all(z == self.particles[0,2] for z in self.particles[:,2]):
            self.NDIM = 2
        else:
            self.NDIM = 3
        
    def printsann(self, filename):
        '''
        Write a file with the SANN results stored as colors.

        Parameters
        ----------
        filename : STR
            Path & filename of file to write.

        Returns
        -------
        None.

        Writes
        ------
        File to filename with N + 4 lines, structured as follows. This
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
        '''
        self.myfile.filename = filename
        self.myfile.writeFile(colors = self.numberNNlist)
        










