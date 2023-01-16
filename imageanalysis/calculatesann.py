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



from imageanalysis.coordinatefilemaniputlations import CoordinateFileManipulation
import numpy as np
from operator import itemgetter

from scipy.spatial import KDTree


    


class CalculateSANN:
    
    _cbrt2 = np.cbrt(2)
    _root2 = np.sqrt(2)
    _sinpi8 = np.sin(np.pi/8)
    _cospi8 = np.cos(np.pi/8)
    
    _amsin = np.arccos(0.5*(1.0 - _sinpi8));
    _apsin = np.arccos(0.5*(1.0 + _sinpi8));
    _amcos = np.arccos(0.5*(1.0 - _cospi8));
    _apcos = np.arccos(0.5*(1.0 + _cospi8));
    _msin = (1.0 - _root2 - 2.0*(_cospi8 - _sinpi8))*_amsin;
    _psin = (1.0 - _root2 + 2.0*(_cospi8 - _sinpi8))*_apsin;
    _mcos = (1.0 + _root2 + 2.0*(_cospi8 + _sinpi8))*_amcos;
    _pcos = (1.0 + _root2 - 2.0*(_cospi8 + _sinpi8))*_apcos;
    _pre0 = 0.25*(_mcos + _msin + _pcos + _psin);
    
    _msin = ( 2.0*_root2 + 9.0*_cospi8 - _sinpi8)*_amsin;
    _psin = ( 2.0*_root2 - 9.0*_cospi8 + _sinpi8)*_apsin;
    _mcos = (-2.0*_root2 - _cospi8 - 9.0*_sinpi8)*_amcos;
    _pcos = (-2.0*_root2 + _cospi8 + 9.0*_sinpi8)*_apcos;
    _pre1 = _mcos + _msin + _pcos + _psin;
    
    _msin = (-_root2 - 12.0*_cospi8)*_amsin;
    _psin = (-_root2 + 12.0*_cospi8)*_apsin;
    _mcos = ( _root2 + 12.0*_sinpi8)*_amcos;
    _pcos = ( _root2 - 12.0*_sinpi8)*_apcos;
    _pre2 = 2.0*(_mcos + _msin + _pcos + _psin);
    _pre3 = 16.0*(_cospi8*(_amsin - _apsin) + _sinpi8*(_apcos - _amcos));
    
    

    
    
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
        
        self.neighborparticles = np.full((self.N, self.N), -1, dtype='int')
        self.distanceMatrix = np.zeros((self.N, self.N))
        alreadyfound = np.zeros(self.N, dtype='int')        
        
        for i, loci in enumerate(self.particles):
            counter = alreadyfound[i]
            if i%1000 == 0:
                print("working on particle", i, "of", self.N, "found", counter,
                      "neighbors so far")
            for j, locj in enumerate(self.particles[i + 1:]):
                j += i+1
                deltaR2 = 0
                for dist1d, maxlenght in zip(loci-locj, self.box[:self.NDIM]):
                    dist1d -= maxlenght * int(2 * dist1d/maxlenght )
                    deltaR2 += dist1d**2
                deltaRabs = np.sqrt(deltaR2)
                self.distanceMatrix[i, j] = deltaRabs
                self.distanceMatrix[j, i] = deltaRabs
                
                if deltaRabs < cutoff:
                    self.neighborparticles[i][counter] = j
                    counter += 1
                    self.neighborparticles[j][alreadyfound[j]] = i
                    alreadyfound[j] += 1
                    
                    
    
    def _rmisolver(self, m, sumr, sumr2, sumr3):
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
        a = CalculateSANN._pre3 * sumr3
        b = CalculateSANN._pre2 * sumr2
        c = CalculateSANN._pre1 * sumr
        d = CalculateSANN._pre0 * m - np.pi
        
        tbdmcc = 3*b*d - c*c
        inner = (9*b*c*d - 27*a*d*d - 2*c*c*c);
        sqrtpart = np.sqrt(inner*inner + 4*tbdmcc*tbdmcc*tbdmcc )
        cuberootpart = np.cbrt(sqrtpart + inner)
        rmi = (cuberootpart/(3 * CalculateSANN._cbrt2 * d)
               - (CalculateSANN._cbrt2 * tbdmcc)/(3 * d * cuberootpart)
               - c/(3.0 * d)
               )
    
        return(rmi)
    
    
    
    def sann2d(self):
        '''
        Does the main part of the SANN algorithm. Can only be called after
        self.verlet() has been called.

        Returns
        -------
        None, but fills self.numberNNlist with number of nearest neighbors for
        each particle and self.Rmilist with the Rmi for each particle.

        '''
        for i, (pids, dist) in enumerate(zip(self.neighborparticles,
                                             self.distanceMatrix)):
            datalist = self.dataarray[i] = []
            for j, particle in enumerate(pids):
                distance = dist[particle]
                if particle == -1:
                    amount = j
                    break
                else:
                    datalist.append((particle, distance))
            
            
            datalist.sort(key=itemgetter(1))
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
                    Rmi = self._rmisolver(m, sumr, sumr2, sumr3)
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
    
    def sanntree2d(self, amount=25):
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
            dists, ids = self.tree.query(loci, k=amount)
            
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
    
    
    def sann3d(self):
        '''
        to be filled with 3d SANN algorithm
        '''
        pass
    def sanntree3d(self):
        '''
        to be filled with 3d SANN algorithm
        '''
        pass
    
    
    def sanntree(self):
        '''
        Calculates SANN using kd tree, thus using no periodic boundaries.
        Automatically determines 2d or 3d, however, only 2d is implemented.
        Sets up self.numberNNlist and self.Rmilist to receive the output.
        '''
        self.numberNNlist = np.zeros(self.N, dtype='int')
        self.Rmilist = np.zeros(self.N, dtype='float')
        
        if self.NDIM == 2:
            self.SANNtree2d()
        if self.NDIM == 3:
            self.SANNtree3d()
    
    def sann(self):
        '''
        Calculates SANN using periodic boundaries. Call only after calling
        self.verlet(). Automatically determines 2d or 3d, however, only 2d is
        implemented. Sets up self.numberNNlist and self.Rmilist.
        '''
        
        self.numberNNlist = np.zeros(self.N, dtype='int')
        self.Rmilist = np.zeros(self.N, dtype='float')
        self.dataarray = [0]*self.N
        
        if self.NDIM == 2:
            self.SANN2d()
        if self.NDIM == 3:
            self.SANN3d()
                        
            
    def printsann(self, filename, remove0=False):
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
        self.myfile = CoordinateFileManipulation(filename)
        self.myfile.inputinfo(self.N, self.box, self.particles,
                              colorlist=self.numberNNlist)
        self.myfile.rearrangecolumn(1)
        if not remove0:
            self.myfile.writeFile(**self.kwargs)
        else:
            self.myfile.removeParticles(0)
            self.myfile.writeFile(**self.kwargs)

        





class FromFileSANN(CalculateSANN):
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
        self.myfile = CoordinateFileManipulation(myfile)
        self.myfile.readfile()
        
        self.N = self.myfile.N
        self.box = self.myfile.box
        self.particles = self.myfile.particlelist
        self.kwargs = kwargs
        
        self.findndim()
        self.tree = KDTree(self.particles)
        
        
    def findndim(self):
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
        










