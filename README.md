# imageAnalysis
Set of functions for dealing with image data based around the python packages opencv and trackpy

## Info
- Created by: Geert Schulpen
- Email: g.h.a.schulpen@uu.nl
- Version: 1.0.0


## Usage
### Image analysis
#### Realspace
Two realspace classes are available `realspace` and `realspacefromfile` to allow loading of images both from a file or from memory. Several functions are defined, which have as main functionality to detect (spherical) particles. The locate function returns an instance of the `calculateSANN` class. Additionally, a DFT function can generate a `Fourier` instance (see below).

#### Fourierspace
The class `Fourier` includes some operations on frequency domain representations of images. High or low frequency filters can be applied and the inverse DFT can generate a `realspace` instance.

#### Batch processing
The class `imageframes` has some preliminary functions to process a batch of images, located in the same folder.

### Nearest Neighbor calculations
The class `calculateSANN` can be used to approximate the SANN nearest neighbors in a 2 dimensional setting. A chebyshev approximation to third order is used to approximate the 2d extension.
The code has non-implemented functions to include 3d SANN.



## Changelog
