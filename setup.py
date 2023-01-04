from setuptools import setup, find_packages

from imageAnalysis import __version__

setup(
    name='imageAnalysis',
    version=__version__,

    url='https://github.com/GeertUU/image-analysis',
    author='Geert',
    author_email='g.h.a.schulpen@uu.nl',

    packages=find_packages(),
    
    install_requires=[
        'numpy',
        'operator',
        'scipy',
        'matplotlib',
        'opencv-python',
        'trackpy',
        'os',
    ],
)
