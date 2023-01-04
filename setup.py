from setuptools import setup, find_packages

from imageAnalysis import __version__

setup(
    name='imageAnalysis',
    version=__version__,

    url='https://github.com/GeertUU/image-analysis',
    author='Geert',
    author_email='g.h.a.schulpen@uu.nl',

    packages=find_packages(include=["imageAnalysis", "imageAnalysis.*"]),
    
    install_requires=[
        "numpy>=1.19.2",
        'operator',
        'scipy',
        'matplotlib',
        'opencv-python',
        'trackpy',
        'os',
    ],
)
