from setuptools import setup, find_packages



setup(
    name = 'imageanalysis',
    version = '1.1.2',

    url='https://github.com/GeertUU/image-analysis',
    author='Geert',
    author_email='g.h.a.schulpen@uu.nl',

    packages=find_packages(include=["imageanalysis", "imageanalysis.*"]),
    
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'opencv-python',
        'trackpy',
    ],
)
