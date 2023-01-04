from setuptools import setup, find_packages



setup(
    name = 'imageAnalysis',
    version = '1.0.3',

    url='https://github.com/GeertUU/image-analysis',
    author='Geert',
    author_email='g.h.a.schulpen@uu.nl',

    packages=find_packages(include=["imageAnalysis", "imageAnalysis.*"]),
    
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'opencv-python',
        'trackpy',
    ],
)
