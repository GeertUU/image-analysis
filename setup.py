'''
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
'''


from setuptools import setup, find_packages



setup(
    name = 'imageanalysis',
    version = '1.1.9',

    url='https://github.com/GeertUU/image-analysis',
    author='Geert',
    author_email='g.h.a.schulpen@uu.nl',

    packages=find_packages(include=["imageanalysis", "imageanalysis.*"]),
    
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'opencv-python',
        'trackpy==0.5.0',
    ],
)
