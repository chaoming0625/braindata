# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages
from setuptools import setup

# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'brainpy_datasets', '__init__.py'), 'r') as f:
  init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# obtain long description from README
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  README = f.read()

# installation packages
packages = find_packages()
if 'docs' in packages:
  packages.remove('docs')
if 'tests' in packages:
  packages.remove('tests')

# setup
setup(
  name='brainpy_datasets',
  version=version,
  description='BrainPy Datasets',
  long_description=README,
  long_description_content_type="text/markdown",
  author='BrainPy Team',
  author_email='chao.brain@qq.com',
  packages=packages,
  python_requires='>=3.7',
  install_requires=['brainpy>=2.2.0', 'requests'],
  url='https://github.com/brainpy/brainpy-datasets',
  project_urls={
    "Bug Tracker": "https://github.com/brainpy/brainpy-datasets/issues",
    # "Documentation": "https://brainpy.readthedocs.io/",
    "Source Code": "https://github.com/brainpy/brainpy-datasets",
  },
  keywords=('computational neuroscience, '
            'brain-inspired computation, '
            'brainpy, '
            'dataset, '
            'brain modeling, '
            'brain dynamics modeling, '
            'brain dynamics programming'),
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
  license='GPL-3.0 license',
)
