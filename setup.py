########################################################################
#       File based on https://github.com/Blosc/bcolz
########################################################################
#
# License: GPL
# Created: June 11, 2017
#       Author:  Carst Vaartjes - cvaartjes@visualfabriq.com
#
########################################################################
from __future__ import absolute_import

import codecs
import os

from setuptools import setup, Extension, find_packages
from os.path import abspath
from sys import version_info as v
from setuptools.command.build_ext import build_ext as _build_ext


# Check this Python version is supported
if any([v < (2, 6), (3,) < v < (3, 6)]):
    raise Exception("Unsupported Python version %d.%d. Requires Python >= 2.7 "
                    "or >= 3.6." % v[:2])


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


HERE = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def get_version():
    with codecs.open(abspath('VERSION'), "r", "utf-8") as f:
        return f.readline().rstrip('\n')

# Sources & libraries
inc_dirs = [abspath('bquery')]
try:
    import numpy as np
    inc_dirs.append(np.get_include())
except ImportError as e:
    pass
lib_dirs = []
libs = []
def_macros = []
sources = []

cmdclass = {'build_ext': build_ext}

optional_libs = []
install_requires = [
    'pip>=19.3.1',
    'setuptools>=41.6.0',
    'numexpr>=2.7.0',
    'statsmodels==0.10.1',
    'seaborn>=0.9.0'
]

setup_requires = []
tests_requires = []

if v < (3,):
    tests_requires.extend(['unittest2', 'mock'])
    install_requires.extend(['numpy==1.16.6'])
    install_requires.extend(['scipy==1.2.2'])
    install_requires.extend(['matplotlib==2.2.4'])
    install_requires.extend(['pandas==0.23.4'])
    install_requires.extend(['scikit-learn==0.20.4'])
    install_requires.extend(['xgboost==0.82'])
    install_requires.extend(['ipython==5.8.0'])
else:
    install_requires.extend(['numpy>=1.18.1'])
    install_requires.extend(['scipy>=1.4.1'])
    install_requires.extend(['pandas>=0.25.3'])
    install_requires.extend(['scikit-learn>=0.22.1'])
    install_requires.extend(['xgboost>=0.90'])


extras_requires = []
ext_modules = []

package_data = {'vf_portalytics': []}
classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
]

setup(
    name="vf_portalytics",
    version=get_version(),
    description='A portable analytics framework for Python',
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    author='Carst Vaartjes',
    author_email='cvaartjes@visualfabriq.com',
    maintainer='Carst Vaartjes',
    maintainer_email='cvaartjes@visualfabriq.com',
    url='https://github.com/visualfabriq/portalytics',
    license='GPLv3',
    platforms=['any'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_requires,
    extras_require=dict(
        optional=extras_requires,
        test=tests_requires
    ),
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    zip_safe=True
)