########################################################################
#       File based on https://github.com/Blosc/bcolz
########################################################################
#
# License: GPL
# Created: June 11, 2017
#      Author:  Carst Vaartjes - cvaartjes@visualfabriq.com
#
########################################################################
import codecs
import os
import platform

import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))
LINUX_OS = "Linux"
WINDOW_OS = "Windows"
CURRENT_OS = platform.system()

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Operating System :: Unix',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.10',
]


def read(*parts):
    """
    Build an absolute path from *parts* and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


install_requires = [
    'joblib==0.13.2; python_version <= "2.7"',
    'joblib>=0.16.0; python_version > "3.3"',
    'numpy==1.16.6; python_version <= "2.7"',
    'numpy>=1.19.1; python_version > "3.3"',
    'pandas==0.24.2; python_version <= "2.7"',
    'pandas>=1.1.0; python_version > "3.3"',
    'numexpr==2.7.3; python_version <= "2.7"',
    'numexpr>=2.7.1; python_version > "3.3"',
    'scipy==1.2.3; python_version <= "2.7"',
    'scipy==1.7.3; python_version > "3.3"',
    'category-encoders==2.2.2',
    # draw libraries
    'matplotlib<=2.2.5; python_version <= "2.7"',
    'matplotlib>=3.1.2; python_version > "3.3"',
    'seaborn==0.9.1; python_version <= "2.7"',
    'seaborn>=0.10.1; python_version > "3.3"',
    # the prediction libraries;
    # sklearn is pinned because if a user pickles a model with a different version, the unpickling creates issues
    'scikit-learn==0.20.4; python_version <= "2.7"',
    'scikit-learn==0.20.4; python_version < "3.10"',
    'scikit-learn==1.1.3; python_version == "3.10"',
    'xgboost==0.82; python_version <= "2.7"',
    'xgboost==0.82; python_version > "3.3"',
    'statsmodels==0.10.2; python_version <= "2.7"',
    'statsmodels>=0.12.0; python_version > "3.3"',
    'ipython==5.8.0; python_version <= "2.7"',
    'ipython>=7.11.1; python_version > "3.3"',
]
tests_requires = [
    'pytest',
    'pytest-cov'
]

extras_requires = {
    'test': tests_requires
}

setuptools.setup(
    name='vf_portalytics',
    description="A consistent interface for creating Machine Learning Models compatible with VisualFabriq environment",
    long_description=read("README.md"),
    author='Christos Tselas',
    author_email='ctselas@visualfabriq.com',
    maintainer='Carst Vaartjes',
    maintainer_email='cvaartjes@visualfabriq.com',
    url='https://github.com/visualfabriq/portalytics',
    setup_requires=['pbr', 'pytest-runner'],
    tests_require=tests_requires,
    extras_require=dict(
        test=tests_requires
    ),
    install_requires=install_requires
)