from __future__ import absolute_import

from sys import version_info as v

import setuptools

# Check this Python version is supported
if v[:2] != (3, 11):
    raise Exception("Unsupported Python version %d.%d. Requires Python == 3.11" % v[:2])

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Operating System :: Unix',
    'Programming Language :: Python :: 3.11'
]


with open('README.md') as f:
    long_description = f.read()

install_requires = [
    "joblib~=1.2.0",
    "numpy==1.23.5",
    "pandas~=1.1",
    "numexpr>=2.7.1",
    "scipy==1.11.1",
    "category-encoders==2.2.2",
    "seaborn>=0.10.1",
    "scikit-learn==1.1.3",
    "xgboost==0.82",
    "statsmodels>=0.12.0",
    "fbprophet==0.5; python_version <= '2.7'",
    "hyperopt==0.2.5",
    "mock==2.0.0"
]
tests_requires = [
    'pytest',
    'pytest-cov'
]

extras_requires = []

setuptools.setup(
    name='vf_portalytics',
    description='A consistent interface for creating Machine Learning Models compatible with VisualFabriq environment',
    long_description=long_description,
    author='Christos Tselas',
    author_email='ctselas@visualfabriq.com',
    maintainer='DataFlow Domain',
    maintainer_email='cvaartjes@visualfabriq.com',
    platforms=['any'],
    package_data={'vf_portalytics': []},
    packages=setuptools.find_packages(),

    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=dict(
        optional=extras_requires,
        test=tests_requires
    ),
    classifiers=classifiers,
)
