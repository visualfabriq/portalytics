from __future__ import absolute_import

import pathlib

from sys import version_info as v

import pkg_resources
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

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

tests_requires = [
    'pytest',
    'pytest-cov',
    'pyarrow'
]

extras_requires = []

setuptools.setup(
    name='vf_portalytics',
    setup_requires=['pbr', 'pytest-runner'],
    pbr=True,
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
