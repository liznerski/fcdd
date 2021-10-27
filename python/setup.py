import os.path as pt
from setuptools import setup, find_packages

PACKAGE_DIR = pt.abspath(pt.join(pt.dirname(__file__)))

packages = find_packages(PACKAGE_DIR)

package_data = {
    package: [
        '*.py',
        '*.txt',
        '*.json',
        '*.npy'
    ]
    for package in packages
}

with open(pt.join(PACKAGE_DIR, 'requirements.txt')) as f:
    dependencies = [l.strip(' \n') for l in f]

setup(
    name='fcdd',
    version='1.1.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='deep-learning anomaly-detection explainability fcdd fully convolutional cnn',
    packages=packages,
    package_data=package_data,
    install_requires=dependencies,
)
