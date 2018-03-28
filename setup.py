
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='svae',
    version='0.1.0',
    author='Nicola De Cao, Tim R. Davidson, Luca Falorsi',
    author_email='nicola.decao@gmail.com',
    description='Tensorflow implementation of Hyperspherical Variational Auto-Encoders',
    license='MIT',
    keywords='SVAE hyperspherical vMF tensorflow VAE',
    url='https://nicola-decao.github.io/SVAE/',
    download_url='https://github.com/nicola-decao/SVAE',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    install_requires=['numpy', 'tensorflow>=1.6.0', 'scipy'],
    packages=find_packages()
)
