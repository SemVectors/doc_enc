import os
from setuptools import (
    setup,
)

setup(
    name='doc_enc',
    version=os.environ.get('version', '0'),
    description='Encoding texts as dense vectors',
    author='dvzubarev',
    author_email='zubarev@isa.ru',
    license='MIT',
    packages=['doc_enc'],
    install_requires=['numpy'],
)
