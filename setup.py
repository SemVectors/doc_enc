import os
from setuptools import setup, find_packages, find_namespace_packages

setup(
    name='doc_enc',
    version=os.environ.get('version', '0'),
    description='Encoding texts as dense vectors',
    author='dvzubarev',
    author_email='zubarev@isa.ru',
    license='MIT',
    packages=find_namespace_packages(include=["hydra_plugins.*"])
    + find_packages(include=["doc_enc*"]),
    # packages=find_packages(),
    install_requires=['numpy'],
    entry_points={
        'console_scripts': [
            'run_training = doc_enc.training.run_training:train_cli',
        ],
    },
)
