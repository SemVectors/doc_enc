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
            'run_preproc = doc_enc.training.run_training:preproc_cli',
            'run_repack = doc_enc.training.run_training:repack_cli',
            'run_eval = doc_enc.eval.run_eval:eval_cli',
            'docenccli = doc_enc.cli:main',
            'fine_tune_classif = doc_enc.finetune_classif:fine_tune_classif_cli',
        ],
    },
)
