from setuptools import setup, find_packages

setup(
    name='gwas',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[],
    author='Thiago Sanches & Apurva Chitre',
    author_email='',
    description='A general package to run GWAS associations and visualize the results',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sanchestm/GWAS-pipeline',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'gwas-cli = gwas.gwas_cli:main',
            'gwas-cli-argparse = gwas.gwas_cli_argparse:main',
        ],
    },
    python_requires='>=3.10',
)