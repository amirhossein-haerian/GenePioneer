from setuptools import setup, find_packages

setup(
    name='genepioneer',
    version='1.0.2',
    description='A Python package for identifying essential genes in cancer.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Amirhossein Haerian, Golnaz Taheri',
    author_email='haerian.amirhossein@gmail.com',
    url='https://github.com/amirhossein-haerian/GenePioneer',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'networkx',
        'pandas',
        'scipy',
        'openpyxl',
        'scipy',
        'scikit-learn',
        'igraph',
        'leidenalg',
        'matplotlib',
        'gprofiler-official'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    package_data={
        'genepioneer': ['Data/**/*', 'Data/**/**/*'],  
    },
    keywords='cancer, genomics, gene-analysis',
)