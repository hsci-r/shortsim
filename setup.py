from setuptools import setup, find_packages

with open('README.md') as fp:
    README = fp.read()

setup(
    name='shortsim',
    version='0.1.0',
    author='Maciej Janicki',
    author_email='maciej.janicki@helsinki.fi',
    description='Short string similarity toolkit.',
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages('src', exclude=['tests', 'tests.*']),
    package_dir={'': 'src'},
    test_suite='tests',
    install_requires=['numpy', 'scipy', 'faiss', 'tqdm'],
    entry_points={
        'console_scripts' : [
            'shortsim-align   = shortsim.scripts.align:main',
            'shortsim-cluster = shortsim.scripts.cluster:main',
            'shortsim-ngrcos  = shortsim.scripts.ngrcos:main',
        ]
    }
)