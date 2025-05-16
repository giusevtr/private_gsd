from setuptools import setup, find_packages

setup(
    name='genetic_sd',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'smartnoise-synth',
        'numpy<2',  # Ensures compatibility with packages compiled against NumPy 1.x
        'jupyter',
        'pytest',  # example dev dependency
        'tqdm'
    ],
    extras_require={
        'cpu': [
            'jax[cpu]==0.4.6',
            'flax',
        ]
    },
    python_requires='>=3.9',
)
