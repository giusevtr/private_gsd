tfrom setuptools import setup

setup(
    name='evo-privsyn',
    version='1.0',
    description='PrivGA',
    url='https://github.com/giusevtr/evolutionary_private_synthetic_data/',
    author='Giuseppe Vietri',
    license='MIT',
    packages=['examples', 'models', 'stats', 'utils'],
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn',
                      'tqdm', 'matplotlib', 'seaborn',
                      'folktables', 'evosax'],
)