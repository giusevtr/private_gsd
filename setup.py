from setuptools import setup

setup(
    name='PrivateGSD',
    version='1.0',
    description='PrivateGSD',
    url='https://github.com/giusevtr/private_genetic_algorithm.git',
    author='Giuseppe Vietri, Jingwu Tang, Terrence Liu',
    license='MIT',
    packages=['dev', 'models', 'stats', 'utils'],
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn',
                      'tqdm', 'matplotlib', 'seaborn', 'chex', 'flax', 'diffprivlib'],
)