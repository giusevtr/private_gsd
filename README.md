# Setup

Set up conda environment
````
conda create -n evo-privsyn python=3.8
conda activate evo-privsyn
pip install --upgrade pip
````

Install via setuptools
````
pip install -e .
````

Install JAX separately
````
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
````

# Execution

Run ADULT
````
cd examples
python adult_example.py
````
