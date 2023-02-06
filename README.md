# Setup

Set up conda environment
````
conda create -n privga python=3.9
conda activate privga
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


Download datasets
````
git clone https://github.com/terranceliu/dp-data
cd dp-data
pip install -e .
./preprocess_all.sh
````




# Execution

Run example: 
````
cd examples
````
