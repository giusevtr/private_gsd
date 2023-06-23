
# Description 
Implementation of the GSD mechanism from https://arxiv.org/abs/2306.03257


# Example 

Visit this [Colab link](https://colab.research.google.com/drive/1t49XFG51pmcIsRqAhF_veHbrbfrVZBuy?usp=sharing) to
start using GSD.


# Setup

Set up conda environment
````
conda create -n gsd python=3.9
conda activate gsd
pip install --upgrade pip
````

Install via setuptools
````
pip install -e .
````

Install [JAX](https://github.com/google/jax#installation) separately. For example,
````
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jaxlib==0.4.11 --upgrade
````
Please make sure that the command you execute matches your system (i.e., tpu vs. gpu, right CUDA/cuDNN versions, etc.)

Download and preprocess datasets using [dp-data](https://github.com/terranceliu/dp-data).
````
git clone https://github.com/terranceliu/dp-data
cd dp-data
pip install -e .
./preprocess_all.sh
````
