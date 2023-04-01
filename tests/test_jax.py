import jax
import jax.numpy as jnp
from jax import lax

def create_histogram(histogram, value):
    """
    Given an integer value from 0 to 100 and a current histogram,
    return an updated histogram with the value incorporated.
    """
    return histogram.at[value].add(1), None

def create_aggregated_histogram(input_array):
    """
    Given a JAX array of integers from 0 to 100,
    return a JAX array of size 100 representing the aggregated histogram of the input array.
    """
    init_histogram = jnp.zeros(100)
    aggregated_histogram, _ = lax.scan(create_histogram, init_histogram, input_array.astype(int))
    return aggregated_histogram

# Example usage
input_array = jnp.array([1, 2, 3, 3])
result = create_aggregated_histogram(input_array)
print(result)