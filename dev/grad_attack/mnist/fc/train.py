import jax
import jax.numpy as np
from jax import random, grad, jit, vmap
import numpy as onp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.scipy.special import logsumexp
import optax
import torch
from torchvision import datasets, transforms
import time
import pickle


def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0, x)


jit_ReLU = jit(ReLU)


def relu_layer(params, x):
    """ Simple ReLu layer for single sample """
    return ReLU(np.dot(params[0], x) + params[1])


def vmap_relu_layer(params, x):
    """ vmap version of the ReLU layer """
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))


def initialize_mlp(sizes, key):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, len(sizes))
    # Initialize a single layer with Gaussian weights -  helper function
    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]





def forward_pass(params, in_array):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU hidden layers
    for w, b in params[:-1]:
        activations = relu_layer([w, b], activations)

    # Perform final trafo to logits
    final_w, final_b = params[-1]
    logits = np.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

# Make a batched version of the `predict` function
batch_forward = vmap(forward_pass, in_axes=(None, 0), out_axes=0)


def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k """
    return np.array(x[:, None] == np.arange(k), dtype)

def loss(params, in_arrays, targets):
    """ Compute the multi-class cross-entropy loss """
    preds = batch_forward(params, in_arrays)
    return -np.sum(preds * targets)

def accuracy(params, data_loader):
    """ Compute the accuracy for a provided dataloader """
    acc_total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        images = np.array(data).reshape(data.size(0), 28*28)
        targets = one_hot(np.array(target), num_classes)

        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(batch_forward(params, images), axis=1)
        acc_total += np.sum(predicted_class == target_class)
    return acc_total/len(data_loader.dataset)



@jit
def get_grads(params, x, y):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, x, y)
    # updates, opt_state = optimizer.update(grads, opt_state)
    # params = optax.apply_updates(params, updates)
    return grads, value

@jit
def update(params, grads, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    # value, grads = value_and_grad(loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state




# opt_init, opt_update, get_params =
# opt_state = opt_init(params)

num_epochs = 50
num_classes = 10

# Set the PyTorch Data Loader for the training & test set
batch_size = 100

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False)


def run_mnist_training_loop(num_epochs, optimizer, params, net_type="MLP"):
    """ Implements a learning loop over epochs. """
    # Initialize placeholder for loggin
    log_acc_train, log_acc_test, train_loss = [], [], []

    # Get initial accuracy after random init
    train_acc = accuracy(params, train_loader)
    test_acc = accuracy(params, test_loader)
    log_acc_train.append(train_acc)
    log_acc_test.append(test_acc)

    opt_state = optimizer.init(params)

    # Loop over the training epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        grads_sum = None

        # open a file, where you ant to store the data
        params_file = open(f'params/param_{epoch}.p', 'wb')
        grad_file = open(f'params/grad_param_{epoch}.p', 'wb')

        # dump information to that file
        pickle.dump(params, params_file)

        for batch_idx, (data, target) in enumerate(train_loader):
            x = None
            if net_type == "MLP":
                # Flatten the image into 784 vectors for the MLP
                x = np.array(data).reshape(data.size(0), 28*28)
            elif net_type == "CNN":
                # No flattening of the input required for the CNN
                x = np.array(data)
            y = one_hot(np.array(target), num_classes)
            this_grads, loss = get_grads(params, x, y)
            if batch_idx == 0:
                grads_sum = this_grads
                pickle.dump(this_grads, grad_file)
            else:
                grads_sum = [(g0[0] + g1[0], g0[1] + g1[1]) for g0, g1 in zip(grads_sum, this_grads)]

            train_loss.append(loss)

        params, opt_state = update(params, grads_sum,  opt_state)

        epoch_time = time.time() - start_time
        train_acc = accuracy(params, train_loader)
        test_acc = accuracy(params, test_loader)
        log_acc_train.append(train_acc)
        log_acc_test.append(test_acc)
        print("Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f}".format(epoch+1, epoch_time,
                                                                    train_acc, test_acc))

    return train_loss, log_acc_train, log_acc_test



if __name__ == '__main__':
    key = random.PRNGKey(1)
    layer_sizes = [784, 512, 512, 10]
    # Return a list of tuples of layer weights
    params = initialize_mlp(layer_sizes, key)

    # Defining an optimizer in Jax
    step_size = 1e-3
    optimizer = optax.adam(step_size)
    train_loss, train_log, test_log = run_mnist_training_loop(num_epochs,
                                                              optimizer,
                                                              params,
                                                              # opt_state,
                                                              net_type="MLP")
