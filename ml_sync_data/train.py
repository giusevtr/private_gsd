import numpy as np
import jax.numpy as jnp
from utils import Dataset, Domain
import jax
import optax
from flax import linen as nn
# from flax.metrics import tensorboard
from flax.training import train_state
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union


LABEL_SIZE = 2
class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=LABEL_SIZE)(x)
        return x


from flax.linen import Module
from flax.linen import Dense

class DenseExplicit(Dense):
    in_features: Optional[int] = None

    def setup(self):
        # We feed a fake batch through the module, which initialized parameters.
        # Assuming we're in a jit, should use no FLOPs -- "just shape inference".
        self.__call__(jnp.zeros((1, self.in_features, )))

class MLP(Module):
    in_dimension: int
    out_dimension: int
    hidden_dimension: int = 100

    def setup(self):
        self.dense1 = DenseExplicit(in_features=self.in_dimension, features=self.hidden_dimension)
        self.dense2 = DenseExplicit(in_features=self.hidden_dimension, features=self.out_dimension)

    def __call__(self, x):
        return self.dense2(nn.relu(self.dense1(x)))


@jax.jit
def apply_model(state, data, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, data)
        one_hot = jax.nn.one_hot(labels, LABEL_SIZE)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['features'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['features']))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['features'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy

def create_train_state(rng, lr=0.01, momentum=0.001):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(lr, momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

# def train_ML(data: Dataset, label_col):
#     print(X.shape)
#     print(Y.shape)

def train_and_evaluate(train_ds,
                       # config: ml_collections.ConfigDict,
                       # workdir: str
                       epochs: 100,
                       batch_size: 32
                       ) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """
    rng = jax.random.PRNGKey(0)

    # summary_writer = tensorboard.SummaryWriter(workdir)
    # summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, lr=0.01)

    for epoch in range(1, epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                        batch_size,
                                                        input_rng)
        _, test_loss, test_accuracy = apply_model(state, train_ds['features'],
                                                  train_ds['label'])

        debug_str = 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f' % (epoch, train_loss, train_accuracy * 100, test_loss,
               test_accuracy * 100)

        print(debug_str)

    return state
# def train_DS(X, domain, label_col, epsilon, iterations: int = 10, sync_data_size: int =100, seed:int = 0):
#     rng = np.random.default_rng(seed)
#     sync_data = Dataset.synthetic_rng(domain, sync_data_size, rng)
#     X_sync = sync_data.to_numpy()
#     ml_stats = []
#     for it in range(iterations):
#         model = train_ML(X_sync, label_col)
#         # Save answers
#         ml_stats.append(model.get_error(X, label_col))


from toy_datasets.classification import get_classification
if __name__ == "__main__":

    # data = get_classification(DATA_SIZE=1000, d=2, seed=0)

    rng = np.random.default_rng(0)
    label_col = 'label'
    domain = Domain(['f1', 'f2', label_col], shape=[1, 1, 2])
    data = Dataset.synthetic_rng(domain=domain, N=1000, rng=rng)
    X = data.drop([label_col]).to_onehot()
    Y = data.project([label_col]).to_numpy()

    train_ds = {
        'features' : X,
        'label' : Y
    }
    # train_ML(data, label_col='label')
    train_and_evaluate(train_ds, epochs=100, batch_size=32)