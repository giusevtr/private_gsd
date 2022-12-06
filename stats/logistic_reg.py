import jax.numpy as jnp
import jax.random
from jax import grad, jit, vmap
from jax import random
from jax import value_and_grad



def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)

# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = jnp.array([True, True, False, True])

# Training loss is the negative log-likelihood of the training examples.

# Initialize random model coefficients
def get_loss_01(W, b, X, Y):
    n, d = X.shape
    preds = predict(W, b, X)
    preds2 = preds > 0.5
    return jnp.sum(preds2 != Y) / n

def train_lg(X, Y):
    iterations = 100
    lr=0.90
    key = jax.random.PRNGKey(0)
    key, W_key, b_key = random.split(key, 3)


    n,d = X.shape
    def loss(W, b):
        preds = predict(W, b, X)
        label_probs = preds * Y + (1 - preds) * (1 - Y)
        return -jnp.sum(jnp.log(label_probs)) / n

    W = random.normal(W_key, (2,))
    b = random.normal(b_key, ())
    init_loss = loss(W, b)
    for it in range(iterations):
        loss_value, (W_g, b_g) = value_and_grad(loss, (0, 1))(W, b)
        W = W - lr * W_g
        b = b - lr * b_g

    return W, b, init_loss, loss(W, b)


key = random.PRNGKey(0)

# train_lg(key, iterations=30, lr=0.1)


from stats import Statistic
class LogRegQuery(Statistic):
    def __init__(self, domain, label_col):

        super().__init__(domain, 'Logistic Regression')
        self.domain = domain
        self.label_col = label_col

        self.train_jit = jax.jit(train_lg)

    def get_sensitivity(self):
        return 1

    def get_stats_fn(self):
        def stat_fn(X):
            feat = X[:, :self.label_col]
            label = X[:, self.label_col] == 1
            W, b, init_loss, final_loss = train_lg(feat, label)
            params = jnp.concatenate((W, b.reshape((1, ))))

            loss = get_loss_01(W, b, feat, label)
            return params
        return stat_fn


    def get_differentiable_stats_fn(self):
        pass



from toy_datasets.classification import get_classification
def test():

    DIM = 2
    data = get_classification(DATA_SIZE=1000, d=DIM)
    X = jnp.array(data.to_numpy())

    q = LogRegQuery(data.domain, DIM)

    stat_fn = q.get_stats_fn()

    print(stat_fn(X))


if __name__ == "__main__":
    test()