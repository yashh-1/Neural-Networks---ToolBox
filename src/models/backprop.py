import random
import math

def random_weight_xavier(fan_in, fan_out):
    """Xavier initialization for better convergence"""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return round(random.uniform(-limit, limit), 6)

lr = 0.02  # Reduced from 0.01 for stability
epochs = 500

def train_network(X, Y, l_rate=None, n_epochs=None):
    # Xavier initialization: 2 inputs + bias -> 2 hidden
    w1=random_weight_xavier(3, 2);w2=random_weight_xavier(3, 2);w3=random_weight_xavier(3, 2);w4=random_weight_xavier(3, 2)
    # 2 hidden -> 1 output
    w5=random_weight_xavier(2, 1);w6=random_weight_xavier(2, 1)
    bh1=0.0;bh2=0.0;bo=0.0

    w1_init=w1;w2_init=w2;w3_init=w3;w4_init=w4;w5_init=w5;w6_init=w6

    l_rate = l_rate if l_rate is not None else lr
    n_epochs = n_epochs if n_epochs is not None else epochs

    loss_history = []  # store loss for each epoch

    for epoch in range(n_epochs):
        total_error = 0
        for i in range(len(X)):
            x1 = X[i][0]
            x2 = X[i][1]

            # forward pass
            zh1 = x1 * w1 + x2 * w3 + bh1
            zh2 = x1 * w2 + x2 * w4 + bh2

            h1 = 1/(1+(2.71828**(-zh1)))
            h2 = 1/(1+(2.71828**(-zh2)))

            zo = h1 * w5 + h2 * w6 + bo
            o = 1/(1+(2.71828**(-zo)))

            # error calculation
            error = Y[i] - o
            total_error = total_error + (error**2)

            # backpropagation
            delta_o = error * o * (1 - o)
            delta_h1 = delta_o * w5 * h1 * (1 - h1)
            delta_h2 = delta_o * w6 * h2 * (1 - h2)

            # update output weights
            w5 = w5 + l_rate * delta_o * h1
            w6 = w6 + l_rate * delta_o * h2
            bo = bo + l_rate * delta_o

            # update hidden weights
            w1 = w1 + l_rate * delta_h1 * x1
            w3 = w3 + l_rate * delta_h1 * x2
            bh1 = bh1 + l_rate * delta_h1

            w2 = w2 + l_rate * delta_h2 * x1
            w4 = w4 + l_rate * delta_h2 * x2
            bh2 = bh2 + l_rate * delta_h2

        loss_history.append(total_error)  # save loss

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Total Error: {total_error:.4f}')

    weights = (w1, w2, w3, w4, w5, w6, bh1, bh2, bo)
    init_weights = (w1_init, w2_init, w3_init, w4_init, w5_init, w6_init, 0.0, 0.0, 0.0)
    return weights, loss_history, init_weights


def predict(x1, x2, weights):
    w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights

    zh1 = x1 * w1 + x2 * w3 + bh1
    zh2 = x1 * w2 + x2 * w4 + bh2

    h1 = 1/(1+(2.71828**(-zh1)))
    h2 = 1/(1+(2.71828**(-zh2)))

    zo = h1 * w5 + h2 * w6 + bo
    o = 1/(1+(2.71828**(-zo)))

    predicted_class = 1 if o >= 0.5 else 0
    return o, predicted_class



DEFAULT_LR = lr
DEFAULT_EPOCHS = epochs
