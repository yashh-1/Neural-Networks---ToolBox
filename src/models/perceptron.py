import numpy as np

DEFAULT_LR = 1.0
DEFAULT_EPOCHS = 10


def train_perceptron(X, Y, lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS):
    # Add bias column (always 1) to inputs
    inputs = np.array([[x[0], x[1], 1] for x in X])

    # Convert binary targets (0/1) to bipolar (1/-1)
    targets = np.array([1 if y == 1 else -1 for y in Y])

    weights = np.array([0.0, 0.0, 0.0])
    learning_rate = lr

    loss_history = []

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(inputs)):
            x = inputs[i]
            t = targets[i]

            Y_in = np.dot(x, weights)

            if Y_in > 0:
                Y_out = 1
            else:
                Y_out = 0

            is_match = (t == 1 and Y_out == 1) or (t == -1 and Y_out == 0)

            if not is_match:
                weights = weights + learning_rate * (t * x)
                total_error += 1

        loss_history.append(total_error)

    return (weights[0], weights[1], weights[2]), loss_history


def predict_perceptron(x1, x2, weights):
    w1, w2, b = weights
    x = np.array([x1, x2, 1])
    w = np.array([w1, w2, b])

    Y_in = np.dot(x, w)

    if Y_in > 0:
        Y_out = 1
    else:
        Y_out = 0

    return Y_in, Y_out
