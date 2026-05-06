import random

DEFAULT_LR = 0.001
DEFAULT_EPOCHS = 100

# ─── Single Variable Linear Regression (MSE) ───────────────
def predict_single(x, w, b):
    return w * x + b


def mse_loss_single(X, y, w, b):
    total = 0
    for i in range(len(X)):
        y_pred = predict_single(X[i], w, b)
        total += (y_pred - y[i]) ** 2
    return total / len(X)


def train_mse_single(X, y, learning_rate=0.1, epochs=100):
    w = random.random()
    b = random.random()
    init_w, init_b = w, b
    loss_history = []

    for epoch in range(epochs):
        dw = 0
        db = 0

        for i in range(len(X)):
            y_pred = predict_single(X[i], w, b)
            error = y_pred - y[i]
            dw += error * X[i]
            db += error

        dw = (2 / len(X)) * dw
        db = (2 / len(X)) * db

        w = w - learning_rate * dw
        b = b - learning_rate * db

        loss = mse_loss_single(X, y, w, b)
        loss_history.append(loss)

    return w, b, init_w, init_b, loss_history


# ─── Two Variable Linear Regression (MSE) ──────────────────
def predict_dual(x1, x2, w1, w2, b):
    return w1 * x1 + w2 * x2 + b


def mse_loss_dual(X1, X2, y, w1, w2, b):
    total = 0
    for i in range(len(X1)):
        y_pred = predict_dual(X1[i], X2[i], w1, w2, b)
        total += (y_pred - y[i]) ** 2
    return total / len(X1)


def train_mse_dual(X1, X2, y, learning_rate=0.001, epochs=100):
    w1 = random.random()
    w2 = random.random()
    b = random.random()
    init_w1, init_w2, init_b = w1, w2, b
    loss_history = []

    for epoch in range(epochs):
        dw1 = 0
        dw2 = 0
        db = 0

        for i in range(len(X1)):
            y_pred = predict_dual(X1[i], X2[i], w1, w2, b)
            error = y_pred - y[i]
            dw1 += error * X1[i]
            dw2 += error * X2[i]
            db += error

        dw1 = (2 / len(X1)) * dw1
        dw2 = (2 / len(X1)) * dw2
        db = (2 / len(X1)) * db

        w1 = w1 - learning_rate * dw1
        w2 = w2 - learning_rate * dw2
        b = b - learning_rate * db

        loss = mse_loss_dual(X1, X2, y, w1, w2, b)
        loss_history.append(loss)

    return w1, w2, b, init_w1, init_w2, init_b, loss_history
